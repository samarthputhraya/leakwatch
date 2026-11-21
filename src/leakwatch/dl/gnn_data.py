"""``make gnn-data``: leak scenarios for the GNN localizer (PRD §4.2, §8.2; D-024, D-038).

Each scenario runs L-Town for two hours from a chosen time of day (EPANET ``pattern_start``) and
one new leak at a random junction from the first step. The leak is an EPANET emitter with exponent
0.5, ``q = Cd·A·sqrt(2·g·h)`` — the same orifice law as WNTR's leak model — run with WNTR's
EpanetSimulator, which is far faster than WNTRSimulator (D-024).

The network a scenario runs on is one of ``worlds`` perturbed copies of the model (D-038), so the
localizer learns what the real network looks like, not only the nominal model: every pipe's
roughness times U(1 − r, 1 + r) (``roughness_noise``) and a few small *background* leaks already
running (``background_leaks``, like the leaks that pile up in BattLeDIM 2019). Every run also
draws its own demand: each junction's base demand times U(1 ± ``demand_noise``) and the whole
network times N(1, ``day_demand_noise``) (day-to-day variation).

Inputs follow PRD §8.2 *train/serve consistency*: a pressure residual is the reading (plus
Gaussian sensor noise) minus a reference built like the real one — the median of
``reference_runs`` runs of the same world without the new leak (each with its own demand draw),
from the same time of day; live, the reference is the sensor's recent history, which already holds
the older leaks. Per sensor, residuals are averaged over the first and the second hour of the leak:
two values per sensor, like two consecutive 1 h feature windows. The label is the leak junction.

``sensitivity`` (D-038): for every junction, the mean pressure change at the sensors over the
first two hours of a ``sensitivity_diameter_m`` leak on the nominal model (start hours
``sensitivity_start_hours``), stored in ``graph.npz``; the localizer correlates the observed
residuals with it (the classic sensitivity-matrix method).

Writes ``data/sim/gnn/{dataset,graph}.npz`` and ``reports/metrics/gnn_dataset.json``.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics

log = get_logger("leakwatch.dl.gnn_data")
G = 9.81
BASE_SEED = 900_000

_STATE: dict[str, Any] = {}  # per worker process: network template, config, worlds, references


def _init(
    inp: str,
    cfg: dict[str, Any],
    references: dict[tuple[int, int], np.ndarray] | None = None,
    worlds: list[dict[str, Any]] | None = None,
) -> None:
    import wntr

    # The EPANET toolkit writes its own temp files ("en" + 6 chars) into the working directory;
    # each worker process works inside its private scratch directory, never the repository.
    os.chdir(paths.private_scratch_dir())
    wn = wntr.network.WaterNetworkModel(inp)
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.emitter_exponent = 0.5
    wn.options.time.duration = 2 * 3600
    wn.options.time.hydraulic_timestep = int(cfg["step_s"])
    wn.options.time.report_timestep = int(cfg["step_s"])
    wn.options.time.pattern_timestep = wn.options.time.pattern_timestep or 3600
    _STATE.update({"wn": wn, "cfg": cfg, "references": references or {}, "worlds": worlds or []})


def emitter(diameter_m: float, cd: float) -> float:
    """EPANET emitter coefficient of an orifice leak (exponent 0.5): Cd·A·sqrt(2g)."""
    return cd * math.pi * (diameter_m / 2) ** 2 * math.sqrt(2 * G)


def make_worlds(
    n: int, junctions: Sequence[str], pipes: Sequence[str], cfg: Mapping[str, Any], seed: int
) -> list[dict[str, Any]]:
    """Perturbed networks: roughness factor per pipe and background leaks (node, diameter)."""
    rng = np.random.default_rng(seed + 31)
    r = float(cfg.get("roughness_noise", 0.0))
    lo_n, hi_n = (int(v) for v in cfg.get("background_leaks", [0, 0]))
    lo_d, hi_d = (float(v) for v in cfg.get("background_diameter_m", [0.003, 0.012]))
    worlds = []
    for _ in range(n):
        k = int(rng.integers(lo_n, hi_n + 1))
        factors = rng.uniform(1 - r, 1 + r, len(pipes))
        worlds.append(
            {
                "roughness": dict(zip(pipes, factors.tolist(), strict=True)),
                "background": [
                    (str(rng.choice(junctions)), float(rng.uniform(lo_d, hi_d))) for _ in range(k)
                ],
            }
        )
    return worlds


def run_pressures(
    seed: int,
    start_s: int,
    leak: tuple[str, float] | Sequence[tuple[str, float]] | None = None,
    world: int | None = None,
    noisy: bool = True,
) -> np.ndarray:
    """Sensor pressures (steps × sensors) for one 2 h run with its own demand draw.

    ``leak`` is one (junction, diameter) or several (D-051: overlapping leaks); ``world`` applies
    that perturbed network (roughness, background leaks); ``noisy=False`` keeps
    the nominal demands (sensitivity runs).
    """
    import wntr

    cfg = _STATE["cfg"]
    wn = copy.deepcopy(_STATE["wn"])
    rng = np.random.default_rng(seed)
    if noisy:
        noise = float(cfg["demand_noise"])
        day = max(0.5, rng.normal(1.0, float(cfg.get("day_demand_noise", 0.0))))
        for _, junction in wn.junctions():
            for demand in junction.demand_timeseries_list:
                demand.base_value = demand.base_value * day * rng.uniform(1 - noise, 1 + noise)
    wn.options.time.pattern_start = int(start_s)
    cd = float(cfg["discharge_coeff"])
    coeff: dict[str, float] = {}
    if world is not None:
        w = _STATE["worlds"][world]
        for name, factor in w["roughness"].items():
            pipe = wn.get_link(name)
            pipe.roughness = pipe.roughness * float(factor)
        for node, diameter in w["background"]:
            coeff[node] = coeff.get(node, 0.0) + emitter(diameter, cd)
    leaks = [] if leak is None else [leak] if isinstance(leak[0], str) else list(leak)
    for node, diameter in leaks:  # type: ignore[misc]
        coeff[str(node)] = coeff.get(str(node), 0.0) + emitter(float(diameter), cd)
    for node, c in coeff.items():
        wn.get_node(node).emitter_coefficient = c
    # A unique scratch prefix per run: parallel workers must not share temp.inp/.bin/.rpt.
    prefix = paths.private_scratch_dir() / f"gnn-{seed}"
    try:
        results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(prefix))
    finally:
        for ext in (".inp", ".bin", ".rpt"):
            prefix.with_suffix(ext).unlink(missing_ok=True)
    return results.node["pressure"][list(cfg["sensors"])].to_numpy(dtype="float64")


def reference(key: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
    """Median of several leak-free runs of one world from one time of day."""
    world, start_s = key
    n = int(_STATE["cfg"]["reference_runs"])
    runs = [
        run_pressures(BASE_SEED + 97 * k + start_s + 100_003 * world, start_s, world=world)
        for k in range(n)
    ]
    return key, np.median(np.stack(runs), axis=0)


def sensitivity(node: str) -> tuple[str, np.ndarray]:
    """Mean sensor pressure change over 2 h for a leak at ``node`` (nominal model, no noise)."""
    cfg = _STATE["cfg"]
    d = float(cfg["sensitivity_diameter_m"])
    rows = []
    for h in cfg["sensitivity_start_hours"]:
        s = int(h) * 3600
        base = _STATE["references"].get((-1, s))
        if base is None:
            base = run_pressures(0, s, None, noisy=False)
            _STATE["references"][(-1, s)] = base
        rows.append((run_pressures(0, s, (node, d), noisy=False) - base)[1:].mean(axis=0))
    return node, np.mean(rows, axis=0)


def scenario(args: tuple[int, int, str, float, int]) -> dict[str, Any]:
    """Residuals of one leak run: (sensors, 2) = mean over the 1st and 2nd hour of the leak."""
    seed, world, node, diameter, start_s = args
    t0 = time.perf_counter()
    cfg = _STATE["cfg"]
    p = run_pressures(seed, start_s, (node, diameter), world=world)
    p = p + np.random.default_rng(seed + 7).normal(0, float(cfg["sensor_noise_m"]), p.shape)
    resid = p - _STATE["references"][(world, start_s)]
    per_hour = 3600 // int(cfg["step_s"])
    w1 = resid[1 : per_hour + 1].mean(axis=0)
    w2 = resid[per_hour + 1 : 2 * per_hour + 1].mean(axis=0)
    return {
        "x": np.stack([w1, w2], axis=1).astype("float32"),
        "node": node,
        "diameter_m": diameter,
        "start_s": start_s,
        "world": world,
        "wall_s": time.perf_counter() - t0,
    }


def graph_arrays(wn: Any, sensors: Sequence[str]) -> dict[str, np.ndarray]:
    """Static graph: nodes (junctions, tanks, reservoirs), undirected edges, node/edge features."""
    nodes = list(wn.node_name_list)
    index = {n: i for i, n in enumerate(nodes)}
    src, dst, length, diameter = [], [], [], []
    for _, link in wn.links():
        a, b = index[link.start_node_name], index[link.end_node_name]
        is_pipe = link.link_type == "Pipe"
        ln = float(link.length) if is_pipe else 1.0
        dm = float(link.diameter) if is_pipe else 0.3
        src += [a, b]
        dst += [b, a]
        length += [ln, ln]
        diameter += [dm, dm]
    elevation, demand = [], []
    for n in nodes:
        node = wn.get_node(n)
        elevation.append(float(getattr(node, "elevation", 0.0) or 0.0))
        if node.node_type == "Junction":
            demand.append(float(sum(d.base_value for d in node.demand_timeseries_list)))
        else:
            demand.append(0.0)
    return {
        "nodes": np.array(nodes),
        "edge_index": np.array([src, dst], dtype="int64"),
        "edge_length_m": np.array(length, dtype="float32"),
        "edge_diameter_m": np.array(diameter, dtype="float32"),
        "elevation_m": np.array(elevation, dtype="float32"),
        "base_demand_m3s": np.array(demand, dtype="float32"),
        "is_junction": np.array([wn.get_node(n).node_type == "Junction" for n in nodes]),
        "sensor_idx": np.array([index[s] for s in sensors], dtype="int64"),
    }


def draw_scenarios(
    n: int, junctions: Sequence[str], cfg: Mapping[str, Any], seed: int, worlds: int = 1
) -> list[tuple[int, int, str, float, int]]:
    """Random (seed, world, junction, leak diameter (log-uniform), start time) per scenario."""
    rng = np.random.default_rng(seed)
    lo, hi = (float(v) for v in cfg["leak_diameter_m"])
    starts = [int(h * 3600) for h in cfg["start_hours"]]
    return [
        (
            int(seed * 1_000 + k),
            int(rng.integers(worlds)),
            str(rng.choice(junctions)),
            round(float(math.exp(rng.uniform(math.log(lo), math.log(hi)))), 5),
            int(rng.choice(starts)),
        )
        for k in range(n)
    ]


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset (see module docstring)."""
    parser = argparse.ArgumentParser(description="Single-leak scenarios for the GNN.")
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    cfg = load_config()
    gcfg = dict(cfg["dl"]["gnn_data"])
    n = int(args.scenarios or gcfg["scenarios"])
    workers = int(args.workers or gcfg["workers"])

    from leakwatch.data import network

    sensors = network.parse_sensor_config()["pressure"]
    gcfg["sensors"] = sensors
    inp = str(paths.network_inp_path())
    wn = network.load_network()
    junctions = sorted(wn.junction_name_list)
    seed = int(cfg["project"]["seed"])
    n_worlds = int(gcfg.get("worlds", 1))
    worlds = make_worlds(n_worlds, junctions, sorted(wn.pipe_name_list), gcfg, seed)
    todo = draw_scenarios(n, junctions, gcfg, seed, n_worlds)
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=(inp, gcfg, None, worlds)) as pool:
        sens = dict(pool.map(sensitivity, junctions, chunksize=8))
        references = dict(pool.map(reference, sorted({(s[1], s[4]) for s in todo})))
    t_base = time.perf_counter() - t0
    with ctx.Pool(workers, initializer=_init, initargs=(inp, gcfg, references, worlds)) as pool:
        rows = pool.map(scenario, todo, chunksize=8)
    wall = time.perf_counter() - t0

    graph = graph_arrays(wn, sensors)
    graph["sensitivity"] = np.stack(
        [sens.get(n, np.zeros(len(sensors))) for n in graph["nodes"]]
    ).astype("float32")
    index = {name: i for i, name in enumerate(graph["nodes"])}
    x = np.stack([r["x"] for r in rows])
    out_dir = paths.ensure_dir(paths.sim_dir() / "gnn")
    np.savez_compressed(
        out_dir / "dataset.npz",
        x=x,
        label=np.array([index[r["node"]] for r in rows], dtype="int64"),
        diameter_m=np.array([r["diameter_m"] for r in rows], dtype="float32"),
        start_s=np.array([r["start_s"] for r in rows], dtype="int64"),
        world=np.array([r["world"] for r in rows], dtype="int64"),
    )
    np.savez_compressed(out_dir / "graph.npz", **graph)
    payload = {
        "engine": "WNTR EpanetSimulator (PDD) with emitter leaks (orifice law, D-024)",
        "scenarios": len(rows),
        "junctions": len(junctions),
        "sensors": len(sensors),
        "leak_diameter_m": gcfg["leak_diameter_m"],
        "start_hours": gcfg["start_hours"],
        "worlds": n_worlds,
        "roughness_noise": gcfg.get("roughness_noise"),
        "background_leaks": gcfg.get("background_leaks"),
        "background_diameter_m": gcfg.get("background_diameter_m"),
        "reference_runs": int(gcfg["reference_runs"]),
        "references": len(references),
        "day_demand_noise": gcfg.get("day_demand_noise"),
        "sensitivity_diameter_m": gcfg["sensitivity_diameter_m"],
        "sensitivity_start_hours": gcfg["sensitivity_start_hours"],
        "demand_noise": gcfg["demand_noise"],
        "sensor_noise_m": gcfg["sensor_noise_m"],
        "wall_s": round(wall, 1),
        "reference_and_sensitivity_wall_s": round(t_base, 1),
        "scenario_wall_s_mean": round(float(np.mean([r["wall_s"] for r in rows])), 3),
        "workers": workers,
        "residual_m": {
            "abs_mean_all": round(float(np.abs(x).mean()), 4),
            "max_abs_per_scenario_median": round(float(np.median(np.abs(x).max(axis=(1, 2)))), 4),
        },
        "unique_leak_nodes": len({r["node"] for r in rows}),
        "output": str(out_dir),
    }
    path = write_metrics("gnn_dataset", payload)
    log.info("%d scenarios in %.0f s -> %s", len(rows), wall, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
