"""``make gnn-data``: single-leak scenarios for the GNN localizer (PRD §4.2, §8.2; D-024).

Each scenario runs L-Town for two hours from a chosen time of day (EPANET ``pattern_start``) with
pressure-dependent demand, a random ±``demand_noise`` factor on every junction's base demand, and
one leak at a random junction from the first step. The leak is an EPANET emitter with exponent 0.5,
``q = Cd·A·sqrt(2·g·h)`` — the same orifice law as WNTR's leak model — run with WNTR's
EpanetSimulator, which is far faster than WNTRSimulator (decision D-024).

Inputs follow PRD §8.2 *train/serve consistency*: a pressure residual is the reading (plus
Gaussian sensor noise) minus a baseline built like the real one — the median of ``baseline_runs``
leak-free runs with their own demand noise, from the same time of day. Per sensor, residuals are
averaged over the first and the second hour of the leak: two values per sensor, like the
``residual`` statistic of two consecutive 1 h feature windows. The label is the leak junction.

Writes ``data/sim/gnn/{dataset,graph}.npz`` and ``reports/metrics/gnn_dataset.json``.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics

log = get_logger("leakwatch.dl.gnn_data")
G = 9.81
BASE_SEED = 900_000

_STATE: dict[str, Any] = {}  # per worker process: network template, config, baselines


def _init(inp: str, cfg: dict[str, Any], baselines: dict[int, np.ndarray] | None = None) -> None:
    import wntr

    wn = wntr.network.WaterNetworkModel(inp)
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.emitter_exponent = 0.5
    wn.options.time.duration = 2 * 3600
    wn.options.time.hydraulic_timestep = int(cfg["step_s"])
    wn.options.time.report_timestep = int(cfg["step_s"])
    wn.options.time.pattern_timestep = wn.options.time.pattern_timestep or 3600
    _STATE.update({"wn": wn, "cfg": cfg, "baselines": baselines or {}})


def run_pressures(seed: int, start_s: int, leak: tuple[str, float] | None = None) -> np.ndarray:
    """Sensor pressures (steps × sensors) for one 2 h run with its own demand noise."""
    import wntr

    cfg = _STATE["cfg"]
    wn = copy.deepcopy(_STATE["wn"])
    rng = np.random.default_rng(seed)
    noise = float(cfg["demand_noise"])
    for _, junction in wn.junctions():
        for demand in junction.demand_timeseries_list:
            demand.base_value = demand.base_value * rng.uniform(1 - noise, 1 + noise)
    wn.options.time.pattern_start = int(start_s)
    if leak is not None:
        node, diameter = leak
        area = math.pi * (diameter / 2) ** 2
        wn.get_node(node).emitter_coefficient = (
            float(cfg["discharge_coeff"]) * area * math.sqrt(2 * G)
        )
    # A unique scratch prefix per run: parallel workers must not share temp.inp/.bin/.rpt.
    folder = Path(tempfile.gettempdir()) / "leakwatch-epanet"
    folder.mkdir(exist_ok=True)
    prefix = folder / f"gnn-{os.getpid()}-{seed}"
    try:
        results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(prefix))
    finally:
        for ext in (".inp", ".bin", ".rpt"):
            prefix.with_suffix(ext).unlink(missing_ok=True)
    return results.node["pressure"][list(cfg["sensors"])].to_numpy(dtype="float64")


def baseline(start_s: int) -> tuple[int, np.ndarray]:
    """Median of several noisy leak-free runs from one time of day."""
    n = int(_STATE["cfg"]["baseline_runs"])
    runs = [run_pressures(BASE_SEED + 97 * k + start_s, start_s) for k in range(n)]
    return start_s, np.median(np.stack(runs), axis=0)


def scenario(args: tuple[int, str, float, int]) -> dict[str, Any]:
    """Residuals of one leak run: (sensors, 2) = mean over the 1st and 2nd hour of the leak."""
    seed, node, diameter, start_s = args
    t0 = time.perf_counter()
    cfg = _STATE["cfg"]
    p = run_pressures(seed, start_s, (node, diameter))
    p = p + np.random.default_rng(seed + 7).normal(0, float(cfg["sensor_noise_m"]), p.shape)
    resid = p - _STATE["baselines"][start_s]
    per_hour = 3600 // int(cfg["step_s"])
    w1 = resid[1 : per_hour + 1].mean(axis=0)
    w2 = resid[per_hour + 1 : 2 * per_hour + 1].mean(axis=0)
    return {
        "x": np.stack([w1, w2], axis=1).astype("float32"),
        "node": node,
        "diameter_m": diameter,
        "start_s": start_s,
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
    n: int, junctions: Sequence[str], cfg: Mapping[str, Any], seed: int
) -> list[tuple[int, str, float, int]]:
    """Random (seed, junction, leak diameter (log-uniform), start time of day) per scenario."""
    rng = np.random.default_rng(seed)
    lo, hi = (float(v) for v in cfg["leak_diameter_m"])
    starts = [int(h * 3600) for h in cfg["start_hours"]]
    return [
        (
            int(seed * 1_000 + k),
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
    todo = draw_scenarios(n, junctions, gcfg, int(cfg["project"]["seed"]))
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=(inp, gcfg)) as pool:
        baselines = dict(pool.map(baseline, sorted({s[3] for s in todo})))
    t_base = time.perf_counter() - t0
    with ctx.Pool(workers, initializer=_init, initargs=(inp, gcfg, baselines)) as pool:
        rows = pool.map(scenario, todo, chunksize=8)
    wall = time.perf_counter() - t0

    graph = graph_arrays(wn, sensors)
    index = {name: i for i, name in enumerate(graph["nodes"])}
    x = np.stack([r["x"] for r in rows])
    out_dir = paths.ensure_dir(paths.sim_dir() / "gnn")
    np.savez_compressed(
        out_dir / "dataset.npz",
        x=x,
        label=np.array([index[r["node"]] for r in rows], dtype="int64"),
        diameter_m=np.array([r["diameter_m"] for r in rows], dtype="float32"),
        start_s=np.array([r["start_s"] for r in rows], dtype="int64"),
    )
    np.savez_compressed(out_dir / "graph.npz", **graph)
    payload = {
        "engine": "WNTR EpanetSimulator (PDD) with emitter leaks (orifice law, D-024)",
        "scenarios": len(rows),
        "junctions": len(junctions),
        "sensors": len(sensors),
        "leak_diameter_m": gcfg["leak_diameter_m"],
        "start_hours": gcfg["start_hours"],
        "baseline_runs_per_start": int(gcfg["baseline_runs"]),
        "demand_noise": gcfg["demand_noise"],
        "sensor_noise_m": gcfg["sensor_noise_m"],
        "wall_s": round(wall, 1),
        "baseline_wall_s": round(t_base, 1),
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
