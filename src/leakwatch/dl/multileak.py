"""``make localizer-multileak``: the localizer candidates on overlapping simulated leaks (D-051).

The GLS localizer (D-044) was chosen on single-leak simulations and the 2018 leaks (3.5 running at
once on average). It assumes one new leak; when several are new at the same time the residual is
their sum. This job asks, on simulated data only, which candidate of ``dl.localizer_selection``
places a report best when ``k`` new leaks appear together.

Scenarios reuse ``dl.gnn_data`` (same perturbed networks, demand draws, sensor noise and leak-size
range): each one adds ``k`` new leaks at random junctions to one world, from one time of day, and
takes the residual against that world's leak-free reference, as in training. A report is the top
node of each candidate; it counts like an official true positive when it lies within
``within_m`` (300 m) of *any* of the new leaks.

**Rule, fixed before the first run:** the system localizer is the candidate with the highest share
of reports within 300 m of a new leak, pooled over the overlapping counts (``leak_counts`` ≥ 2);
ties go to the better best-of-5 share, then to the D-044 mean rank (``dl_localizer.json``). The
single-leak group (k = 1) is reported for context and does not vote. 2019 is not used.

Writes ``reports/metrics/dl_localizer_multileak.json``.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import read_metrics, write_metrics
from leakwatch.dl import gnn, gnn_data

log = get_logger("leakwatch.dl.multileak")


def draw(
    cfg: Mapping[str, Any], junctions: Sequence[str], n_worlds: int, seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, list[tuple[str, float]]]]]:
    """Reference pairs (world, start) and scenarios (seed, world, start, leaks) per leak count."""
    mcfg = cfg["dl"]["multileak"]
    gcfg = cfg["dl"]["gnn_data"]
    rng = np.random.default_rng(seed + int(mcfg["seed_offset"]))
    starts = [int(h * 3600) for h in gcfg["start_hours"]]
    pairs = sorted(
        {
            (int(rng.integers(n_worlds)), int(rng.choice(starts)))
            for _ in range(int(mcfg["reference_pairs"]))
        }
    )
    lo, hi = (math.log(float(v)) for v in gcfg["leak_diameter_m"])
    todo = []
    for k in (int(v) for v in mcfg["leak_counts"]):
        for i in range(int(mcfg["scenarios_per_count"])):
            world, start = pairs[int(rng.integers(len(pairs)))]
            nodes = rng.choice(junctions, size=k, replace=False)
            leaks = [(str(n), round(float(math.exp(rng.uniform(lo, hi))), 5)) for n in nodes]
            todo.append((int(seed * 10_000 + 1_000 * k + i), world, start, leaks))
    return pairs, todo


def scenario(args: tuple[int, int, int, list[tuple[str, float]]]) -> dict[str, Any]:
    """Residuals (sensors, 2) of one run with several new leaks (worker; see ``gnn_data``)."""
    seed, world, start_s, leaks = args
    cfg = gnn_data._STATE["cfg"]
    p = gnn_data.run_pressures(seed, start_s, leaks, world=world)
    p = p + np.random.default_rng(seed + 7).normal(0, float(cfg["sensor_noise_m"]), p.shape)
    resid = p - gnn_data._STATE["references"][(world, start_s)]
    per_hour = 3600 // int(cfg["step_s"])
    w1 = resid[1 : per_hour + 1].mean(axis=0)
    w2 = resid[per_hour + 1 : 2 * per_hour + 1].mean(axis=0)
    return {"x": np.stack([w1, w2], axis=1).astype("float32"), "leaks": leaks}


def report_metrics(
    dist: np.ndarray,
    scores: np.ndarray,
    leak_idx: Sequence[Sequence[int]],
    largest: Sequence[int],
    within: float,
) -> dict[str, Any]:
    """Shares of reports within ``within`` m of any new leak (top-1, best of 5), of the largest."""
    top = np.argsort(-scores, axis=1)[:, :5]
    any1, any5, big1, d1 = [], [], [], []
    for i, leaks in enumerate(leak_idx):
        dl = dist[np.ix_(top[i], list(leaks))]  # (5, k)
        any1.append(bool(dl[0].min() <= within))
        any5.append(bool(dl.min() <= within))
        big1.append(bool(dist[top[i, 0], largest[i]] <= within))
        d1.append(float(dl[0].min()))
    return {
        "n": len(leak_idx),
        "within_m_of_a_leak_top1": round(float(np.mean(any1)), 4),
        "within_m_of_a_leak_best_of_top5": round(float(np.mean(any5)), 4),
        "within_m_of_the_largest_top1": round(float(np.mean(big1)), 4),
        "median_distance_to_nearest_leak_top1_m": round(float(np.median(d1)), 1),
    }


def choose(pooled: Mapping[str, Mapping[str, float]], mean_rank: Mapping[str, float]) -> str:
    """The D-051 rule: pooled top-1 share, then best-of-5 share, then the D-044 mean rank."""
    return max(
        pooled,
        key=lambda n: (
            round(float(pooled[n]["within_m_of_a_leak_top1"]), 4),
            round(float(pooled[n]["within_m_of_a_leak_best_of_top5"]), 4),
            -float(mean_rank.get(n, 99.0)),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Simulate overlapping leaks, score every candidate, apply the rule, write metrics."""
    parser = argparse.ArgumentParser(description="Localizer candidates on overlapping leaks.")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    cfg = load_config()
    gcfg = dict(cfg["dl"]["gnn_data"])
    mcfg = cfg["dl"]["multileak"]
    seed = int(cfg["project"]["seed"])
    within = float(cfg["dl"]["gnn"]["within_m"])
    from leakwatch.data import network

    sensors = network.parse_sensor_config()["pressure"]
    gcfg["sensors"] = sensors
    wn = network.load_network()
    junctions = sorted(wn.junction_name_list)
    n_worlds = int(gcfg.get("worlds", 1))
    worlds = gnn_data.make_worlds(n_worlds, junctions, sorted(wn.pipe_name_list), gcfg, seed)
    pairs, todo = draw(cfg, junctions, n_worlds, seed)
    workers = int(args.workers or mcfg["workers"])
    inp = str(paths.network_inp_path())
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=gnn_data._init, initargs=(inp, gcfg, None, worlds)) as pool:
        references = dict(pool.map(gnn_data.reference, pairs))
    with ctx.Pool(
        workers, initializer=gnn_data._init, initargs=(inp, gcfg, references, worlds)
    ) as pool:
        rows = pool.map(scenario, todo, chunksize=4)
    sim_s = time.perf_counter() - t0

    loaded = gnn.load_localizer()
    if loaded is None:
        raise FileNotFoundError("no saved localizer: run `make gnn` first")
    g = loaded[0]
    x = np.stack([r["x"] for r in rows])
    counts = np.array([len(r["leaks"]) for r in rows])
    leak_idx = [[g.index[n] for n, _ in r["leaks"]] for r in rows]
    largest = [g.index[max(r["leaks"], key=lambda t: t[1])[0]] for r in rows]
    lsel = cfg["dl"]["localizer_selection"]
    candidates = {k: dict(v) for k, v in lsel["candidates"].items()}
    for v in candidates.values():
        v.setdefault("interpolation_m", float(lsel["interpolation_m"]))
    voting = counts >= 2
    results: dict[str, Any] = {}
    pooled: dict[str, dict[str, float]] = {}
    for name, lcfg in candidates.items():
        sc = gnn.SystemLocalizer(lcfg, loaded=loaded).scores(x)
        by_k = {
            str(k): report_metrics(
                g.dist,
                sc[counts == k],
                [leak_idx[i] for i in np.flatnonzero(counts == k)],
                [largest[i] for i in np.flatnonzero(counts == k)],
                within,
            )
            for k in sorted(set(counts.tolist()))
        }
        pooled[name] = report_metrics(
            g.dist,
            sc[voting],
            [leak_idx[i] for i in np.flatnonzero(voting)],
            [largest[i] for i in np.flatnonzero(voting)],
            within,
        )
        results[name] = {"by_leak_count": by_k, "pooled_overlapping": pooled[name]}
        log.info("%-16s pooled k>=2: %s", name, pooled[name])
    try:
        mean_rank = {
            n: float(c["mean_rank"]) for n, c in read_metrics("dl_localizer")["candidates"].items()
        }
    except (FileNotFoundError, KeyError):
        mean_rank = {}
    chosen = choose(pooled, mean_rank)
    configured = gnn.localizer_config(cfg)
    payload = {
        "question": "which localizer candidate places a report best when several new leaks "
        "appear together (simulated, 2019 not used)?",
        "rule": "highest share of top-1 reports within 300 m of any new leak, pooled over leak "
        "counts >= 2; ties: best-of-5 share, then the D-044 mean rank. Fixed before the first run",
        "scenarios": len(rows),
        "leak_counts": [int(v) for v in mcfg["leak_counts"]],
        "scenarios_per_count": int(mcfg["scenarios_per_count"]),
        "reference_pairs": len(pairs),
        "leak_diameter_m": gcfg["leak_diameter_m"],
        "within_m": within,
        "candidates": results,
        "d044_mean_rank": mean_rank,
        "chosen": chosen,
        "chosen_config": candidates[chosen],
        "configured": configured,
        "config_matches_choice": all(
            configured.get(k) == v for k, v in candidates[chosen].items() if k != "interpolation_m"
        ),
        "simulation_wall_s": round(sim_s, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    path = write_metrics("dl_localizer_multileak", payload)
    log.info("chosen %s (config matches: %s) -> %s", chosen, payload["config_matches_choice"], path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
