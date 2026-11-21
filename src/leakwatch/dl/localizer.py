"""``make localizer``: model-based inversion, cue fusion and the system localizer's choice (D-044).

1. **Inversion models** (``dl.inversion``): the GLS fit against the sensitivity matrix with two
   noise covariances — ``sim`` from the residuals of the simulated *training* scenarios after their
   true leak's fit (ridge chosen on the validation scenarios, medium + large top-5), and ``real``
   from the re-baselined pressure residuals of the 2018 normal windows (D-015), paired exactly like
   the localizer input. Both are saved under ``data/models/localizer/``.
2. **Held-out simulated leaks** (the ``make gnn`` split): the PRD §10 metric — top-5 accuracy of the
   exact leak junction on *medium + large* leaks (≥ ``size_buckets_m[0]``) — with a 95% Wilson
   interval, next to the all-size number and the within-300 m numbers, for every cue alone and for
   every candidate in ``dl.localizer_candidates``.
3. **Choice on simulations and 2018 only.** Each candidate gets three development measures: the
   PRD metric on held-out simulations, the share of the 14 recorded 2018 leaks with the top-1 node
   within 300 m (oracle time, ``gnn.real_leak_benchmark``), and the official score of one report
   per 2018 batch alert at its top node. The system uses the best mean rank (ties: the 2018 official
   score). The official score alone (the D-041/D-042 rule) swings by one alert on 2018, so it is
   one vote of three; its own choice is reported. 2019 is computed after the choice, for the chosen
   candidate, that alternative and the previous system only.

Writes ``reports/metrics/dl_localizer.json`` and ``data/models/localizer/inversion_{sim,real}.npz``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.common import lake, paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.dl import gnn
from leakwatch.dl.inversion import (
    Inversion,
    describe,
    fit_residuals,
    interpolation_scores,
    inversion_path,
    real_noise_windows,
    second_moment,
)

log = get_logger("leakwatch.dl.localizer")


def wilson(p: float, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval of a proportion ``p`` from ``n`` trials."""
    if n == 0:
        return [math.nan, math.nan]
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return [round(mid - half, 4), round(mid + half, 4)]


def sim_metrics(
    g: gnn.GraphData,
    scores: np.ndarray,
    y: np.ndarray,
    diam: np.ndarray,
    edges: list[float],
    within: float,
) -> dict[str, Any]:
    """All sizes, per size bucket, and the PRD medium + large bucket (with a Wilson interval)."""
    ml = diam >= edges[0]
    prd = gnn.distance_metrics(g, scores[ml], y[ml], within)
    prd["top5_ci95"] = wilson(prd["top5"], prd["n"])
    return {
        "all_sizes": gnn.distance_metrics(g, scores, y, within),
        "by_size": gnn.by_bucket(g, scores, y, diam, edges, within),
        "medium_large": prd,
    }


def choose_ridge(
    xm: np.ndarray,
    y: np.ndarray,
    diam: np.ndarray,
    s: np.ndarray,
    split: tuple[np.ndarray, np.ndarray],
    g: gnn.GraphData,
    grid: list[float],
    min_diam: float,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Ridge with the best validation top-5 on medium + large leaks (ties: the larger ridge)."""
    tr, va = split
    cov = second_moment(fit_residuals(xm[tr], s[y[tr]]))
    vml = va[diam[va] >= min_diam]
    by_ridge: dict[str, float] = {}
    best_r, best_v = grid[0], -1.0
    for r in sorted(grid):
        inv = Inversion(s, cov, r, g.is_junction)
        v = gnn.topk_accuracy(inv.loglik(xm[vml]), y[vml], 5)
        by_ridge[f"{r:g}"] = round(v, 4)
        if v >= best_v - 1e-12:
            best_r, best_v = r, v
    return best_r, by_ridge, cov


def real_covariance(cfg: Mapping[str, Any], sensors: list[str]) -> tuple[np.ndarray, int]:
    """Second moment of 2018 normal-window localizer inputs (re-baselined residuals, D-015)."""
    from leakwatch.data.ground_truth import load_ground_truth
    from leakwatch.features.normal import intervals_from_config, window_normal_mask

    feats = gnn.rebaselined_pressure(sensors)
    feats = feats[feats.index.year == int(cfg["bdt"]["train_year"])]
    cols = [f"pressure__{s}__residual" for s in sensors]
    starts = pd.to_datetime(feats["window_start"], utc=True).reset_index(drop=True)
    window_s = int(cfg["features"]["window_minutes"]) * 60
    step_s = int(cfg["cleaning"]["step_seconds"])
    iv = intervals_from_config(load_ground_truth(), cfg["normal"])
    normal = window_normal_mask(starts, window_s, step_s, iv)
    x = real_noise_windows(feats, cols, normal)
    x = x[~np.isnan(x).any(axis=1)]
    return second_moment(x), len(x)


def official_placement(
    loc: gnn.SystemLocalizer,
    alerts: pd.DataFrame,
    year: int,
    scorer: Any,
    sensors: list[str],
) -> dict[str, Any]:
    """Official score of one report per alert at the localizer's top node (placement only)."""
    from leakwatch.eval.battledim_score import year_bounds

    first, last = year_bounds(year)
    a = alerts[(alerts["start_ts"] >= first) & (alerts["start_ts"] <= last)]
    x = gnn.benchmark_inputs(loc.g, list(a["start_ts"]), sensors)
    top = np.argmax(loc.scores(x), axis=1)
    rows = [
        {"ts": ts, "link_id": link}
        for ts, j in zip(a["start_ts"], top, strict=True)
        if (link := scorer.dist.link_at_node(loc.g.nodes[j])) is not None
    ]
    r = scorer.score(pd.DataFrame(rows, columns=["ts", "link_id"]), year)
    keep = ("total_score_eur", "true_positives", "false_positives")
    return {k: r[k] for k in keep} | {"alerts": len(a)}


#: Development measures the choice averages ranks over (all higher-is-better).
MEASURES = ("sims_medium_large_top5", "recorded_2018_within_300m", "official_2018_eur")


def measures(cand: Mapping[str, Any]) -> dict[str, float]:
    """The three development measures of one candidate."""
    return {
        "sims_medium_large_top5": float(cand["sims"]["medium_large"]["top5"]),
        "recorded_2018_within_300m": float(cand["recorded_2018"]["within_m_top1"]),
        "official_2018_eur": float(cand["official_2018"]["total_score_eur"]),
    }


def mean_ranks(candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Mean rank (1 = best, ties share the better rank) over :data:`MEASURES`."""
    vals = {n: measures(c) for n, c in candidates.items()}
    out = dict.fromkeys(candidates, 0.0)
    for m in MEASURES:
        for n in candidates:
            out[n] += 1 + sum(vals[o][m] > vals[n][m] + 1e-9 for o in candidates)
    return {n: round(v / len(MEASURES), 3) for n, v in out.items()}


def choose(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    """Best mean rank over the development measures; ties: the better 2018 official score."""
    r = mean_ranks(candidates)
    return min(candidates, key=lambda n: (r[n], -measures(candidates[n])["official_2018_eur"]))


def choose_official_only(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    """The D-041/D-042 rule alone: best 2018 official score (reported, not used)."""
    return max(candidates, key=lambda n: measures(candidates[n])["official_2018_eur"])


def main(argv: list[str] | None = None) -> int:
    """Fit the inversions, evaluate every cue and candidate, choose on 2018, write metrics."""
    parser = argparse.ArgumentParser(description="Model-based inversion + system localizer.")
    parser.add_argument("--no-write", action="store_true", help="print only (no metrics/models)")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    cfg = load_config()
    gcfg, lsel = cfg["dl"]["gnn"], cfg["dl"]["localizer_selection"]
    seed = int(cfg["project"]["seed"])
    within = float(gcfg["within_m"])
    edges = [float(v) for v in gcfg["size_buckets_m"]]
    data_dir = paths.sim_dir() / "gnn"
    ds = np.load(data_dir / "dataset.npz")
    graph = np.load(data_dir / "graph.npz", allow_pickle=False)
    loaded = gnn.load_localizer()
    if loaded is None:
        raise FileNotFoundError("no saved localizer: run `make gnn` first")
    g, model, scale, meta = loaded
    if "sensitivity" not in graph:
        raise ValueError("graph.npz has no sensitivity matrix: run `make gnn-data`")
    s = np.asarray(graph["sensitivity"], dtype="float64")
    x, y, diam = ds["x"].astype("float32"), ds["label"].astype("int64"), ds["diameter_m"]
    xm = x.astype("float64").mean(axis=2)
    tr, va, te, _ = gnn.split_indices(diam, gcfg, seed)

    from leakwatch.data import network

    sensors = network.parse_sensor_config()["pressure"]
    ridge, by_ridge, cov_sim = choose_ridge(
        xm, y, diam, s, (tr, va), g, [float(r) for r in lsel["ridge_grid"]], edges[0]
    )
    cov_real, n_real = real_covariance(cfg, sensors)
    inversions = {
        "sim": Inversion(s, cov_sim, ridge, g.is_junction),
        "real": Inversion(s, cov_real, ridge, g.is_junction),
    }
    eig = np.linalg.eigvalsh(cov_sim)[::-1]
    noise = {
        "sim": {
            "from": "training scenarios, residual after the true leak's fit",
            "n": len(tr),
            "rms_m": round(float(np.sqrt(np.trace(cov_sim) / len(cov_sim))), 4),
            "top_eigen_share": [round(float(v), 3) for v in (eig[:3] / eig.sum())],
        },
        "real": {
            "from": "2018 normal windows (D-015), re-baselined residuals, 2 h means",
            "n": n_real,
            "rms_m": round(float(np.sqrt(np.trace(cov_real) / len(cov_real))), 4),
        },
        "ridge_grid_val_medium_large_top5": by_ridge,
        "ridge": ridge,
    }
    log.info("ridge %g (val %s); real noise from %d windows", ridge, by_ridge, n_real)

    def system(lcfg: Mapping[str, Any]) -> gnn.SystemLocalizer:
        kind = lcfg.get("gls")
        return gnn.SystemLocalizer(
            lcfg, loaded=loaded, inversion=inversions[str(kind)] if kind else None
        )

    # ---- held-out simulated leaks (test split)
    cues: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "nearest_sensor": lambda xb: gnn.nearest_sensor_scores(g, xb),
        "sensitivity_cosine": lambda xb: gnn.sensitivity_scores(g, xb),
        "network": lambda xb: gnn.predict(model, g, xb, scale),
        "interpolation": lambda xb: np.where(
            g.is_junction[None, :],
            interpolation_scores(g.dist, g.sensor_idx, xb, float(lsel["interpolation_m"])),
            -np.inf,
        ),
        "gls_sim": inversions["sim"].loglik,
        "gls_real": inversions["real"].loglik,
    }
    candidates_cfg = {k: dict(v) for k, v in lsel["candidates"].items()}
    for v in candidates_cfg.values():
        v.setdefault("interpolation_m", float(lsel["interpolation_m"]))
    sims: dict[str, Any] = {}
    for name, fn in cues.items():
        sims[name] = sim_metrics(g, fn(x[te]), y[te], diam[te], edges, within)
    for name, lcfg in candidates_cfg.items():
        sims[f"candidate:{name}"] = sim_metrics(
            g, system(lcfg).scores(x[te]), y[te], diam[te], edges, within
        )
    for name, r in sims.items():
        log.info("sims %-28s med+large top5 %.3f", name, r["medium_large"]["top5"])

    # ---- 2018 (development): official placement score + recorded leaks
    from leakwatch.eval.battledim_score import YearScorer

    scorer = YearScorer(cfg)
    alerts = lake.query_df(
        lake.connect(),
        "SELECT alert_id, start_ts FROM alerts WHERE origin = 'batch' ORDER BY start_ts",
    )
    cands: dict[str, Any] = {}
    locs = {name: system(lcfg) for name, lcfg in candidates_cfg.items()}
    bench18 = gnn.real_leak_benchmark(
        2018, g, {n: loc.scores for n, loc in locs.items()}, within, sensors
    )
    for name, loc in locs.items():
        cands[name] = {
            "config": candidates_cfg[name],
            "description": describe(candidates_cfg[name]),
            "sims": sims[f"candidate:{name}"],
            "official_2018": official_placement(loc, alerts, 2018, scorer, sensors),
            "recorded_2018": bench18["localizers"][name],
        }
        log.info(
            "2018 %-16s EUR %10.0f  recorded within 300 m %.3f",
            name,
            cands[name]["official_2018"]["total_score_eur"],
            cands[name]["recorded_2018"]["within_m_top1"],
        )
    ranks_ = mean_ranks(cands)
    for name in cands:
        cands[name]["mean_rank"] = ranks_[name]
    chosen = choose(cands)
    official_only = choose_official_only(cands)
    previous = str(lsel.get("previous", "posterior"))
    log.info("mean ranks %s -> chosen %s (official-only rule: %s)", ranks_, chosen, official_only)
    if args.no_write:
        log.info("nothing written, 2019 not computed")
        return 0

    # ---- 2019 (test): reported for the chosen candidate, the official-only alternative and the
    # previous system; computed after the choice, never used for it
    report_2019 = sorted({chosen, official_only, previous} & set(locs))
    bench19 = gnn.real_leak_benchmark(
        2019, g, {n: locs[n].scores for n in report_2019}, within, sensors
    )
    configured = gnn.localizer_config(cfg)
    chosen_cfg = candidates_cfg[chosen]
    matches = all(configured.get(k) == v for k, v in chosen_cfg.items() if k != "interpolation_m")
    payload = {
        "method": "GLS inversion against the sensitivity matrix (1-D leak-size fit, noise "
        "covariance from simulated training residuals or 2018 normal windows) and its fusion with "
        "the network, the sensor prior and graph interpolation (D-044)",
        "prd_metric": "top-5 accuracy of the exact leak junction, held-out simulated medium + "
        f"large leaks (diameter >= {edges[0]:g} m), test split of `make gnn`",
        "noise": noise,
        "simulated_test": sims,
        "candidates": cands,
        "rule": "best mean rank over three development measures: held-out simulated medium + "
        "large top-5 (PRD), 2018 recorded leaks within 300 m (top-1), 2018 official score of the "
        "placement (one report per 2018 batch alert at the top node); ties: the 2018 official "
        "score. 2019 never used",
        "measures": list(MEASURES),
        "chosen": chosen,
        "official_only_rule_choice": official_only,
        "chosen_config": chosen_cfg,
        "previous": previous,
        "configured": configured,
        "config_matches_choice": matches,
        "recorded_2018": {k: v for k, v in bench18.items() if k != "localizers"},
        "recorded_2019_reported_only": bench19,
        "network_model": meta.get("model"),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    for kind, inv in inversions.items():
        inv.save(inversion_path(kind))
    path = write_metrics("dl_localizer", payload)
    log.info("chosen %s (config matches: %s) -> %s", chosen, matches, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
