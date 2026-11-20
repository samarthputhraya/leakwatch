"""``make gnn-prior``: the posterior localizer, network × a prior around the most-dropped sensor.

On real leaks the trained network alone is no closer than the most-dropped pressure sensor
(``dl_gnn.json``, ``real_leaks``). It learned on simulated residuals; real ones also carry demand
noise and the other leaks running at the time. The sensor ranking is robust but can only name one
of 33 junctions. The posterior combines both (D-042):

    p(node) ∝ p_net(node) · exp(−d(node, s*) / L)

with s* the pressure sensor that fell most and d the pipe distance. L = ∞ (``none``) is the network
alone; a small L is the sensor alone.

L is chosen on 2018 only, by the official BattLeDIM score (D-037) of the live model's 2018 batch
alerts, each reported at the posterior's top node (``dl.gnn.sensor_prior_grid_m``; ties go to the
longer L, the weaker prior). The 2019 score of every L is reported, never used. The real-leak
benchmark of ``make gnn`` (each leak localized 2 h after its outflow first reaches half its peak)
is repeated for the network, the sensor and the posterior at the chosen L, with paired bootstrap
95% intervals over leaks for the posterior's gain over each.

Writes ``reports/metrics/dl_gnn_prior.json``. ``dl.gnn.sensor_prior_m`` in the config is what the
alerts and the live stream use; the job reports whether it matches the choice.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.common import lake
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.dl import gnn

log = get_logger("leakwatch.dl.gnn_prior")


def length_key(length_m: float | None) -> str:
    """JSON key of a prior length: ``none`` or the metres (``300``)."""
    return "none" if not length_m else f"{length_m:g}"


def choose_length(scores_2018: Mapping[float | None, float]) -> float | None:
    """The prior length with the best 2018 score; ties go to the longer one (``None`` longest)."""
    return max(scores_2018, key=lambda m: (scores_2018[m], m or math.inf))


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[[np.ndarray], float],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """``stat(a) − stat(b)`` for paired per-leak values, with a 95% percentile interval.

    Leaks are resampled with replacement ``draws`` times; both columns use the same draw.
    """
    x, y = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    diffs = np.array([stat(x[i]) - stat(y[i]) for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "estimate": round(float(stat(x) - stat(y)), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "draws": draws,
    }


def gains(
    per_leak: list[Mapping[str, Any]], other: str, within_m: float, draws: int, seed: int
) -> dict[str, Any]:
    """The posterior's gain over ``other`` on real leaks (positive = posterior better).

    ``within_m_top1_gain``: share of leaks with the top node within ``within_m``, posterior minus
    other; ``median_distance_gain_m``: other's median top-1 distance minus the posterior's.
    """
    if not per_leak:
        return {}
    post = [float(r["posterior_distance_top1_m"]) for r in per_leak]
    base = [float(r[f"{other}_distance_top1_m"]) for r in per_leak]
    return {
        "within_m_top1_gain": paired_bootstrap(
            post, base, lambda v: float(np.mean(v <= within_m)), draws, seed
        ),
        "median_distance_gain_m": paired_bootstrap(
            base, post, lambda v: float(np.median(v)), draws, seed
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Choose the prior length on 2018, benchmark the posterior on real leaks, write metrics."""
    parser = argparse.ArgumentParser(description="Posterior localizer: choose the sensor prior.")
    parser.parse_args(argv)
    from leakwatch.data import network
    from leakwatch.eval.battledim_score import YearScorer, year_bounds

    cfg = load_config()
    gcfg = cfg["dl"]["gnn"]
    loaded = gnn.load_localizer()
    if loaded is None:
        raise FileNotFoundError("no saved localizer: run `make gnn` first")
    g, model, scale, meta = loaded
    sensors = network.parse_sensor_config()["pressure"]
    scorer = YearScorer(cfg)
    alerts = lake.query_df(
        lake.connect(),
        "SELECT alert_id, start_ts FROM alerts WHERE origin = 'batch' ORDER BY start_ts",
    )
    grid: list[float | None] = [None, *(float(v) for v in gcfg["sensor_prior_grid_m"])]
    official: dict[str, dict[str, Any]] = {length_key(m): {} for m in grid}
    for year in (2018, 2019):
        first, last = year_bounds(year)
        a = alerts[(alerts["start_ts"] >= first) & (alerts["start_ts"] <= last)]
        x = gnn.benchmark_inputs(g, list(a["start_ts"]), sensors)
        probs = gnn.predict(model, g, x, scale)
        for length in grid:
            top = np.argmax(gnn.posterior(probs, g, x, length), axis=1)
            rows = [
                {"ts": ts, "link_id": link}
                for ts, j in zip(a["start_ts"], top, strict=True)
                if (link := scorer.dist.link_at_node(g.nodes[j])) is not None
            ]
            r = scorer.score(pd.DataFrame(rows, columns=["ts", "link_id"]), year)
            official[length_key(length)][str(year)] = {
                k: r[k] for k in ("total_score_eur", "true_positives", "false_positives")
            } | {"alerts": len(a)}
    chosen = choose_length({m: official[length_key(m)]["2018"]["total_score_eur"] for m in grid})

    within = float(gcfg["within_m"])
    localizers = {
        "nearest_sensor": lambda xb: gnn.nearest_sensor_scores(g, xb),
        "network": lambda xb: gnn.predict(model, g, xb, scale),
        "posterior": lambda xb: gnn.posterior(gnn.predict(model, g, xb, scale), g, xb, chosen),
    }
    draws, seed = int(gcfg["bootstrap_draws"]), int(cfg["project"]["seed"])
    real = {}
    for year in (2018, 2019):
        bench = gnn.real_leak_benchmark(year, g, localizers, within, sensors)
        bench["posterior_gain"] = {
            other: gains(bench.get("per_leak", []), other, within, draws, seed)
            for other in ("nearest_sensor", "network")
        }
        real[str(year)] = bench
    configured = gnn.prior_length(cfg)
    payload = {
        "network": meta["model"],
        "rule": "posterior = network probability x exp(-pipe distance to the most-dropped "
        "pressure sensor / L); L by the official 2018 score, ties to the longer L (D-042)",
        "grid_m": [length_key(m) for m in grid],
        "official_score_by_length": official,
        "chosen_sensor_prior_m": chosen,
        "configured_sensor_prior_m": configured,
        "config_matches_choice": configured == chosen,
        "real_leaks": real,
    }
    path = write_metrics("dl_gnn_prior", payload)
    log.info(
        "chosen L = %s m (2018 %.0f EUR; none %.0f EUR); config %s -> %s",
        length_key(chosen),
        official[length_key(chosen)]["2018"]["total_score_eur"],
        official["none"]["2018"]["total_score_eur"],
        length_key(configured),
        path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
