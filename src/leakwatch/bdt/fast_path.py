"""``make bdt-fast-path``: choose the live model's fast-path factor on 2018 (D-043).

The live model opens an alert after m = 32 windows above τ (8 h, D-038). m was chosen on 2018 by
the official score, which values months of saved water and barely notices hours. A burst does not
need 8 h of confirmation: with the fast path one window whose score exceeds k·τ opens the alert at
once (``bdt.alerts.AlertRule.fast_threshold``).

k is chosen on 2018 only. The rule is the smallest k in ``bdt.fast_path_grid`` that makes nothing
2018 can measure worse:

- the official 2018 score is not below the score without a fast path. Reports use the system's
  placement and revisit rule from ``battledim_score.json``; if that file is missing, the localizer
  and no revisit rule;
- no more alerts open inside the 2018 normal windows (D-015) than without it.

If no k qualifies, the fast path stays off. 2019 is not looked at. The job writes
``reports/metrics/bdt_fast_path.json``. ``bdt.fast_path_factor`` in the config is what
``make bdt-eval``, the live stream and ``make bdt-synthetic`` use; the job reports whether it
matches the choice.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics

log = get_logger("leakwatch.bdt.fast_path")


def choose_factor(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """The smallest factor no worse than ``factor=None`` on the 2018 score and normal alerts."""
    base = next(r for r in rows if r["factor"] is None)
    ok = [
        float(r["factor"])
        for r in rows
        if r["factor"] is not None
        and r["score_2018_eur"] >= base["score_2018_eur"]
        and r["normal_window_alerts"] <= base["normal_window_alerts"]
    ]
    return min(ok, default=None)


def main(argv: list[str] | None = None) -> int:
    """Score 2018 with each fast-path factor and write the choice."""
    argparse.ArgumentParser(description="Choose the fast-path factor on 2018.").parse_args(argv)
    from leakwatch.bdt.alerts import run_alerts
    from leakwatch.bdt.data import load_features, normal_mask, series_area_map
    from leakwatch.bdt.report import revisit_filter, sensor_drop_detections
    from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
    from leakwatch.bdt.train import feature_view, inflow_residual_columns, model_dir
    from leakwatch.common.metrics import read_metrics
    from leakwatch.data.ground_truth import load_ground_truth
    from leakwatch.eval.alerting import false_alarms_on, make_rule
    from leakwatch.eval.battledim_score import YearScorer
    from leakwatch.features.spec import load_feature_spec

    cfg = load_config()
    bdt = cfg["bdt"]
    spec = load_feature_spec(cfg)
    live = str(bdt["live_model"])
    params = KMeansParams.load(model_dir(live))
    tau, m = float(params.threshold), int(params.persistence_windows)  # type: ignore[arg-type]
    feats = feature_view(load_features(2018), spec, "rebaselined", cfg)
    scorer = KMeansScorer(params)
    scored, shares = scorer.score_frame(feats)
    areas = series_area_map(spec)
    normal = normal_mask(feats, spec, load_ground_truth(), cfg, "main")
    normal_starts = feats.loc[normal, "window_start"]
    official = YearScorer(cfg)
    try:
        system = read_metrics("battledim_score")
    except FileNotFoundError:
        system = {}
    placement = str(system.get("placement") or "localizer")
    days = float((system.get("revisit_rule") or {}).get("chosen_days") or 0.0)
    if placement == "sensor_drop":

        def place(alerts: pd.DataFrame) -> pd.DataFrame:
            return sensor_drop_detections(alerts, feats, spec, official.dist)
    else:
        from leakwatch.dl import gnn

        localizer = gnn.SystemLocalizer(gnn.localizer_config(cfg))

        def place(alerts: pd.DataFrame) -> pd.DataFrame:
            times = list(alerts["start_ts"])
            nodes = gnn.localize_times(times, localizer)
            rows = [
                {"ts": t, "link_id": link}
                for t, n in zip(times, nodes, strict=True)
                if (link := official.dist.link_at_node(n)) is not None
            ]
            return pd.DataFrame(rows, columns=["ts", "link_id"])

    rows: list[dict[str, Any]] = []
    for factor in [None, *(float(k) for k in bdt["fast_path_grid"])]:
        rule = dataclasses.replace(
            make_rule(tau, m, cfg, areas, "kmeans"),
            fast_threshold=None if factor is None else factor * tau,
        )
        alerts = run_alerts(
            scored, shares, scorer.series, rule, inflow_residual_columns(cfg), feats
        )
        r = official.score(revisit_filter(place(alerts), days), 2018)
        tp = [d for d in r["per_detection"] if d["label"] == "tp"]
        n_normal, weeks = false_alarms_on(alerts, normal_starts, spec.window_s, spec.slide_s)
        rows.append(
            {
                "factor": factor,
                "fast_threshold": None if factor is None else round(factor * tau, 4),
                "alerts": len(alerts),
                "score_2018_eur": r["total_score_eur"],
                "true_positives": r["true_positives"],
                "false_positives": r["false_positives"],
                "mean_delay_of_true_positives_h": round(
                    float(np.mean([d["delay_h"] for d in tp])), 2
                )
                if tp
                else None,
                "normal_window_alerts": int(n_normal),
                "normal_weeks": round(float(weeks), 2),
            }
        )
        log.info("factor %s: %s", factor, {k: v for k, v in rows[-1].items() if k != "factor"})
    chosen = choose_factor(rows)
    configured = bdt.get("fast_path_factor")
    payload = {
        "model": live,
        "threshold": tau,
        "persistence_windows": m,
        "placement": placement,
        "revisit_days": days,
        "rule": "smallest k with the 2018 official score not lower and no more alerts in the 2018 "
        "normal windows than without a fast path (D-043)",
        "grid": rows,
        "chosen_factor": chosen,
        "configured_factor": configured,
        "config_matches_choice": (None if not configured else float(configured)) == chosen,
    }
    path = write_metrics("bdt_fast_path", payload)
    log.info("chosen fast-path factor %s (config %s) -> %s", chosen, configured, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
