"""``make bdt-eval``: score 2018 + 2019 with the trained detectors and evaluate on 2019 (PRD §7.4).

2019 is never used for tuning: τ and m come from 2018 (``bdt.train``). This job

1. scores every 2018 and 2019 feature window (numpy scorer = Spark model arithmetic);
2. runs the alert rule over the two years as one stream (alerts may span New Year);
3. writes the main model's scores and alerts to the lake (``origin=batch``) for the copilot and
   dashboard;
4. evaluates 2019 with the event-based metrics of ``eval.detection`` for the main model and the
   strict-normal ablation, plus 2018 in-sample numbers for reference;
5. draws the 2019 score timeline with leak bands and a per-leak detection chart;
6. writes ``reports/metrics/bdt_kmeans_2019.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leakwatch.bdt.alerts import AlertRule, run_alerts
from leakwatch.bdt.data import load_features, series_area_map, write_alerts, write_scores
from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
from leakwatch.bdt.train import MODELS, feature_view, inflow_residual_columns, model_dir
from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.eval.detection import chance_detection, evaluate_detections, onset_contrast
from leakwatch.features.spec import load_feature_spec
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.bdt.evaluate")


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last instant of a calendar year (UTC)."""
    return (
        pd.Timestamp(f"{year}-01-01 00:00", tz="UTC"),
        pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"),
    )


def score_and_alert(
    params: KMeansParams, features: pd.DataFrame, cfg: dict, areas: dict, now: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores for every window and the final state of every alert."""
    scorer = KMeansScorer(params)
    scored, shares = scorer.score_frame(features)
    rule = AlertRule(
        threshold=float(params.threshold),  # type: ignore[arg-type]
        persistence_windows=int(params.persistence_windows),  # type: ignore[arg-type]
        close_after_windows=int(cfg["bdt"]["close_after_windows"]),
        top_n=int(cfg["bdt"]["top_sensors"]),
        series_area=areas,
        origin="batch",
        model="kmeans",
    )
    alerts = run_alerts(
        scored,
        shares,
        scorer.series,
        rule,
        inflow_residual_columns(cfg),
        features,
        now=now,
    )
    return scored, alerts


def plot_timeline(
    scored: pd.DataFrame,
    alerts: pd.DataFrame,
    gt: pd.DataFrame,
    tau: float,
    year: int,
    suffix: str = "",
) -> str:
    """Daily max score with leak bands (new leaks in colour, carried-over in grey) and alerts."""
    start, end = year_bounds(year)
    s = scored[(scored["window_start"] >= start) & (scored["window_start"] <= end)]
    daily = s.set_index("window_start")["score"].resample("6h").max()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for leak in gt.itertuples(index=False):
        a = max(leak.start_ts, start)
        b = min(leak.end_ts if pd.notna(leak.end_ts) else end, end)
        if a > end or b < start:
            continue
        new = leak.start_ts >= start
        color = ("#d62728" if leak.type == "abrupt" else "#ff7f0e") if new else "#bbbbbb"
        ax.axvspan(
            a,
            min(b, leak.start_ts + pd.Timedelta(days=7)) if new else b,
            color=color,
            alpha=0.18 if new else 0.08,
            lw=0,
        )
    ax.plot(daily.index, daily.values, color="#1f77b4", lw=0.8, label="max score per 6 h")
    ax.axhline(tau, color="black", ls="--", lw=1, label=f"τ = {tau:.2f}")
    a = alerts[(alerts["start_ts"] >= start) & (alerts["start_ts"] <= end)]
    ax.scatter(
        a["start_ts"],
        np.full(len(a), tau),
        marker="v",
        color="#2ca02c",
        s=28,
        zorder=3,
        label="alert fired",
    )
    ax.set_yscale("log")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("K-Means score (log)")
    ax.set_title(
        f"{year}: K-Means score vs leaks (red abrupt / orange incipient: first 7 days; "
        "grey: carried over)"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = paths.FIGURES_DIR / f"bdt_score_timeline_{year}{suffix}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def plot_per_leak(per_leak: list[dict[str, Any]], year: int, suffix: str = "") -> str:
    """Time to detection per new leak (missed leaks marked)."""
    df = pd.DataFrame(per_leak).sort_values("start")
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#d62728" if t == "abrupt" else "#ff7f0e" for t in df["type"]]
    vals = [v if v is not None else 0 for v in df["time_to_detection_h"]]
    ax.bar(range(len(df)), vals, color=colors)
    for i, (det, peak) in enumerate(zip(df["detected"], df["peak_outflow_m3h"], strict=True)):
        if not det:
            ax.text(i, 2, "missed", rotation=90, ha="center", va="bottom", fontsize=7)
        ax.text(i, -12, f"{peak:.0f}", ha="center", fontsize=6, color="grey")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["pipe_id"], rotation=90, fontsize=7)
    ax.set_ylabel("time to detection (h)")
    ax.set_title(
        f"{year} leaks: hours to first alert (red abrupt, orange incipient; grey: peak m³/h)"
    )
    fig.tight_layout()
    out = paths.FIGURES_DIR / f"bdt_detection_per_leak_{year}{suffix}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def main(argv: list[str] | None = None) -> int:
    """Score, write, evaluate (see module docstring)."""
    parser = argparse.ArgumentParser(description="Evaluate the K-Means detector on 2019.")
    parser.add_argument("--no-write", action="store_true", help="do not write scores/alerts")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    cfg = load_config()
    spec = load_feature_spec(cfg)
    gt = load_ground_truth()
    areas = series_area_map(spec)
    train_year, eval_year = int(cfg["bdt"]["train_year"]), int(cfg["bdt"]["eval_year"])
    features = pd.concat([load_features(train_year), load_features(eval_year)], ignore_index=True)
    now = datetime.now(UTC)
    horizon = float(cfg["bdt"]["detection_horizon_days"])
    results: dict[str, Any] = {}
    figures: list[str] = []
    for rule, name in MODELS.items():
        directory = model_dir(name)
        if not (directory / "params.json").exists():
            log.warning("no trained model in %s (run `make bdt-train`)", directory)
            continue
        params = KMeansParams.load(directory)
        view = str(params.meta.get("feature_view", "absolute"))
        rows = feature_view(features, spec, view)
        scored, alerts = score_and_alert(params, rows, cfg, areas, now)
        ev = evaluate_detections(alerts, gt, *year_bounds(eval_year), horizon)
        ev_train = evaluate_detections(alerts, gt, *year_bounds(train_year), horizon)
        ev["onset_contrast_24h"] = onset_contrast(scored, gt, *year_bounds(eval_year))
        ev["chance_level"] = chance_detection(
            ev["alerts"]["total"], gt, *year_bounds(eval_year), horizon
        )
        monthly = scored.assign(m=scored["window_start"].dt.strftime("%Y-%m"))
        monthly = monthly[monthly["window_start"].dt.year == eval_year]
        results[rule] = {
            "model_dir": str(directory),
            "feature_view": view,
            "k": params.k,
            "threshold": params.threshold,
            "persistence_windows": params.persistence_windows,
            "eval_2019": ev,
            "in_sample_2018": {
                k: ev_train[k]
                for k in (
                    "detection",
                    "time_to_detection_h",
                    "alerts",
                    "precision",
                    "false_alarms_per_week",
                )
            },
            "windows_above_threshold_2019": round(
                float(scored.loc[scored["window_start"].dt.year == eval_year, "above"].mean()), 4
            ),
            "windows_above_threshold_by_month": {
                k: round(float(v), 4) for k, v in monthly.groupby("m")["above"].mean().items()
            },
        }
        if rule in ("main", "delta"):
            suffix = "" if rule == "main" else f"_{rule}"
            figures += [
                plot_timeline(scored, alerts, gt, float(params.threshold), eval_year, suffix),
                plot_per_leak(ev["per_leak"], eval_year, suffix),
            ]
        if rule == "main" and not args.no_write:
            spark = get_spark(cfg["spark"], "leakwatch-bdt-eval")
            n_scores = write_scores(spark, scored, "kmeans", "batch")
            n_alerts = write_alerts(spark, alerts, "batch", mode="overwrite")
            log.info("wrote %d scores and %d alerts (origin=batch)", n_scores, n_alerts)
            results[rule]["lake"] = {"scores_written": n_scores, "alerts_written": n_alerts}
    payload = {
        "train_year": train_year,
        "eval_year": eval_year,
        "label_rules": "eval.detection (D-020): tp = alert within [leak start, +7 d]; "
        "preexisting = only carried-over 2018 leaks active; fp = everything else",
        "models": results,
        "figures": figures,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    path = write_metrics("bdt_kmeans_2019", payload)
    main_ev = results.get("main", {}).get("eval_2019", {})
    log.info(
        "2019: detection %s, precision %s, false alarms/week %s -> %s",
        main_ev.get("detection", {}).get("overall"),
        main_ev.get("precision"),
        main_ev.get("false_alarms_per_week"),
        path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
