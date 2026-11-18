"""``make bdt-train``: train the K-Means detector on normal 2018 windows (PRD §7.3, D-020).

1. Load the 2018 feature table; keep windows that are normal under D-015 (``rule=main``) and,
   for the ablation, under the strict leak-free rule (D-010).
2. Split them by UTC day into train / threshold / validation days (``bdt.split_cycle_days``).
3. Fit StandardScaler → PCA → K-Means (k by elbow, silhouette reported) with Spark MLlib on the
   train days (``bdt.model``).
4. τ = ``threshold_quantile`` of the scores on the threshold days.
5. Tune the persistence m on 2018 only: the smallest m whose false alarms per week on the
   validation days stay within ``max_false_alarms_per_week``; 2018 leak detection is reported for
   each candidate.
6. Save the model (``data/models/kmeans*/``), figures and ``reports/metrics/bdt_kmeans_train.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leakwatch.bdt.alerts import AlertRule, run_alerts
from leakwatch.bdt.data import (
    load_features,
    normal_mask,
    series_area_map,
    split_roles,
    weekly_change,
)
from leakwatch.bdt.model import fit_kmeans
from leakwatch.bdt.scorer import KMeansScorer
from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.eval.detection import evaluate_detections
from leakwatch.features.spec import FeatureSpec, load_feature_spec
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.bdt.train")

#: Detector variants: model directory, D-015 rule for "normal", feature view.
#: ``absolute`` = the PRD §7.2 features; ``weekly_change`` = each window minus the same window
#: 7 days earlier (an improvement experiment, D-020): accumulated leakage cancels out.
MODELS = {"main": "kmeans", "strict": "kmeans_strict", "delta": "kmeans_delta"}
VARIANTS: dict[str, tuple[str, str]] = {
    "main": ("main", "absolute"),
    "strict": ("strict", "absolute"),
    "delta": ("main", "weekly_change"),
}


def model_dir(name: str):  # -> Path
    """Directory of a trained detector under ``data/models``."""
    return paths.models_dir() / name


def inflow_residual_columns(cfg: Mapping[str, Any]) -> list[str]:
    """Residual feature columns whose sum is the excess network inflow (m³/h)."""
    return [f"{s}__residual" for s in cfg["bdt"]["inflow_series"]]


def tune_persistence(
    scorer: KMeansScorer,
    features: pd.DataFrame,
    scored: pd.DataFrame,
    shares: np.ndarray,
    validation_normal: np.ndarray,
    spec: FeatureSpec,
    cfg: Mapping[str, Any],
    gt: pd.DataFrame,
    areas: Mapping[str, str],
) -> tuple[int, list[dict[str, Any]]]:
    """Pick m on 2018 (see module docstring); returns m and the per-candidate table."""
    bdt = cfg["bdt"]
    val_starts = features.loc[validation_normal, "window_start"]
    val_weeks = validation_normal.sum() * spec.slide_s / (7 * 86400)
    val_days = set(val_starts.dt.floor("D"))
    rows = []
    for m in bdt["persistence_candidates"]:
        rule = AlertRule(
            threshold=float(scorer.params.threshold),  # type: ignore[arg-type]
            persistence_windows=int(m),
            close_after_windows=int(bdt["close_after_windows"]),
            top_n=int(bdt["top_sensors"]),
            series_area=areas,
            origin="batch",
            model="kmeans",
        )
        alerts = run_alerts(
            scored, shares, scorer.series, rule, inflow_residual_columns(cfg), features
        )
        # A false alarm on validation days: an alert that fires inside a normal validation window.
        fired = alerts["start_ts"]
        in_val = fired.dt.floor("D").isin(val_days) & fired.apply(
            lambda t: bool(
                ((val_starts <= t) & (t <= val_starts + pd.Timedelta(seconds=spec.window_s))).any()
            )
        )
        fa_week = float(in_val.sum() / val_weeks) if val_weeks else None
        det = evaluate_detections(
            alerts,
            gt,
            pd.Timestamp("2018-01-01", tz="UTC"),
            pd.Timestamp("2018-12-31 23:59", tz="UTC"),
            float(bdt["detection_horizon_days"]),
        )
        rows.append(
            {
                "m": int(m),
                "alerts_2018": len(alerts),
                "validation_false_alarms": int(in_val.sum()),
                "validation_false_alarms_per_week": None if fa_week is None else round(fa_week, 3),
                "detection_2018": det["detection"]["overall"],
                "median_ttd_h_2018": det["time_to_detection_h"]["median"],
            }
        )
        log.info("m=%d: %s", m, rows[-1])
    limit = float(bdt["max_false_alarms_per_week"])
    ok = [
        r
        for r in rows
        if r["validation_false_alarms_per_week"] is not None
        and r["validation_false_alarms_per_week"] <= limit
    ]
    chosen = (
        min(ok, key=lambda r: r["m"])
        if ok
        else min(rows, key=lambda r: r["validation_false_alarms_per_week"])
    )
    return int(chosen["m"]), rows


def feature_view(features: pd.DataFrame, spec: FeatureSpec, view: str) -> pd.DataFrame:
    """The feature rows a variant sees: as stored (absolute) or week-over-week change."""
    if view == "absolute":
        return features
    if view == "weekly_change":
        return weekly_change(features, spec.feature_columns())
    raise ValueError(f"unknown feature view {view!r}")


def plot_k_curve(curve: list[dict[str, Any]], k: int, name: str) -> str:
    """Elbow (SSE) and silhouette against k."""
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ks = [c["k"] for c in curve]
    ax1.plot(ks, [c["sse"] for c in curve], "o-", color="#1f77b4", label="within-cluster SSE")
    ax1.set_xlabel("k")
    ax1.set_ylabel("SSE", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(ks, [c["silhouette"] for c in curve], "s--", color="#d62728", label="silhouette")
    ax2.set_ylabel("silhouette", color="#d62728")
    ax1.axvline(k, color="grey", lw=1)
    ax1.set_title(f"K-Means model selection ({name}): elbow at k = {k}")
    fig.tight_layout()
    out = paths.FIGURES_DIR / f"bdt_k_selection_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def plot_scores(
    scored: pd.DataFrame, roles: np.ndarray, normal: np.ndarray, tau: float, name: str
) -> str:
    """Score distribution: train / threshold / validation normal days vs non-normal windows."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, np.quantile(scored["score"], 0.995), 80)
    for label, mask, color in [
        ("normal train days", normal & (roles == "train"), "#2ca02c"),
        ("normal threshold days", normal & (roles == "threshold"), "#1f77b4"),
        ("normal validation days", normal & (roles == "validation"), "#9467bd"),
        ("not normal (leaks, D-015)", ~normal, "#d62728"),
    ]:
        ax.hist(
            scored.loc[mask, "score"],
            bins=bins,
            histtype="step",
            density=True,
            label=label,
            color=color,
        )
    ax.axvline(tau, color="black", lw=1, ls="--", label=f"τ = {tau:.2f}")
    ax.set_xlabel("distance to nearest centroid (2018 windows)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(f"K-Means anomaly scores, 2018 ({name})")
    fig.tight_layout()
    out = paths.FIGURES_DIR / f"bdt_score_hist_2018_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out.name


def train_one(
    spark,
    rule: str,
    features: pd.DataFrame,
    spec: FeatureSpec,
    cfg: Mapping[str, Any],
    gt: pd.DataFrame,
    areas: Mapping[str, str],
) -> dict[str, Any]:
    """Train, threshold, tune and save one detector variant (see ``VARIANTS``)."""
    t0 = time.perf_counter()
    bdt = cfg["bdt"]
    variant = rule
    rule, view = VARIANTS[variant]
    features = feature_view(features, spec, view)
    normal = normal_mask(features, spec, gt, cfg, rule)
    if view == "weekly_change":  # the reference window a week earlier must be normal too
        earlier = features.assign(window_start=features["window_start"] - pd.Timedelta(days=7))
        normal &= normal_mask(earlier, spec, gt, cfg, rule)
    roles = split_roles(features["window_start"], bdt)
    train = features[normal & (roles == "train")]
    log.info("%s: %d normal windows (%d train)", rule, normal.sum(), len(train))
    trained = fit_kmeans(spark, train, spec.feature_columns(), bdt, int(cfg["project"]["seed"]))
    scorer = KMeansScorer(trained.params)
    scored, shares = scorer.score_frame(features)
    thr_scores = scored.loc[normal & (roles == "threshold"), "score"]
    tau = float(np.quantile(thr_scores, float(bdt["threshold_quantile"])))
    scorer = KMeansScorer(trained.params.with_threshold(tau))
    scored, shares = scorer.score_frame(features)
    validation_normal = normal & (roles == "validation")
    m, table = tune_persistence(
        scorer, features, scored, shares, validation_normal, spec, cfg, gt, areas
    )
    params = trained.params.with_threshold(tau, m)
    params.meta.update(
        {
            "variant": variant,
            "rule": rule,
            "feature_view": view,
            "threshold_quantile": float(bdt["threshold_quantile"]),
        }
    )
    out_dir = model_dir(MODELS[variant])
    params.save(out_dir)
    trained.spark_model.write().overwrite().save(str(out_dir / "spark_pipeline"))
    val = scored.loc[validation_normal, "score"]
    return {
        "rule": rule,
        "feature_view": view,
        "model_dir": str(out_dir),
        "normal_windows": int(normal.sum()),
        "windows_by_role": {
            r: int((normal & (roles == r)).sum()) for r in ("train", "threshold", "validation")
        },
        "pca_components": trained.n_components,
        "pca_explained_variance": round(trained.explained_variance, 4),
        "k": trained.k,
        "k_curve": trained.curve,
        "silhouette_at_k": next(c["silhouette"] for c in trained.curve if c["k"] == trained.k),
        "best_silhouette_k": max(trained.curve, key=lambda c: c["silhouette"])["k"],
        "threshold": tau,
        "threshold_quantile": float(bdt["threshold_quantile"]),
        "validation_windows_above_threshold": round(float((val > tau).mean()), 4),
        "score_quantiles": {
            "normal_train_p50": float(
                np.quantile(scored.loc[normal & (roles == "train"), "score"], 0.5)
            ),
            "normal_validation_p99": float(np.quantile(val, 0.99)),
            "not_normal_p50": float(np.quantile(scored.loc[~normal, "score"], 0.5))
            if (~normal).any()
            else None,
        },
        "persistence_windows": m,
        "persistence_tuning": table,
        "figures": [
            plot_k_curve(trained.curve, trained.k, variant),
            plot_scores(scored, roles, normal, tau, variant),
        ],
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    """Train the main detector and the strict-normal ablation."""
    parser = argparse.ArgumentParser(description="Train the K-Means detector (Spark MLlib).")
    parser.add_argument("--rules", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args(argv)
    cfg = load_config()
    spec = load_feature_spec(cfg)
    gt = load_ground_truth()
    areas = series_area_map(spec)
    features = load_features(int(cfg["bdt"]["train_year"]))
    spark = get_spark(cfg["spark"], "leakwatch-bdt-train")
    results = {rule: train_one(spark, rule, features, spec, cfg, gt, areas) for rule in args.rules}
    path = write_metrics(
        "bdt_kmeans_train",
        {
            "train_year": int(cfg["bdt"]["train_year"]),
            "features": len(spec.feature_columns()),
            "windows_2018": len(features),
            "split": {
                k: cfg["bdt"][k] for k in ("split_cycle_days", "threshold_day", "validation_day")
            },
            "models": results,
        },
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
