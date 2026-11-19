"""Alert-rule tuning and operating points shared by every detector (K-Means, LSTM-AE).

- :func:`tune_persistence`: the smallest m (consecutive windows above τ) whose false alarms per
  week on the 2018 *validation* days stay within ``bdt.max_false_alarms_per_week`` (2019 is never
  used for tuning);
- :func:`threshold_sweep`: thresholds from low quantiles of the threshold-day scores up to
  multiples of their maximum, wide enough that every detector's curve spans the same
  false-alarm range;
- :func:`operating_points`: detection rate, false alarms per week and the chance level in an
  evaluation year for each threshold at a fixed m;
- :func:`best_within`: the best operating point within a false-alarm budget — how detectors are
  compared at matched false-alarm rates (PRD §8.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.bdt.alerts import AlertRule, run_alerts
from leakwatch.eval.detection import chance_detection, evaluate_detections


def make_rule(
    threshold: float, m: int, cfg: Mapping[str, Any], areas: Mapping[str, str], model: str
) -> AlertRule:
    """Alert rule with the configured closing and explanation settings."""
    bdt = cfg["bdt"]
    return AlertRule(
        threshold=float(threshold),
        persistence_windows=int(m),
        close_after_windows=int(bdt["close_after_windows"]),
        top_n=int(bdt["top_sensors"]),
        series_area=dict(areas),
        origin="batch",
        model=model,
    )


def false_alarms_on(
    alerts: pd.DataFrame, window_starts: pd.Series, window_s: int, slide_s: int
) -> tuple[int, float]:
    """Alerts that fire inside the given (normal) windows, and those windows' length in weeks."""
    starts = np.sort(window_starts.to_numpy(dtype="datetime64[ns]").astype("int64"))
    fired = alerts["start_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    span = np.int64(window_s) * 10**9
    pos = np.searchsorted(starts, fired, side="right") - 1
    inside = (pos >= 0) & (fired - starts[np.clip(pos, 0, None)] <= span)
    weeks = len(starts) * slide_s / (7 * 86400)
    return int(inside.sum()), weeks


def tune_persistence(
    threshold: float,
    scored: pd.DataFrame,
    shares: np.ndarray,
    series: Sequence[str],
    validation_normal: np.ndarray,
    features: pd.DataFrame | None,
    inflow_columns: Sequence[str],
    cfg: Mapping[str, Any],
    gt: pd.DataFrame,
    areas: Mapping[str, str],
    model: str,
    window_s: int,
    slide_s: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Choose m on the training year (see module docstring); returns m and the table."""
    bdt = cfg["bdt"]
    year = int(bdt["train_year"])
    val_starts = scored.loc[validation_normal, "window_start"]
    rows = []
    for m in bdt["persistence_candidates"]:
        rule = make_rule(threshold, int(m), cfg, areas, model)
        alerts = run_alerts(scored, shares, series, rule, inflow_columns, features)
        fa, weeks = false_alarms_on(alerts, val_starts, window_s, slide_s)
        det = evaluate_detections(
            alerts,
            gt,
            pd.Timestamp(f"{year}-01-01", tz="UTC"),
            pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"),
            float(bdt["detection_horizon_days"]),
        )
        rows.append(
            {
                "m": int(m),
                "alerts_train_year": len(alerts),
                "validation_false_alarms": fa,
                "validation_false_alarms_per_week": round(fa / weeks, 3) if weeks else None,
                "detection_train_year": det["detection"]["overall"],
                "median_ttd_h_train_year": det["time_to_detection_h"]["median"],
            }
        )
    limit = float(bdt["max_false_alarms_per_week"])
    ok = [
        r
        for r in rows
        if r["validation_false_alarms_per_week"] is not None
        and r["validation_false_alarms_per_week"] <= limit
    ]
    if ok:
        chosen = min(ok, key=lambda r: r["m"])
    else:
        chosen = min(rows, key=lambda r: r["validation_false_alarms_per_week"] or np.inf)
    return int(chosen["m"]), rows


def threshold_sweep(
    threshold_scores: np.ndarray,
    quantiles: Sequence[float],
    max_multiples: Sequence[float],
) -> list[tuple[str, float]]:
    """(label, threshold) pairs: quantiles of the threshold-day scores, then multiples of their max.

    Quantiles alone stop at the largest normal score; a detector whose alerts are frequent even
    there needs higher thresholds to reach a low false-alarm rate, hence the multiples.
    """
    scores = np.asarray(threshold_scores, dtype="float64")
    out = [(f"q{q:g}", float(np.quantile(scores, q))) for q in quantiles]
    top = float(scores.max())
    out += [(f"{k:g}x max", top * float(k)) for k in max_multiples]
    return out


def operating_points(
    thresholds: Sequence[tuple[str, float]],
    m: int,
    scored: pd.DataFrame,
    shares: np.ndarray,
    series: Sequence[str],
    cfg: Mapping[str, Any],
    gt: pd.DataFrame,
    areas: Mapping[str, str],
    model: str,
    period: tuple[pd.Timestamp, pd.Timestamp],
    chance_iterations: int = 200,
) -> list[dict[str, Any]]:
    """Detection, false alarms per week and chance level in ``period`` for each threshold."""
    out = []
    horizon = float(cfg["bdt"]["detection_horizon_days"])
    seed = int(cfg["project"]["seed"])
    for label, tau in thresholds:
        alerts = run_alerts(scored, shares, series, make_rule(tau, m, cfg, areas, model))
        ev = evaluate_detections(alerts, gt, period[0], period[1], horizon)
        chance = chance_detection(
            ev["alerts"]["total"], gt, period[0], period[1], horizon, chance_iterations, seed
        )
        out.append(
            {
                "level": label,
                "threshold": float(tau),
                "alerts": ev["alerts"]["total"],
                "detection_rate": ev["detection"]["overall"]["rate"],
                "abrupt_rate": ev["detection"]["abrupt"]["rate"],
                "incipient_rate": ev["detection"]["incipient"]["rate"],
                "chance_level": chance["mean_rate"],
                "false_alarms_per_week": ev["false_alarms_per_week"],
                "precision": ev["precision"],
                "median_ttd_h": ev["time_to_detection_h"]["median"],
            }
        )
    return out


def best_within(points: Sequence[Mapping[str, Any]], max_fa_per_week: float) -> dict[str, Any]:
    """The operating point with the highest detection rate among those within the budget.

    Ties go to fewer false alarms. Returns ``{"within_budget": False}`` when no threshold gets
    the detector down to ``max_fa_per_week``.
    """
    ok = [
        p
        for p in points
        if p.get("false_alarms_per_week") is not None
        and p["false_alarms_per_week"] <= max_fa_per_week
    ]
    if not ok:
        lowest = min(
            (
                p["false_alarms_per_week"]
                for p in points
                if p.get("false_alarms_per_week") is not None
            ),
            default=None,
        )
        return {"within_budget": False, "lowest_false_alarms_per_week": lowest}
    best = max(ok, key=lambda p: (p["detection_rate"], -p["false_alarms_per_week"]))
    keys = (
        "level",
        "detection_rate",
        "chance_level",
        "abrupt_rate",
        "false_alarms_per_week",
        "precision",
        "median_ttd_h",
        "alerts",
    )
    return {"within_budget": True, **{k: best.get(k) for k in keys}}
