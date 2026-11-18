"""Event-based detection metrics (PRD §7.4; labels defined in docs/DECISIONS.md D-020).

For an evaluation period (a calendar year):

- *new leaks* started in the period; a new leak is **detected** if an alert starts within
  ``[leak start, leak start + horizon]`` (7 days); time to detection = first such alert start −
  leak start (alert start = the event time the alert fired);
- each alert is labelled:
  - ``tp`` — it starts inside some new leak's detection window;
  - ``preexisting`` — otherwise, and at its start no new leak is active but a leak that started
    before the period is (PRD: tracked separately, not a false alarm);
  - ``fp`` — everything else (a false alarm). The subset that starts while a new leak is active
    beyond its 7-day window is reported as ``fp_during_active_new_leak`` for context;
- precision = tp / (tp + fp); false alarms per week = fp / weeks in the period.

Used by every detector (K-Means, LSTM-AE) so the comparison is like for like.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _active(gt: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    return (gt["start_ts"] <= t) & (gt["end_ts"].isna() | (gt["end_ts"] >= t))


def label_alerts(
    alert_starts: pd.Series,
    gt: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    horizon: pd.Timedelta,
) -> pd.DataFrame:
    """Label alert start times (see module docstring); returns ``label, matched_leaks``."""
    new = gt[(gt["start_ts"] >= period_start) & (gt["start_ts"] <= period_end)]
    old = gt[gt["start_ts"] < period_start]
    labels, matched = [], []
    for t in pd.to_datetime(alert_starts, utc=True):
        hit = new[(new["start_ts"] <= t) & (t <= new["start_ts"] + horizon)]
        if len(hit):
            labels.append("tp")
            matched.append(list(hit["leak_id"]))
        elif not _active(new, t).any() and _active(old, t).any():
            labels.append("preexisting")
            matched.append([])
        else:
            labels.append("fp")
            matched.append([])
    return pd.DataFrame({"label": labels, "matched_leaks": matched}, index=alert_starts.index)


def evaluate_detections(
    alerts: pd.DataFrame,
    gt: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    horizon_days: float = 7.0,
) -> dict[str, Any]:
    """Detection metrics for alerts (``start_ts`` column) against the leak ground truth."""
    horizon = pd.Timedelta(days=horizon_days)
    in_period = alerts[
        (alerts["start_ts"] >= period_start) & (alerts["start_ts"] <= period_end)
    ].sort_values("start_ts")
    labelled = label_alerts(in_period["start_ts"], gt, period_start, period_end, horizon)
    new = gt[(gt["start_ts"] >= period_start) & (gt["start_ts"] <= period_end)]

    per_leak = []
    for leak in new.itertuples(index=False):
        window = in_period[
            (in_period["start_ts"] >= leak.start_ts)
            & (in_period["start_ts"] <= leak.start_ts + horizon)
        ]
        ttd = (
            (window["start_ts"].iloc[0] - leak.start_ts).total_seconds() / 3600
            if len(window)
            else None
        )
        per_leak.append(
            {
                "leak_id": leak.leak_id,
                "pipe_id": leak.pipe_id,
                "type": leak.type,
                "start": leak.start_ts.isoformat(),
                "peak_outflow_m3h": None
                if pd.isna(leak.peak_outflow_m3h)
                else float(leak.peak_outflow_m3h),
                "detected": ttd is not None,
                "time_to_detection_h": None if ttd is None else round(ttd, 2),
                "first_alert": None if ttd is None else window["start_ts"].iloc[0].isoformat(),
            }
        )
    leaks = pd.DataFrame(per_leak)
    counts = labelled["label"].value_counts().to_dict()
    tp, fp = int(counts.get("tp", 0)), int(counts.get("fp", 0))
    weeks = (period_end - period_start).total_seconds() / (7 * 86400)
    fp_alerts = in_period[labelled["label"] == "fp"]
    during_new = sum(_active(new, t).any() for t in fp_alerts["start_ts"])

    def rate(mask: pd.Series) -> dict[str, Any]:
        sub = leaks[mask] if len(leaks) else leaks
        n = len(sub)
        hit = int(sub["detected"].sum()) if n else 0
        return {"leaks": n, "detected": hit, "rate": round(hit / n, 4) if n else None}

    ttds = [x for x in (leaks["time_to_detection_h"] if len(leaks) else []) if x is not None]
    return {
        "period": {"start": period_start.isoformat(), "end": period_end.isoformat()},
        "horizon_days": horizon_days,
        "detection": {
            "overall": rate(pd.Series(True, index=leaks.index)),
            "abrupt": rate(leaks["type"] == "abrupt")
            if len(leaks)
            else rate(pd.Series(dtype=bool)),
            "incipient": rate(leaks["type"] == "incipient")
            if len(leaks)
            else rate(pd.Series(dtype=bool)),
        },
        "time_to_detection_h": {
            "median": float(np.median(ttds)) if ttds else None,
            "mean": float(np.mean(ttds)) if ttds else None,
            "values": ttds,
        },
        "alerts": {
            "total": len(in_period),
            "tp": tp,
            "fp": fp,
            "preexisting": int(counts.get("preexisting", 0)),
            "fp_during_active_new_leak": int(during_new),
        },
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "false_alarms_per_week": round(fp / weeks, 3) if weeks > 0 else None,
        "weeks": round(weeks, 2),
        "per_leak": per_leak,
    }
