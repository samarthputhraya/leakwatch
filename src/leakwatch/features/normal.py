"""Which timestamps count as "normal" operation (docs/DECISIONS.md D-010, exact rule D-015).

BattLeDIM 2018 has almost no leak-free time (2.2% of steps), so "normal" allows background
leakage, as utilities do. The rule, per leak in the ground truth:

- **significant** leak (abrupt, or peak outflow above ``background_max_peak_m3h``, or peak
  unknown): excluded from its start to its repair (to the end of the data if never repaired);
- **background** leak (incipient and peak outflow at most ``background_max_peak_m3h``): only its
  onset is excluded, ``[start, start + onset_exclusion_days]`` (capped at its repair).

A timestamp is normal when it lies in none of these intervals (bounds inclusive). The strict rule
(the D-010 ablation) excludes every leak from start to repair.

Used for the hour-of-week baselines (Phase 2) and the K-Means training windows (Phase 3).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from leakwatch.data.ground_truth import DATA_END

Rule = Literal["main", "strict"]
INTERVAL_COLUMNS = [
    "leak_id",
    "pipe_id",
    "type",
    "peak_outflow_m3h",
    "reason",
    "start_ts",
    "end_ts",
]


def excluded_intervals(
    gt: pd.DataFrame,
    background_max_peak_m3h: float,
    onset_exclusion_days: float,
    rule: Rule = "main",
    data_end: pd.Timestamp = DATA_END,
) -> pd.DataFrame:
    """Intervals that are not normal, one per leak, sorted by start (see module docstring).

    Args:
        gt: ground-truth table (``data.ground_truth``).
        background_max_peak_m3h: incipient leaks peaking at or below this are background leakage.
        onset_exclusion_days: length of the excluded onset of a background leak.
        rule: ``"main"`` (D-015) or ``"strict"`` (every leak excluded while active).
        data_end: end of an unrepaired leak.

    Returns:
        Columns ``leak_id, pipe_id, type, peak_outflow_m3h, reason, start_ts, end_ts``; ``reason``
        is ``active`` (whole leak excluded) or ``onset`` (background leak, first days only).

    """
    onset = pd.Timedelta(days=onset_exclusion_days)
    rows = []
    for leak in gt.itertuples(index=False):
        end = leak.end_ts if pd.notna(leak.end_ts) else data_end
        peak = leak.peak_outflow_m3h
        background = (
            leak.type == "incipient" and pd.notna(peak) and float(peak) <= background_max_peak_m3h
        )
        if rule == "main" and background:
            reason, stop = "onset", min(leak.start_ts + onset, end)
        else:
            reason, stop = "active", end
        rows.append(
            {
                "leak_id": leak.leak_id,
                "pipe_id": leak.pipe_id,
                "type": leak.type,
                "peak_outflow_m3h": peak,
                "reason": reason,
                "start_ts": leak.start_ts,
                "end_ts": stop,
            }
        )
    out = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
    return out.sort_values(["start_ts", "leak_id"]).reset_index(drop=True)


def intervals_from_config(
    gt: pd.DataFrame, normal_cfg: Mapping[str, Any], rule: Rule = "main"
) -> pd.DataFrame:
    """:func:`excluded_intervals` with parameters from the ``normal`` config section."""
    return excluded_intervals(
        gt,
        background_max_peak_m3h=float(normal_cfg["background_max_peak_m3h"]),
        onset_exclusion_days=float(normal_cfg["onset_exclusion_days"]),
        rule=rule,
    )


def is_normal(ts: pd.DatetimeIndex | pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    """Boolean mask: True where ``ts`` lies in none of the excluded intervals (inclusive)."""
    values = pd.DatetimeIndex(ts)
    normal = np.ones(len(values), dtype=bool)
    for start, end in zip(intervals["start_ts"], intervals["end_ts"], strict=True):
        normal &= ~((values >= start) & (values <= end))
    return normal


def normal_condition(ts_col: str, intervals: pd.DataFrame):  # -> pyspark.sql.Column
    """Spark column: True where ``ts_col`` lies in none of the excluded intervals (inclusive)."""
    from pyspark.sql import functions as F

    cond = F.lit(True)
    for start, end in zip(intervals["start_ts"], intervals["end_ts"], strict=True):
        lo = F.timestamp_seconds(F.lit(int(start.timestamp())))
        hi = F.timestamp_seconds(F.lit(int(end.timestamp())))
        cond = cond & ~F.col(ts_col).between(lo, hi)
    return cond


def normal_summary(index: pd.DatetimeIndex, intervals: pd.DataFrame, step_s: int) -> dict[str, Any]:
    """How much of a regular time index (``step_s`` apart) is normal, and the normal stretches."""
    mask = is_normal(index, intervals)
    per_day = 86400 / step_s
    stretches = []
    if mask.any():
        # Start/end positions of each run of consecutive normal steps.
        edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True):
            stretches.append(
                {
                    "start": index[s].isoformat(),
                    "end": index[e].isoformat(),
                    "days": round((e - s + 1) / per_day, 2),
                }
            )
    return {
        "steps": len(index),
        "normal_steps": int(mask.sum()),
        "normal_share": round(float(mask.mean()), 4) if len(index) else None,
        "normal_days": round(float(mask.sum()) / per_day, 2),
        "normal_stretches": stretches,
    }
