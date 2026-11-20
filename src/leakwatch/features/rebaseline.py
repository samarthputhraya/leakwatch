"""Re-baselined residuals (D-038): each window minus the series' own recent level.

The ``residual`` feature compares a window with the 2018 hour-of-week median. In 2019 leaks pile
up (10 running at once on average), so that residual drifts and a detector on it stays alarmed for
most of the year. Every top BattLeDIM team instead compared with a reference that already holds
the leaks found so far. Here the reference is the series' median residual over the previous
``window_days`` days, ending ``gap_days`` before the window (so a new leak is not absorbed into its
own reference for ``gap_days`` + ``window_days`` / 2 days):

    rebaselined(t) = residual(t) − median{residual(s) : t − gap − window ≤ s < t − gap}

Rows are 15-minute windows; the reference needs at least half of its rows, else it is missing.
It is also missing when ``t − gap`` falls in a hole in the data (more than one step after the last
row): an old reference is never carried forward. The same function serves batch training, the
localizer's inputs and the live stream (which keeps the last ``gap_days + window_days`` of windows
in memory).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rebaseline:
    """Reference window (days) and gap (days) of the re-baselined residual."""

    window_days: float = 7.0
    gap_days: float = 1.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Rebaseline:
        """Read ``features.rebaseline``."""
        r = cfg["features"]["rebaseline"]
        return cls(float(r["window_days"]), float(r["gap_days"]))

    @property
    def history(self) -> pd.Timedelta:
        """How far back a window's reference reaches."""
        return pd.Timedelta(days=self.window_days + self.gap_days)


def rebaselined(
    features: pd.DataFrame, columns: Sequence[str], rb: Rebaseline, time_col: str = "window_start"
) -> pd.DataFrame:
    """``features`` with ``columns`` replaced by their re-baselined values (same row order).

    The reference of each row is a time-based rolling median over
    ``[t − gap − window, t − gap)``, so gaps in the data are handled by time, not by row count.
    """
    if features.empty:
        return features.copy()
    order = np.argsort(features[time_col].to_numpy(), kind="stable")
    sorted_f = features.iloc[order]
    t = pd.DatetimeIndex(sorted_f[time_col])
    values = sorted_f[list(columns)].astype("float64")
    values.index = t
    window = pd.Timedelta(days=rb.window_days)
    u = t.unique()  # a repeated timestamp must not make the step zero
    step = (u[1:] - u[:-1]).min() if len(u) > 1 else pd.Timedelta(minutes=15)
    min_rows = max(1, int(window / max(step, pd.Timedelta(minutes=1)) // 2))
    med = values.rolling(window, closed="left", min_periods=min_rows).median()
    # reference for t = median over [t - gap - window, t - gap): the rolling value at t - gap
    shifted = med.copy()
    shifted.index = shifted.index + pd.Timedelta(days=rb.gap_days)
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    ref = shifted.reindex(t, method="ffill", tolerance=step if len(u) > 1 else None)
    out = sorted_f.copy()
    out[list(columns)] = values.to_numpy() - ref.to_numpy()
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return out.iloc[back].set_axis(features.index)
