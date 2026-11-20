"""Alerts as BattLeDIM leak reports with the detector's own location (D-038).

The K-Means detector says *when*; to score it with the official BattLeDIM rules on its own (before
the DL localizer), each alert is placed at the pressure sensor whose re-baselined residual fell
most over the alert's first hour (the coarse first stage of LILA, the Leakbusters method), at the
pipe nearest that sensor. ``make battledim-score`` scores the full system with the localizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakwatch.features.spec import FeatureSpec


def pressure_columns(spec: FeatureSpec) -> list[str]:
    """Residual feature column of every pressure sensor."""
    return [f"{s}__residual" for s in spec.sensor_series if s.startswith("pressure__")]


def sensor_drop_detections(
    alerts: pd.DataFrame, features: pd.DataFrame, spec: FeatureSpec, dist: Any
) -> pd.DataFrame:
    """One detection per alert: (``ts`` = alert start, ``link_id`` at the most-dropped sensor).

    ``features`` holds the (re-baselined) residual windows; the drop is their mean over the
    windows ending in the hour up to the alert start.
    """
    cols = pressure_columns(spec)
    nodes = [c.split("__")[1] for c in cols]
    f = features.set_index(pd.to_datetime(features["window_end"], utc=True))[cols].sort_index()
    ends = f.index.asi8  # UTC nanoseconds
    values = f.to_numpy(dtype="float64")
    hour = pd.Timedelta(hours=1).value
    rows = []
    for a in alerts.itertuples(index=False):
        t = pd.Timestamp(a.start_ts).value
        lo = np.searchsorted(ends, t - hour, side="right")
        hi = np.searchsorted(ends, t, side="right")
        if hi <= lo:
            continue
        drop = np.nanmean(values[lo:hi], axis=0)
        if np.all(np.isnan(drop)):
            continue
        link = dist.link_at_node(nodes[int(np.nanargmin(drop))])
        if link is not None:
            rows.append({"ts": pd.Timestamp(a.start_ts), "link_id": link})
    return pd.DataFrame(rows, columns=["ts", "link_id"])
