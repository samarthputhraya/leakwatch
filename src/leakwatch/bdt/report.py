"""Alerts as BattLeDIM leak reports with the detector's own location (D-038), and dispatch rules.

The K-Means detector says *when*; to score it with the official BattLeDIM rules on its own (before
the DL localizer), each alert is placed at the pressure sensor whose re-baselined residual fell
most over the alert's first hour (the coarse first stage of LILA, the Leakbusters method), at the
pipe nearest that sensor. ``make battledim-score`` scores the full system with the localizer.

Two dispatch rules turn a list of reports into what a utility would act on (D-042):

- :func:`revisit_filter` — no second crew to the same pipe within ``days`` of the last report
  there (the crew has just been; a still-running leak there is not new);
- :func:`carry_forward` — a report of the previous year that no repair explains is still an open
  work order on 1 January. It uses only the system's own reports and the utility's repair report
  (``2018_Fixed_Leakages_Report.txt``, which every BattLeDIM team had), never the leak labels.
"""

from __future__ import annotations

from collections.abc import Callable
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


def revisit_filter(detections: pd.DataFrame, days: float) -> pd.DataFrame:
    """``detections`` (``ts``, ``link_id``) without repeats at a pipe within ``days`` (time order).

    A report is dropped when the same pipe was reported less than ``days`` before it; the window
    runs from the last report *kept* there, so a pipe is visited at most once per ``days``.
    ``days`` = 0 keeps everything.
    """
    det = detections.sort_values("ts", kind="stable")
    if days <= 0 or det.empty:
        return det.reset_index(drop=True)
    window = pd.Timedelta(days=days)
    last: dict[str, pd.Timestamp] = {}
    keep = []
    for ts, link in zip(det["ts"], det["link_id"], strict=True):
        t = pd.Timestamp(ts)
        prev = last.get(link)
        keep.append(prev is None or t - prev >= window)
        if keep[-1]:
            last[link] = t
    return det[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def carry_forward(
    detections: pd.DataFrame,
    repairs: pd.DataFrame,
    distance: Callable[[str, str], float],
    max_distance_m: float,
    at: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Open work orders of the previous year, re-reported at ``at`` (one per pipe).

    A report (``ts``, ``link_id``) is *closed* when a repair (``link_id``, ``end_ts``) at most
    ``max_distance_m`` away along the pipes happened at or after it (the crew found that leak and
    fixed it). Every other report is open; each open pipe is reported once at ``at``.
    Returns the new detections and a summary.
    """
    open_links: list[str] = []
    closed = 0
    for ts, link in zip(detections["ts"], detections["link_id"], strict=True):
        t = pd.Timestamp(ts)
        if any(
            pd.Timestamp(r.end_ts) >= t and distance(str(r.link_id), str(link)) <= max_distance_m
            for r in repairs.itertuples(index=False)
        ):
            closed += 1
        elif link not in open_links:
            open_links.append(str(link))
    out = pd.DataFrame({"ts": [at] * len(open_links), "link_id": open_links})
    summary = {
        "previous_reports": len(detections),
        "closed_by_a_repair": closed,
        "open_pipes": open_links,
        "reported_at": at.isoformat(),
    }
    return out, summary
