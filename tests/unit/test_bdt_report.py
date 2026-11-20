"""Alerts as leak reports placed at the most-dropped pressure sensor (D-038)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from leakwatch.bdt.report import pressure_columns, sensor_drop_detections

SPEC = SimpleNamespace(sensor_series=("pressure__a", "pressure__b", "flow__p1"))
DIST = SimpleNamespace(link_at_node=lambda node: {"a": "pa", "b": "pb"}.get(node))


def test_report_at_the_sensor_that_dropped_most_in_the_last_hour():
    ends = pd.date_range("2019-01-01 01:00", periods=12, freq="15min", tz="UTC")
    a = np.zeros(12)
    b = np.zeros(12)
    b[6:9] = -2.0  # sensor b drops just before the alert at ends[8]
    a[:4] = -5.0  # an older drop at a, outside the alert's last hour
    feats = pd.DataFrame(
        {"window_end": ends, "pressure__a__residual": a, "pressure__b__residual": b}
    )
    alerts = pd.DataFrame({"start_ts": [ends[8], ends[0] - pd.Timedelta(hours=3)]})
    det = sensor_drop_detections(alerts, feats, SPEC, DIST)
    assert pressure_columns(SPEC) == ["pressure__a__residual", "pressure__b__residual"]
    assert list(det["link_id"]) == ["pb"]  # the second alert has no windows before it
    assert det["ts"].iloc[0] == ends[8]
