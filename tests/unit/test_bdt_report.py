"""Alerts as leak reports placed at the most-dropped pressure sensor (D-038)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from leakwatch.bdt.report import (
    carry_forward,
    pressure_columns,
    revisit_filter,
    sensor_drop_detections,
)

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


def _dets(rows):
    return pd.DataFrame(
        [{"ts": pd.Timestamp(t, tz="UTC"), "link_id": link} for t, link in rows],
        columns=["ts", "link_id"],
    )


def test_revisit_filter_skips_a_pipe_reported_within_the_window():
    dets = _dets(
        [
            ("2019-01-10", "p1"),
            ("2019-01-01", "p1"),  # out of order on purpose: the rule works in time order
            ("2019-01-05", "p2"),
            ("2019-01-14", "p1"),  # 13 days after the kept 1 Jan report
            ("2019-01-16", "p1"),  # 15 days after it: kept, and the window restarts here
            ("2019-01-20", "p1"),
        ]
    )
    kept = revisit_filter(dets, 14)
    assert list(zip(kept["ts"].dt.day, kept["link_id"], strict=True)) == [
        (1, "p1"),
        (5, "p2"),
        (16, "p1"),
    ]
    assert len(revisit_filter(dets, 0)) == len(dets)
    assert revisit_filter(dets.iloc[:0], 14).empty


def test_revisit_window_boundary_is_inclusive_of_exactly_d_days():
    dets = _dets([("2019-01-01", "p1"), ("2019-01-08", "p1")])
    assert len(revisit_filter(dets, 7)) == 2  # exactly 7 days later: a new visit
    assert len(revisit_filter(dets, 7.5)) == 1


def test_carry_forward_keeps_reports_no_later_repair_explains():
    reports = _dets(
        [
            ("2018-02-01", "pa"),  # repaired nearby later -> closed
            ("2018-06-01", "pb"),  # repaired nearby, but *before* the report -> open
            ("2018-07-01", "pc"),  # nothing repaired near it -> open
            ("2018-08-01", "pc"),  # same pipe again: one open work order
        ]
    )
    repairs = pd.DataFrame(
        {
            "link_id": ["ra", "rb"],
            "end_ts": pd.to_datetime(["2018-03-01", "2018-05-01"], utc=True),
        }
    )
    near = {("ra", "pa"): 100.0, ("rb", "pb"): 50.0}

    def distance(a, b):
        return near.get((a, b), 5000.0)

    at = pd.Timestamp("2019-01-01 00:05", tz="UTC")
    out, summary = carry_forward(reports, repairs, distance, 300.0, at)
    assert list(out["link_id"]) == ["pb", "pc"]
    assert (out["ts"] == at).all()
    assert summary["closed_by_a_repair"] == 1 and summary["previous_reports"] == 4
    assert summary["open_pipes"] == ["pb", "pc"]
    far, _ = carry_forward(reports, repairs, distance, 99.0, at)  # ra is 100 m from pa
    assert list(far["link_id"]) == ["pa", "pb", "pc"]
