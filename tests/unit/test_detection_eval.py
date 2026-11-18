"""Event-based detection metrics (eval.detection, D-020)."""

from __future__ import annotations

import pandas as pd
import pytest

from leakwatch.eval.detection import evaluate_detections, label_alerts


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


GT = pd.DataFrame(
    {
        "leak_id": ["old", "new_abrupt", "new_incip"],
        "pipe_id": ["p1", "p2", "p3"],
        "type": ["incipient", "abrupt", "incipient"],
        "start_ts": [ts("2018-06-01"), ts("2019-02-01"), ts("2019-06-01")],
        "end_ts": [pd.NaT, ts("2019-02-20"), pd.NaT],
        "peak_outflow_m3h": [6.0, 20.0, 12.0],
    }
)
START, END = ts("2019-01-01"), ts("2019-12-31 23:59:59")


def test_labels():
    alerts = pd.Series(
        [
            ts("2019-01-10"),  # only the carried-over leak active -> preexisting
            ts("2019-02-02 06:00"),  # new abrupt leak, day 1 -> tp
            ts("2019-02-15"),  # abrupt leak still active, beyond 7 days -> fp
            ts("2019-03-15"),  # nothing new active -> preexisting (old leak never repaired)
            ts("2019-06-05"),  # incipient day 4 -> tp
        ]
    )
    labels = label_alerts(alerts, GT, START, END, pd.Timedelta(days=7))
    assert list(labels["label"]) == ["preexisting", "tp", "fp", "preexisting", "tp"]
    assert labels.iloc[1]["matched_leaks"] == ["new_abrupt"]


def test_metrics():
    alerts = pd.DataFrame(
        {"start_ts": [ts("2019-02-02 06:00"), ts("2019-02-15"), ts("2019-06-05"), ts("2018-12-30")]}
    )
    ev = evaluate_detections(alerts, GT, START, END, horizon_days=7)
    assert ev["detection"]["overall"] == {"leaks": 2, "detected": 2, "rate": 1.0}
    assert ev["detection"]["abrupt"]["detected"] == 1
    assert ev["time_to_detection_h"]["values"] == [30.0, 96.0]
    assert ev["time_to_detection_h"]["median"] == pytest.approx(63.0)
    assert ev["alerts"] == {
        "total": 3,  # the 2018 alert is outside the period
        "tp": 2,
        "fp": 1,
        "preexisting": 0,
        "fp_during_active_new_leak": 1,
    }
    assert ev["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert ev["false_alarms_per_week"] == pytest.approx(1 / ev["weeks"], abs=1e-3)


def test_missed_leak():
    ev = evaluate_detections(
        pd.DataFrame({"start_ts": pd.to_datetime([], utc=True)}), GT, START, END
    )
    assert ev["detection"]["overall"]["rate"] == 0.0
    assert ev["precision"] is None
    assert all(not p["detected"] for p in ev["per_leak"])


def test_chance_level_grows_with_alert_count():
    from leakwatch.eval.detection import chance_detection

    few = chance_detection(2, GT, START, END, n_iter=200)
    many = chance_detection(200, GT, START, END, n_iter=200)
    assert 0.0 <= few["mean_rate"] < many["mean_rate"] <= 1.0
    assert many["mean_rate"] > 0.9  # 200 random alerts a year hit almost every 7-day window
    assert chance_detection(0, GT, START, END)["mean_rate"] == 0.0


def test_time_to_detection_ignores_missed_leaks():
    alerts = pd.DataFrame({"start_ts": [ts("2019-02-02 06:00")]})  # only the abrupt leak
    ev = evaluate_detections(alerts, GT, START, END, horizon_days=7)
    assert ev["time_to_detection_h"]["values"] == [30.0]
    assert ev["time_to_detection_h"]["median"] == 30.0
