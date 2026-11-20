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


# Gaps found by mutation testing (D-040).


def test_per_leak_rows_and_summary_fields():
    gt = GT.copy()
    gt.loc[2, "peak_outflow_m3h"] = float("nan")
    alerts = pd.DataFrame({"start_ts": [ts("2019-06-05"), ts("2019-02-02 06:20")]})  # unsorted
    ev = evaluate_detections(alerts, gt, START, END, horizon_days=7)
    abrupt, incip = ev["per_leak"]
    assert abrupt == {
        "leak_id": "new_abrupt",
        "pipe_id": "p2",
        "type": "abrupt",
        "start": "2019-02-01T00:00:00+00:00",
        "peak_outflow_m3h": 20.0,
        "detected": True,
        "time_to_detection_h": 30.33,
        "first_alert": "2019-02-02T06:20:00+00:00",
    }
    assert incip["peak_outflow_m3h"] is None and incip["time_to_detection_h"] == 96.0
    assert ev["detection"]["incipient"] == {"leaks": 1, "detected": 1, "rate": 1.0}
    assert ev["detection"]["abrupt"] == {"leaks": 1, "detected": 1, "rate": 1.0}
    # the mean is of the rounded hours
    assert ev["time_to_detection_h"]["mean"] == pytest.approx((30.33 + 96.0) / 2)
    assert ev["period"] == {"start": START.isoformat(), "end": END.isoformat()}
    assert ev["horizon_days"] == 7
    assert ev["weeks"] == pytest.approx(52.14, abs=0.01)
    assert ev["false_alarms_per_week"] == 0.0 and ev["precision"] == 1.0


def test_first_alert_is_the_earliest_in_the_window():
    alerts = pd.DataFrame(
        {"start_ts": [ts("2019-02-05"), ts("2019-02-03"), ts("2019-02-04")]}  # unsorted
    )
    ev = evaluate_detections(alerts, GT, START, END, horizon_days=7)
    assert ev["per_leak"][0]["first_alert"] == "2019-02-03T00:00:00+00:00"
    assert ev["per_leak"][0]["time_to_detection_h"] == 48.0


def test_window_and_period_bounds_are_inclusive():
    leak = ts("2019-02-01")
    on_start = evaluate_detections(pd.DataFrame({"start_ts": [leak]}), GT, START, END)
    assert on_start["per_leak"][0]["time_to_detection_h"] == 0.0
    last = evaluate_detections(
        pd.DataFrame({"start_ts": [leak + pd.Timedelta(days=7)]}), GT, START, END
    )
    assert last["per_leak"][0]["detected"] and last["alerts"]["tp"] == 1
    late = evaluate_detections(
        pd.DataFrame({"start_ts": [leak + pd.Timedelta(days=7, seconds=1)]}), GT, START, END
    )
    assert not late["per_leak"][0]["detected"]
    assert late["alerts"]["fp"] == 1 and late["alerts"]["fp_during_active_new_leak"] == 1
    edges = evaluate_detections(pd.DataFrame({"start_ts": [START, END]}), GT, START, END)
    assert edges["alerts"]["total"] == 2


def test_labels_at_the_edges_of_the_ground_truth():
    gt = GT.copy()
    # a leak that ended: an alert after its end is no longer "active"
    alerts = pd.Series([ts("2019-02-20"), ts("2019-02-20 00:00:01")])
    labels = label_alerts(alerts, gt, START, END, pd.Timedelta(days=7))
    assert list(labels["label"]) == ["fp", "preexisting"]
    # a new leak starting on the period start counts as new
    gt.loc[1, "start_ts"] = START
    first = label_alerts(pd.Series([START]), gt, START, END, pd.Timedelta(days=7))
    assert list(first["label"]) == ["tp"]
    assert list(first.index) == [0]


def test_chance_level_matches_the_closed_form():
    from leakwatch.eval.detection import chance_detection

    one = GT.iloc[[1]]  # one leak, far from the period's ends
    n = 10
    expect = 1 - (1 - 7 / 365) ** n
    got = chance_detection(n, one, START, END, n_iter=4000, seed=1)
    assert got["mean_rate"] == pytest.approx(expect, abs=0.03)
    assert got["p95_rate"] == 1.0  # one leak: each draw detects it or not
    assert got["n_alerts"] == n and got["iterations"] == 4000
    assert chance_detection(n, one, START, END, n_iter=300, seed=7) == chance_detection(
        n, one, START, END, n_iter=300, seed=7
    )
    empty = chance_detection(5, GT.iloc[[0]], START, END, n_iter=10)  # no leak starts in 2019
    assert empty == {"n_alerts": 5, "mean_rate": 0.0, "p95_rate": 0.0, "iterations": 10}


def test_onset_contrast_sees_a_rise():
    from leakwatch.eval.detection import onset_contrast

    t = pd.date_range("2019-01-25", "2019-06-10", freq="15min", tz="UTC")
    score = pd.Series(1.0, index=t)
    score[(t > ts("2019-02-01 01:00")) & (t <= ts("2019-02-02"))] = 3.0  # abrupt leak rises
    scored = pd.DataFrame({"window_end": t, "score": score.to_numpy()})
    out = onset_contrast(scored, GT, START, END, hours=24)
    assert out["leaks"] == 2 and out["hours"] == 24
    rows = {r["leak_id"]: r for r in out["per_leak"]}
    assert rows["new_abrupt"]["after_mean"] == pytest.approx(3.0)
    assert rows["new_abrupt"]["ratio"] == pytest.approx(3.0)
    assert rows["new_abrupt"]["after_above_before_p95"] is True
    assert rows["new_incip"]["after_above_before_p95"] is False
    assert out["share_rising_abrupt"] == 1.0 and out["share_rising_incipient"] == 0.0
    assert out["share_rising_overall"] == 0.5 and out["median_ratio_abrupt"] == pytest.approx(3.0)


def test_chance_level_defaults():
    from leakwatch.eval.detection import chance_detection

    default = chance_detection(3, GT, START, END)
    assert default["iterations"] == 500
    assert default == chance_detection(3, GT, START, END, horizon_days=7.0, n_iter=500, seed=42)
    assert chance_detection(3, GT, START, END, horizon_days=1.0)["mean_rate"] < default["mean_rate"]
