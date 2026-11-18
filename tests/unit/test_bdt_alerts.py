"""Alert rule (bdt.alerts): persistence, closing, explanations, resumable state."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.bdt.alerts import AlertRule, AlertTracker, alert_id_for, run_alerts
from leakwatch.common.schemas import ALERT_LAKE_COLUMNS

SERIES = ["pressure__n1", "pressure__n2", "flow__p1"]
AREAS = {"pressure__n1": "A", "pressure__n2": "B", "flow__p1": "C"}
T0 = 1_546_300_800  # 2019-01-01 00:00 UTC


def rule(m: int = 3, close: int = 2) -> AlertRule:
    return AlertRule(
        threshold=1.0,
        persistence_windows=m,
        close_after_windows=close,
        top_n=2,
        series_area=AREAS,
        origin="batch",
        model="kmeans",
    )


def feed(tracker: AlertTracker, scores, shares=None, excess=None, start=0):
    n = len(scores)
    ends = [T0 + (start + i + 1) * 900 for i in range(n)]
    shares = np.tile([0.6, 0.3, 0.1], (n, 1)) if shares is None else shares
    excess = [5.0] * n if excess is None else excess
    return tracker.update(ends, scores, shares, excess)


def test_needs_m_consecutive_windows_above():
    t = AlertTracker(rule(m=3), "battledim", None, SERIES)
    assert feed(t, [2, 2, 0.5, 2, 2]) == []  # broken run: no alert
    rows = feed(t, [2], start=5)  # third consecutive (windows 4, 5, 6)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "open"
    assert row["start_ts"] == pd.Timestamp(T0 + 6 * 900, unit="s", tz="UTC")  # end of m-th window
    assert row["windows_above"] == 3
    assert row["version"] == 1
    assert set(row) == set(ALERT_LAKE_COLUMNS)


def test_closes_after_windows_below_and_keeps_end_of_last_above():
    t = AlertTracker(rule(m=2, close=2), "battledim", None, SERIES)
    rows = feed(t, [3, 4, 0.1, 5, 0.2, 0.3, 0.1])
    assert len(rows) == 1  # one alert, final state
    row = rows[0]
    assert row["status"] == "closed"
    assert row["windows_above"] == 3  # windows 1, 2 and 4 (one dip did not close it)
    assert row["kmeans_score_max"] == 5.0
    assert row["end_ts"] == pd.Timestamp(T0 + 4 * 900, unit="s", tz="UTC")
    assert row["top_sensors"][0] == {"sensor_id": "pressure__n1", "contribution": 0.6}
    assert row["area"] == "A"
    assert row["est_loss_m3h"] == 5.0


def test_negative_excess_inflow_is_floored_and_missing_is_none():
    t = AlertTracker(rule(m=1), "battledim", None, SERIES)
    assert feed(t, [2], excess=[-3.0])[0]["est_loss_m3h"] == 0.0
    t2 = AlertTracker(rule(m=1), "battledim", None, SERIES)
    assert feed(t2, [2], excess=[None])[0]["est_loss_m3h"] is None


def test_state_round_trip_gives_same_alerts():
    scores = [0.1, 2, 2, 2, 0.5, 2, 0.2, 0.1, 0.1, 3, 3, 3, 3]
    one = AlertTracker(rule(), "battledim", None, SERIES)
    whole = feed(one, scores)
    a = AlertTracker(rule(), "battledim", None, SERIES)
    first = feed(a, scores[:5])
    b = AlertTracker(rule(), "battledim", None, SERIES, state=a.state())
    second = feed(b, scores[5:], start=5)
    final = {r["alert_id"]: r for r in [*first, *second]}
    assert {r["alert_id"]: (r["status"], r["windows_above"]) for r in whole} == {
        k: (v["status"], v["windows_above"]) for k, v in final.items()
    }


def test_replayed_windows_are_ignored():
    t = AlertTracker(rule(m=1), "battledim", None, SERIES)
    feed(t, [2, 2])
    assert feed(t, [9, 9]) == []  # same window ends again: skipped


def test_alert_ids_are_deterministic():
    assert alert_id_for("batch", "kmeans", "battledim", None, T0) == alert_id_for(
        "batch", "kmeans", "battledim", None, T0
    )
    assert alert_id_for("batch", "kmeans", "battledim", None, T0) != alert_id_for(
        "stream", "kmeans", "battledim", None, T0
    )


def test_invalid_rule():
    with pytest.raises(ValueError):
        rule(m=0)


def test_run_alerts_batch_groups_by_scenario():
    n = 8
    starts = pd.date_range("2019-01-01", periods=n, freq="15min", tz="UTC")
    scored = pd.DataFrame(
        {
            "source": ["sim"] * n * 2,
            "scenario_id": ["s1"] * n + ["s2"] * n,
            "window_start": list(starts) * 2,
            "window_end": list(starts + pd.Timedelta(hours=1)) * 2,
            "score": [2.0] * n + [0.1] * n,
        }
    )
    feats = pd.DataFrame({"flow__p1__residual": [1.5] * (2 * n)})
    shares = np.tile([0.1, 0.2, 0.7], (2 * n, 1))
    out = run_alerts(scored, shares, SERIES, rule(m=3), ["flow__p1__residual"], feats)
    assert len(out) == 1
    assert out.iloc[0]["scenario_id"] == "s1"
    assert out.iloc[0]["area"] == "C"
    assert out.iloc[0]["est_loss_m3h"] == 1.5
