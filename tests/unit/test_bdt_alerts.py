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


# Gaps found by mutation testing (D-040).


def test_alert_id_depends_on_every_field_and_treats_no_scenario_as_empty():
    import uuid

    from leakwatch.bdt.alerts import ALERT_NAMESPACE

    base = ("batch", "kmeans", "battledim", None, T0)
    assert alert_id_for(*base) == str(uuid.uuid5(ALERT_NAMESPACE, f"batch|kmeans|battledim||{T0}"))
    assert alert_id_for(*base) == alert_id_for("batch", "kmeans", "battledim", "", T0)
    for i, other in enumerate(("stream", "lstm_ae", "sim", "s1", T0 + 900)):
        changed = list(base)
        changed[i] = other
        assert alert_id_for(*changed) != alert_id_for(*base)


def test_rule_allows_closing_after_one_window():
    assert rule(m=1, close=1).close_after_windows == 1
    with pytest.raises(ValueError):
        rule(m=1, close=0)


def test_epoch_seconds_floors_and_reads_naive_as_utc():
    from leakwatch.bdt.alerts import epoch_seconds

    aware = pd.Series(pd.to_datetime(["2019-01-01 00:00:01.900"], utc=True))
    naive = pd.Series(pd.to_datetime(["2019-01-01 00:00:01.900"]))
    assert epoch_seconds(aware).tolist() == [T0 + 1] == epoch_seconds(naive).tolist()


def test_run_alerts_keeps_each_groups_source_scenario_and_id():
    n = 6
    starts = pd.date_range("2019-01-01", periods=n, freq="15min", tz="UTC")
    block = {
        "window_start": list(starts),
        "window_end": list(starts + pd.Timedelta(hours=1)),
        "score": [2.0] * n,
    }
    sim = pd.DataFrame({"source": "sim", "scenario_id": "s2", **block})
    bd = pd.DataFrame({"source": "battledim", "scenario_id": None, **block})
    scored = pd.concat([sim, bd], ignore_index=True).iloc[::-1]  # any order
    shares = np.tile([0.6, 0.3, 0.1], (len(scored), 1))
    out = run_alerts(scored, shares, SERIES, rule(m=3), ["flow__p1__residual"], None)
    assert list(out.columns) == list(ALERT_LAKE_COLUMNS)
    got = {(r.source, r.scenario_id): r for r in out.itertuples(index=False)}
    assert set(got) == {("sim", "s2"), ("battledim", None)}
    start_s = int(starts[2].timestamp()) + 3600  # m-th window's end
    for (source, scenario), r in got.items():
        assert r.alert_id == alert_id_for("batch", "kmeans", source, scenario, start_s)
        assert r.est_loss_m3h is None  # no feature rows: no excess inflow
    # feature rows with every inflow value missing give no estimate either (not 0)
    feats = pd.DataFrame({"flow__p1__residual": [np.nan] * len(scored)})
    out2 = run_alerts(scored, shares, SERIES, rule(m=3), ["flow__p1__residual"], feats)
    assert out2["est_loss_m3h"].isna().all()
