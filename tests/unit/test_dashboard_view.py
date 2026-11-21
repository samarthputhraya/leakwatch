"""Control-room screen (D-052): what the page shows for each runner and alert state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from leakwatch.dashboard import view

TZ = "Asia/Kolkata"
NOW = datetime(2026, 11, 21, 14, 20, 30, tzinfo=UTC)
CFG = {
    "project": {"display_timezone": TZ},
    "dashboard": {"stale_after_s": 15},
    "demo": {"background_start": "2018-11-19 00:00", "background_hours": 168},
}
# The alert of the demo run of 21 Nov 2026 (leak at n470), as the lake returned it.
ALERT = {
    "alert_id": "6b6eaace-c106-5dee-897f-b1fc55e4e824",
    "status": "open",
    "area": "A",
    "start_ts": "2018-11-21T20:45:00+00:00",
    "kmeans_score_max": 106.0730597617,
    "threshold": 1.3664359811,
}


def state(status: str = "running", age_s: float = 1.0, **extra) -> dict:
    return {
        "status": status,
        "sim_ts": "2018-11-22T05:30:00+00:00",
        "speed": 2880.0,
        "background_start": "2018-11-19 00:00",
        "updated_at": (NOW - timedelta(seconds=age_s)).isoformat(),
        **extra,
    }


def test_times_are_shown_in_the_display_time_zone():
    ts = datetime(2018, 11, 21, 20, 45, tzinfo=UTC)
    assert view.fmt_time(ts, TZ) == "Thu 22 Nov, 02:15 IST"
    assert view.fmt_time(ts, TZ, date=False) == "02:15 IST"
    assert view.fmt_time(ts, "UTC") == "Wed 21 Nov, 20:45 UTC"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0 s"),
        (9.4, "9 s"),
        (59.6, "1 min 00 s"),
        (113.1, "1 min 53 s"),
        (7800, "2 h 10 min"),
        (-3, "0 s"),
    ],
)
def test_durations_read_naturally(seconds, text):
    assert view.fmt_duration(seconds) == text


def test_to_utc_accepts_what_the_state_file_and_the_lake_return():
    expected = datetime(2018, 11, 21, 20, 45, tzinfo=UTC)
    assert view.to_utc("2018-11-21T20:45:00Z") == expected
    assert view.to_utc("2018-11-21 20:45") == expected  # naive strings are UTC
    assert view.to_utc("2018-11-22T02:15:00+05:30") == expected
    assert view.to_utc(pd.Timestamp("2018-11-21 20:45", tz="UTC")) == expected
    assert view.to_utc(expected.replace(tzinfo=None)) == expected
    for missing in (None, "", "not a time", float("nan"), pd.NaT, 5):
        assert view.to_utc(missing) is None


def test_runner_states():
    assert view.runner_view({}, NOW, 15)["code"] == "offline"
    assert "make demo" in view.runner_view({}, NOW, 15)["detail"]
    assert view.runner_view(state("starting"), NOW, 15)["label"] == "Starting"
    assert view.runner_view(state("warming up"), NOW, 15)["code"] == "warming"
    running = view.runner_view(state("running"), NOW, 15)
    assert running == {"code": "running", "label": "Running", "detail": "", "live": True}
    assert view.runner_view(state("something new"), NOW, 15)["code"] == "unknown"


def test_a_runner_that_stopped_writing_is_reported_and_a_finished_replay_is_not():
    stale = view.runner_view(state("running", age_s=75), NOW, 15)
    assert stale["code"] == "stale" and stale["live"] is False
    assert "1 min 15 s ago" in stale["detail"]
    finished = view.runner_view(state("finished", age_s=600), NOW, 15)
    assert finished["code"] == "finished" and "Restart the replay" in finished["detail"]


def test_clock_shows_simulated_time_speed_and_what_is_left_of_the_replay():
    clock = view.clock_view(state(), CFG["demo"], TZ)
    assert clock["sim_time"] == "Thu 22 Nov, 11:00 IST"
    assert clock["speed"] == "2,880×"
    assert clock["replay_fraction"] == pytest.approx(77.5 / 168, abs=1e-4)
    assert clock["replay_left"] == "1 min 53 s"  # 90.5 simulated hours at 2880x
    assert view.clock_view({"status": "starting"}, CFG["demo"], TZ) is None


def test_clock_is_bounded_and_survives_a_missing_speed():
    late = view.clock_view(state(sim_ts="2018-11-27T00:00:00+00:00"), CFG["demo"], TZ)
    assert late["replay_fraction"] == 1.0 and late["replay_left"] == "0 s"
    paused = view.clock_view(state(speed=None), CFG["demo"], TZ)
    assert paused["speed"] is None and paused["replay_left"] is None


def banner(runner_state: dict, alerts=(), n=33, error=None) -> dict:
    return view.banner_view(view.runner_view(runner_state, NOW, 15), list(alerts), n, error, TZ)


def test_banner_says_no_open_alerts_only_when_the_screen_can_know():
    assert banner(state()) == {
        "kind": "normal",
        "title": "No open alerts",
        "sub": "Watching 33 pressure sensors",
    }
    assert banner({})["kind"] == "unknown"
    assert banner(state("running", age_s=90))["kind"] == "unknown"
    assert banner(state(), error="The demo lake could not be read.") == {
        "kind": "unknown",
        "title": "Alerts cannot be read",
        "sub": "The demo lake could not be read.",
    }
    assert banner(state("starting"))["kind"] == "waiting"
    assert banner(state("warming up"))["title"] == "Warming up"
    assert banner(state("finished"))["title"] == "Replay finished"


def test_banner_for_one_alert_names_area_time_and_score():
    assert banner(state(), [ALERT]) == {
        "kind": "alert",
        "title": "Alert in Area A",
        "sub": "since Thu 22 Nov, 02:15 IST, score 78 times the threshold",
    }
    assert (
        banner(state("finished"), [ALERT])["kind"] == "alert"
    )  # an open alert outlives the replay


def test_banner_for_several_alerts_counts_them_and_uses_the_earliest_start():
    later = {**ALERT, "area": "C", "start_ts": "2018-11-22T01:00:00+00:00", "kmeans_score_max": 3.0}
    both = banner(state(), [later, ALERT])
    assert both["title"] == "2 open alerts in Area A, C"
    assert both["sub"].startswith("since Thu 22 Nov, 02:15 IST")


def test_banner_tolerates_missing_numbers():
    bare = {
        "status": "open",
        "area": "B",
        "start_ts": None,
        "kmeans_score_max": None,
        "threshold": 0,
    }
    assert banner(state(), [bare]) == {"kind": "alert", "title": "Alert in Area B", "sub": ""}
    assert view.score_ratio({"kmeans_score_max": float("nan"), "threshold": 1.0}) is None


def test_live_view_counts_only_open_alerts():
    closed = {**ALERT, "status": "closed"}
    out = view.live_view(state=state(), alerts=[closed], now=NOW, cfg=CFG, n_pressure=33)
    assert out["open_alerts"] == 0 and out["banner"]["kind"] == "normal"
    out = view.live_view(state=state(), alerts=[closed, ALERT], now=NOW, cfg=CFG, n_pressure=33)
    assert out["open_alerts"] == 1 and out["banner"]["title"] == "Alert in Area A"
    assert out["runner"]["code"] == "running" and out["clock"]["speed"] == "2,880×"


def test_live_view_without_a_runner_has_no_clock_and_no_false_all_clear():
    out = view.live_view(state={}, alerts=[], now=NOW, cfg=CFG)
    assert out["clock"] is None and out["runner"]["code"] == "offline"
    assert out["banner"]["kind"] == "unknown"
