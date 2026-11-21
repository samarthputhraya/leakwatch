"""Control-room screen (D-052): what the page shows for each runner and alert state."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from leakwatch.dashboard import view

TZ = "Asia/Kolkata"
NOW = datetime(2026, 11, 21, 14, 20, 30, tzinfo=UTC)
CFG = {
    "project": {"display_timezone": TZ},
    "dashboard": {
        "stale_after_s": 15,
        "starting_stale_after_s": 180,
        "ring_floor_m": 0.7,
        "score_axis_min": 0.01,
    },
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


def test_starting_may_stay_silent_longer_than_running():
    quiet = state("starting", age_s=60)
    assert view.runner_view(quiet, NOW, 15, 180)["code"] == "starting"
    assert view.runner_view(state("starting", age_s=200), NOW, 15, 180)["code"] == "stale"
    assert view.runner_view(state("warming up", age_s=60), NOW, 15, 180)["code"] == "stale"
    out = view.live_view(state=quiet, alerts=[], now=NOW, cfg=CFG)
    assert out["runner"]["code"] == "starting" and out["banner"]["kind"] == "waiting"


def test_a_runner_simulating_a_leak_is_given_time_before_it_counts_as_silent():
    busy = state("running", age_s=40, pending_leaks=[{"leak_id": "a", "node_id": "n187"}])
    out = view.live_view(state=busy, alerts=[], now=NOW, cfg=CFG)
    assert out["runner"]["code"] == "running" and out["leaks"]["pending"][0]["node_id"] == "n187"
    assert view.runner_view(state("running", age_s=40), NOW, 15, 180)["code"] == "stale"
    assert view.runner_view(state("running", age_s=400), NOW, 15, 180, busy=True)["code"] == "stale"


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


def test_alerts_of_the_previous_replay_are_not_shown_while_the_runner_starts():
    out = view.live_view(state=state("starting"), alerts=[ALERT], now=NOW, cfg=CFG)
    assert out["open_alerts"] == 0 and out["banner"]["title"] == "Starting"
    out = view.live_view(state=state("warming up"), alerts=[ALERT], now=NOW, cfg=CFG)
    assert out["open_alerts"] == 1  # once the lake is this replay's, its alerts count


def test_live_view_without_a_runner_has_no_clock_and_no_false_all_clear():
    out = view.live_view(state={}, alerts=[], now=NOW, cfg=CFG)
    assert out["clock"] is None and out["runner"]["code"] == "offline"
    assert out["banner"]["kind"] == "unknown"


# The demo run of 21 Nov 2026, 16:03 UTC (leak at n470), as the state file and the lake hold it.
LEAK = {
    "leak_id": "demo-21d0e4",
    "node_id": "n470",
    "diameter_m": 0.03,
    "requested_at": "2026-11-21T16:02:59.222032+00:00",
    "est_flow_m3h": 57.03,
    "start_ts": "2018-11-19T19:35:00+00:00",
}
RUN_ALERT = {
    "alert_id": "5bab5c3a",
    "status": "open",
    "area": "A",
    "start_ts": "2018-11-19T20:15:00+00:00",
    "kmeans_score_max": 108.14352,
    "threshold": 1.3664359811,
    "est_loss_m3h": 53.966,
    "top_sensors": [
        {"sensor_id": "pressure__n469", "contribution": 0.7899},
        {"sensor_id": "pressure__n114", "contribution": 0.0838},
        {"sensor_id": "pressure__n516", "contribution": 0.0358},
    ],
    "gnn_top5": [
        {"node_id": "n469", "prob": 1.0},
        {"node_id": "n470", "prob": 0.0},
        {"node_id": "n471", "prob": 0.0},
        {"node_id": "n94", "prob": 0.0},
        {"node_id": "n468", "prob": 0.0},
    ],
    "created_at": "2026-11-21T16:03:13.791235+00:00",
    "updated_at": "2026-11-21T16:04:53.129395+00:00",
}
VERSIONS = [
    {"alert_id": "5bab5c3a", "updated_at": "2026-11-21T16:03:13.791235+00:00", "located": True},
    {"alert_id": "5bab5c3a", "updated_at": "2026-11-21T16:03:20.347019+00:00", "located": True},
]


def test_sensor_ids_lose_their_column_prefix_and_suffix():
    assert view.sensor_id("pressure__n469") == "n469"
    assert view.sensor_id("pressure__n469__residual") == "n469"
    assert view.sensor_id("n469") == "n469"


def test_drops_are_metres_below_the_usual_level_and_hidden_under_the_floor():
    drops = view.drops_view(
        "2018-11-23T06:30:00Z",
        {"n469": -1.994, "n114": -0.81, "n516": -0.69, "n54": 0.4, "n1": None, "n4": float("nan")},
        0.7,
        TZ,
        ["n469", "n114", "n516"],
    )
    assert drops["values"] == {
        "n469": 1.99,
        "n114": 0.81,
        "n516": 0.0,
        "n54": 0.0,
        "n1": 0.0,
        "n4": 0.0,
    }
    assert drops["shown"] == 2 and drops["time"] == "12:00 IST" and drops["floor_m"] == 0.7
    assert drops["labels"] == [
        {"sensor": "n469", "text": "n469  −2.0 m"},
        {"sensor": "n114", "text": "n114  −0.8 m"},
    ]
    assert view.drops_view(None, {"n469": -2.0}, 0.7, TZ) is None


def test_an_alert_belongs_to_the_newest_leak_that_began_before_it():
    early = {**LEAK, "leak_id": "early", "start_ts": "2018-11-19T08:00:00+00:00"}
    after = {**LEAK, "leak_id": "after", "start_ts": "2018-11-20T08:00:00+00:00"}
    assert view.match_leak(RUN_ALERT, [early, LEAK, after])["leak_id"] == "demo-21d0e4"
    assert view.match_leak(RUN_ALERT, [after]) is None
    assert view.match_leak({"start_ts": None}, [LEAK]) is None


def test_incident_of_the_demo_run():
    out = view.incident_view(RUN_ALERT, [LEAK], VERSIONS, TZ)
    assert out["started"] == "01:45 IST" and out["area"] == "A"
    assert out["after_click_s"] == 15  # 16:03:13.8 - 16:02:59.2
    assert out["after_leak_min"] == 40  # 19:35 -> 20:15 simulated
    assert out["ratio"] == 79
    assert out["sensors"] == [
        {"id": "n469", "kind": "pressure", "label": "n469", "share_pct": 79},
        {"id": "n114", "kind": "pressure", "label": "n114", "share_pct": 8},
        {"id": "n516", "kind": "pressure", "label": "n516", "share_pct": 4},
    ]
    assert out["candidates"] == ["n469", "n470", "n471", "n94", "n468"]  # rank only
    assert out["located_after_s"] == 15  # the first version already had candidates
    assert out["loss_now_m3h"] == 54.0
    assert out["focus"] == out["candidates"]


def test_incident_without_candidates_or_a_leak_from_this_screen():
    bare = {**RUN_ALERT, "gnn_top5": None, "top_sensors": None, "est_loss_m3h": None}
    out = view.incident_view(bare, [], [], TZ)
    assert out["candidates"] == [] and out["located_after_s"] is None
    assert out["after_click_s"] is None and out["after_leak_min"] is None
    assert out["sensors"] == [] and out["loss_now_m3h"] is None and out["leak_id"] is None
    later = {**VERSIONS[1], "located": True}
    out = view.incident_view(RUN_ALERT, [LEAK], [{**VERSIONS[0], "located": False}, later], TZ)
    assert out["located_after_s"] == 21  # the localizer answered with the second version


def test_live_view_shows_the_newest_open_alert_and_the_newest_window():
    older = {**RUN_ALERT, "alert_id": "old", "start_ts": "2018-11-19T02:00:00+00:00"}
    residuals = [
        {"window_end": "2018-11-23T06:15:00Z", "n469": -1.9, "n114": -0.3},
        {"window_end": "2018-11-23T06:30:00Z", "n469": -2.0, "n114": -0.75},
    ]
    out = view.live_view(
        state=state(leaks=[LEAK]),
        alerts=[older, RUN_ALERT],
        now=NOW,
        cfg=CFG,
        versions=VERSIONS,
        residuals=residuals,
    )
    assert out["incident"]["alert_id"] == "5bab5c3a" and out["incident"]["after_click_s"] == 15
    assert out["drops"]["values"] == {"n469": 2.0, "n114": 0.75}
    assert [label["sensor"] for label in out["drops"]["labels"]] == ["n469", "n114"]
    quiet = view.live_view(state=state(), alerts=[], now=NOW, cfg=CFG, residuals=residuals)
    assert quiet["incident"] is None and quiet["drops"]["labels"] == []
    starting = view.live_view(
        state=state("starting"), alerts=[RUN_ALERT], now=NOW, cfg=CFG, residuals=residuals
    )
    assert starting["incident"] is None and starting["drops"] is None  # the previous replay's


def windows(hours: float, start: str = "2018-11-19T18:00:00Z") -> list[datetime]:
    first = view.to_utc(start)
    return [first + timedelta(minutes=15 * k) for k in range(int(hours * 4))]


def test_charts_mark_the_leak_and_the_alert_and_keep_real_scores_for_the_table():
    ends = windows(6)  # 18:00 .. 23:45 UTC; leak 19:35, alert 20:15
    scores = [
        {
            "window_end": t,
            "score": 0.004 if k == 0 else (0.5 if t < ends[9] else 104.2),
            "threshold": 1.3664,
        }
        for k, t in enumerate(ends)
    ]
    residuals = [
        {"window_end": t, "n469": -2.0, "n114": -0.5, "n516": None, "n1": -3.0} for t in ends
    ]
    incident = view.incident_view(RUN_ALERT, [LEAK], VERSIONS, TZ)
    out = view.charts_view(scores, residuals, incident, [LEAK], TZ, 0.01)
    assert out["tz_name"] == "IST" and out["start"] == view.epoch_ms(ends[0])
    assert [(m["kind"], m["text"]) for m in out["marks"]] == [
        ("leak", "Leak starts"),
        ("alert", "Alert"),
    ]
    assert out["marks"][0]["at"] == view.epoch_ms(view.to_utc(LEAK["start_ts"]))
    first = out["score"]["points"][0]
    assert first[1] == 0.01 and first[2] == 0.004  # drawn at the axis floor, real value kept
    assert out["score"]["threshold_text"] == "Alert threshold 1.37"
    assert out["score"]["table"]["rows"][0] == ["Tue 20 Nov, 05:15 IST", "104.20", "above"]
    assert out["score"]["table"]["rows"][-1][2] == "below"  # newest first, oldest last
    pressure = out["pressure"]
    assert [s["sensor"] for s in pressure["series"]] == ["n469", "n114", "n516"]  # the alert's
    assert pressure["why"] == "the alert's sensors"
    assert pressure["series"][2]["points"][0][1] is None  # a missing reading stays a gap
    assert pressure["table"]["columns"] == ["Time", "n469 (m)", "n114 (m)", "n516 (m)"]
    assert pressure["table"]["rows"][0][1:] == ["−2.00", "−0.50", ""]


def test_charts_without_an_alert_show_the_largest_drops_and_no_marks():
    ends = windows(2)
    residuals = [
        {"window_end": t, "n1": -0.1, "n4": -0.9, "n31": 0.2, "n54": -0.4, "n105": -0.6}
        for t in ends
    ]
    out = view.charts_view([], residuals, None, [], TZ, 0.01)
    assert [s["sensor"] for s in out["pressure"]["series"]] == ["n4", "n105", "n54"]
    assert out["pressure"]["why"] == "the deepest drops"
    assert out["marks"] == [] and out["score"]["points"] == [] and out["score"]["threshold"] is None
    assert view.charts_view([], [], None, [], TZ, 0.01) is None
    later = {**LEAK, "start_ts": "2018-11-25T00:00:00Z"}  # outside the shown hours: no mark
    assert view.charts_view([], residuals, None, [later], TZ, 0.01)["marks"] == []


# The alert of the live run at n300 (21 Nov 2026, 17:0x UTC): mostly flow sensors moved.
N300_SENSORS = [
    {"sensor_id": "flow__p227", "contribution": 0.78},
    {"sensor_id": "flow__PUMP_1", "contribution": 0.08},
    {"sensor_id": "pressure__n296", "contribution": 0.03},
]


def test_flow_sensors_are_named_as_flows_and_get_no_ring_label():
    alert = {**RUN_ALERT, "top_sensors": N300_SENSORS}
    out = view.incident_view(alert, [LEAK], VERSIONS, TZ)
    assert [s["label"] for s in out["sensors"]] == ["p227 (flow)", "PUMP_1 (flow)", "n296"]
    assert view.sensor_kind("level__T1") == "level" and view.sensor_kind("n1") == "pressure"
    live = view.live_view(
        state=state(leaks=[LEAK]),
        alerts=[alert],
        now=NOW,
        cfg=CFG,
        residuals=[{"window_end": "2018-11-21T07:45:00Z", "n296": -0.9, "p227": -5.0}],
    )
    assert [label["sensor"] for label in live["drops"]["labels"]] == ["n296"]
    assert live["analysed"] == "Wed 21 Nov, 13:15 IST"


def test_pressure_chart_takes_the_alerts_pressure_sensors_first_then_the_largest_drops():
    alert = {**RUN_ALERT, "top_sensors": N300_SENSORS}
    incident = view.incident_view(alert, [LEAK], VERSIONS, TZ)
    rows = [
        {"window_end": t, "n296": -0.1, "n31": -0.6, "n54": -0.2, "n4": -0.4} for t in windows(2)
    ]
    out = view.charts_view([], rows, incident, [LEAK], TZ, 0.01)["pressure"]
    assert [s["sensor"] for s in out["series"]] == ["n296", "n31", "n4"]
    assert out["why"] == "the alert's, then the deepest"


# Work orders of the same run, as the lake holds them (text fields shortened).
SOP = {
    "doc": "LeakWatch leak response standard operating procedure",
    "chapter": "6. Confirm the leak before digging",
    "page": "n/a",
}
ORDER = {
    "priority": "P1",
    "pipes_to_inspect": ["p437", "p438", "p439", "p440", "p95", "p435", "p436"],
    "recommended_actions": [
        "Check the top sensors' recent readings.",
        "Walk the pipes.",
        "Confirm.",
        "Isolate.",
        "Record.",
    ],
    "manual_citations": [SOP, SOP],
    "summary": "A possible leak was detected in area A.",
}
TEMPLATE = {
    "alert_id": "5bab5c3a",
    "generator": "template",
    "generated_at": "2026-11-21T16:03:17.146328Z",
    "work_order_json": json.dumps({**ORDER, "estimated_loss_m3_per_hour": 40.814}),
}
MODEL = {
    "alert_id": "5bab5c3a",
    "generator": "llm",
    "generated_at": "2026-11-21T16:04:02.889945Z",
    "work_order_json": json.dumps({**ORDER, "estimated_loss_m3_per_hour": 56.037}),
}
LOSS_VERSIONS = [
    {
        "updated_at": "2026-11-21T16:03:13.791Z",
        "est_loss_m3h": 40.814,
        "last_window_end": "2018-11-19T21:00:00Z",
    },
    {
        "updated_at": "2026-11-21T16:03:57.372Z",
        "est_loss_m3h": 56.037,
        "last_window_end": "2018-11-21T10:00:00Z",
    },
    {
        "updated_at": "2026-11-21T16:04:05.530Z",
        "est_loss_m3h": 55.622,
        "last_window_end": "2018-11-21T14:00:00Z",
    },
]


def test_order_card_before_any_order_and_when_none_can_be_read():
    assert view.order_view([], LOSS_VERSIONS, LEAK, TZ) == {"state": "writing"}
    broken = {**TEMPLATE, "work_order_json": "{not json"}
    assert view.order_view([broken], LOSS_VERSIONS, LEAK, TZ) == {"state": "unreadable"}
    assert (
        view.order_view([{**TEMPLATE, "work_order_json": None}], [], LEAK, TZ)["state"]
        == "unreadable"
    )


def test_template_order_comes_first_with_its_own_loss_time():
    out = view.order_view([TEMPLATE], LOSS_VERSIONS, LEAK, TZ)
    assert out["state"] == "template" and out["priority"] == "P1"
    assert out["loss_m3h"] == 40.8 and out["loss_at"] == "02:30 IST"  # version 1, 21:00 UTC
    assert out["version"] == (
        "Template version, 18 s after the click. The model's version replaces it when it arrives."
    )


def test_model_order_replaces_the_template_and_names_both_times():
    out = view.order_view([MODEL, TEMPLATE], LOSS_VERSIONS, LEAK, TZ)
    assert out["state"] == "model"
    assert out["loss_m3h"] == 56.0 and out["loss_at"] == "15:30 IST"  # version 5
    assert out["version"] == "Model version, 64 s after the click; template version at 18 s"
    assert out["pipes"] == ORDER["pipes_to_inspect"] and len(out["actions"]) == 5
    assert out["sources"] == [  # the duplicate merged, no "p. n/a"
        f"{SOP['doc']}, section 6. Confirm the leak before digging"
    ]
    broken = {**MODEL, "work_order_json": "{not json"}
    assert view.order_view([TEMPLATE, broken], LOSS_VERSIONS, LEAK, TZ)["state"] == "template"


def test_loss_time_falls_back_to_the_newest_version_before_the_order():
    at = view.loss_time(99.9, MODEL["generated_at"], LOSS_VERSIONS)
    assert at == datetime(2018, 11, 21, 10, 0, tzinfo=UTC)
    assert view.loss_time(None, None, LOSS_VERSIONS) is None
    assert view.sources_view([{"doc": "CPHEEO manual", "chapter": "15.3", "page": "412"}, "x"]) == [
        "CPHEEO manual, section 15.3, p. 412"
    ]


def test_live_view_puts_the_order_of_the_shown_alert_on_the_incident():
    other = {**MODEL, "alert_id": "someone else"}
    out = view.live_view(
        state=state(leaks=[LEAK]),
        alerts=[RUN_ALERT],
        now=NOW,
        cfg=CFG,
        versions=VERSIONS,
        orders=[TEMPLATE, other],
    )
    assert out["incident"]["order"]["state"] == "template"


# Pipe distances from n470 in L-Town, metres (Geometry.pipe_distance_m, measured 21 Nov 2026).
N470 = {"n469": 49.0, "n470": 0.0, "n471": 38.3, "n94": 108.0, "n468": 169.2}


def test_reveal_of_the_n470_run_names_rank_distance_and_both_flows():
    out = view.truth_view(
        {**LEAK, "est_flow_m3h": 57.03},
        ["n469", "n470", "n471", "n94", "n468"],
        54.0,
        lambda a, b: N470[b],
    )
    assert out == {
        "node_id": "n470",
        "rank": 2,
        "of": 5,
        "to_first_m": 49,
        "nearest_m": 0,
        "nearest_rank": 2,
        "flow_m3h": 57.0,
        "loss_now_m3h": 54.0,
    }


def test_reveal_when_the_leak_is_not_a_candidate_or_cannot_be_reached():
    far = {"a": 300.0, "b": 120.0, "c": None}
    out = view.truth_view({**LEAK, "node_id": "n300"}, ["a", "b", "c"], None, lambda x, y: far[y])
    assert out["rank"] is None and out["to_first_m"] == 300
    assert out["nearest_m"] == 120 and out["nearest_rank"] == 2
    none = view.truth_view(LEAK, ["a"], None, lambda x, y: None)
    assert none["to_first_m"] is None and none["nearest_m"] is None


def test_live_view_adds_the_truth_only_with_a_leak_candidates_and_distances():
    common = {"state": state(leaks=[LEAK]), "alerts": [RUN_ALERT], "now": NOW, "cfg": CFG}
    assert view.live_view(**common)["incident"]["truth"] is None  # no geometry yet
    out = view.live_view(**common, distance=lambda a, b: N470[b])
    assert out["incident"]["truth"]["rank"] == 2
    alone = view.live_view(**{**common, "state": state()}, distance=lambda a, b: 1.0)
    assert alone["incident"]["truth"] is None  # no leak from this screen to check against
