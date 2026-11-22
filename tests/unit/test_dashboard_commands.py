"""Control-room screen (D-052): leak requests, replay controls and their failure messages."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from leakwatch.common.config import load_config
from leakwatch.dashboard import api, view

CFG = load_config(environ={})
TZ = "Asia/Kolkata"
PAGE = {"base_url": "http://localhost", "headers": {"X-Requested-With": "leakwatch"}}
NOW = datetime(2026, 11, 21, 14, 20, 30, tzinfo=UTC)
GEOMETRY = SimpleNamespace(
    payload={"junctions": ["n100", "n470"], "nodes": {"n470": [224.6, 728.1]}}
)
LEAK = {
    "leak_id": "demo-97c88e",
    "node_id": "n470",
    "diameter_m": 0.03,
    "requested_at": "2026-11-21T14:20:09.492609+00:00",
    "est_flow_m3h": 57.28,
    "start_ts": "2018-11-21T20:05:00+00:00",
}


def vm(code: str = "running") -> dict:
    return {"runner": {"code": code, "label": code, "detail": "", "live": code == "running"}}


def app(sent: list, code: str = "running", sender=None) -> TestClient:
    def record(command: str, **params) -> None:
        sent.append((command, params))

    return TestClient(
        api.create_app(
            lambda: vm(code),
            background=False,
            cfg=CFG,
            sender=sender or record,
            geometry_loader=lambda: GEOMETRY,
        ),
        **PAGE,
    )


def test_inject_sends_the_leak_command_in_metres():
    sent: list = []
    res = app(sent).post("/api/demo/inject", json={"node_id": "n470", "diameter_mm": 30})
    assert res.status_code == 200 and res.json() == {"ok": True}
    assert sent == [("inject_leak", {"node_id": "n470", "diameter_m": 0.03})]


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        ({"node_id": "n470", "diameter_mm": 5}, 422, "Leak size must be between 10 and 50 mm."),
        ({"node_id": "n470", "diameter_mm": 80}, 422, "Leak size must be between 10 and 50 mm."),
        ({"node_id": "R1", "diameter_mm": 30}, 422, "R1 is not a junction of L-Town."),
    ],
)
def test_inject_refuses_what_the_runner_cannot_simulate(body, status, message):
    sent: list = []
    res = app(sent).post("/api/demo/inject", json=body)
    assert res.status_code == status and res.json() == {"ok": False, "message": message}
    assert sent == []


@pytest.mark.parametrize("code", ["starting", "warming", "finished", "offline", "stale"])
def test_inject_waits_for_a_running_replay(code):
    sent: list = []
    res = app(sent, code).post("/api/demo/inject", json={"node_id": "n470", "diameter_mm": 30})
    assert res.status_code == 409 and "not running yet" in res.json()["message"]
    assert sent == []


def test_a_command_kafka_did_not_take_is_reported_in_plain_words():
    def down(command: str, **params) -> None:
        raise api.CommandNotSentError(command)

    res = app([], sender=down).post("/api/demo/inject", json={"node_id": "n470", "diameter_mm": 30})
    assert res.status_code == 503
    assert res.json() == {"ok": False, "message": api.KAFKA_DOWN}
    assert "make up" in api.KAFKA_DOWN

    def crash(command: str, **params) -> None:
        raise ConnectionError("broker transport failure: localhost:9092")

    res = app([], sender=crash).post("/api/demo/restart")
    assert res.status_code == 503 and "9092" not in res.text


def test_speed_and_restart():
    sent: list = []
    c = app(sent)
    assert c.post("/api/demo/speed", json={"speed": 1440}).json() == {"ok": True}
    assert c.post("/api/demo/speed", json={"speed": 999}).status_code == 422
    assert c.post("/api/demo/restart").json() == {"ok": True}
    assert sent == [("set_speed", {"speed": 1440}), ("reset", {})]
    assert app([], "finished").post("/api/demo/restart").status_code == 200
    offline = app([], "offline").post("/api/demo/restart")
    assert offline.status_code == 409 and "make demo" in offline.json()["message"]


def test_settings_offer_the_configured_sizes_and_speeds():
    body = app([]).get("/api/settings").json()
    assert body["leak_mm"] == {"min": 10, "max": 50, "default": 30}
    assert body["speeds"] == [180, 360, 720, 1440, 2880]
    # The localizer's measured hit rates come from reports/metrics/dl_localizer.json.
    assert "not one answer" in body["localizer"]["sentence"]
    assert 0 < body["localizer"]["in_top5_pct"] <= 100
    assert app([], "finishing").post("/api/demo/restart").status_code == 200
    refused = app([], "finishing").post(
        "/api/demo/inject", json={"node_id": "n100", "diameter_mm": 30}
    )
    assert refused.status_code == 409


def test_network_endpoint_serves_the_geometry_or_says_what_to_run():
    assert app([]).get("/api/network").json()["junctions"] == ["n100", "n470"]

    def missing():
        raise FileNotFoundError("/data/L-TOWN.inp missing: run `make data`")

    c = TestClient(
        api.create_app(lambda: vm(), background=False, cfg=CFG, geometry_loader=missing), **PAGE
    )
    res = c.get("/api/network")
    assert res.status_code == 503 and res.json()["message"].endswith("Run `make data`.")
    assert "/data/" not in res.text


def test_send_demo_command_raises_when_kafka_left_the_message_unsent(monkeypatch):
    from leakwatch.demo import runner

    seen = []
    monkeypatch.setattr(runner, "send_command", lambda command, **p: seen.append((command, p)) or 0)
    api.send_demo_command("reset")
    assert seen == [("reset", {})]
    monkeypatch.setattr(runner, "send_command", lambda command, **p: 1)
    with pytest.raises(api.CommandNotSentError):
        api.send_demo_command("reset")


# ------------------------------------------------------------------------------------- leaks


def test_leak_view_rounds_and_converts_to_display_time():
    assert view.leak_view(LEAK, TZ) == {
        "leak_id": "demo-97c88e",
        "node_id": "n470",
        "diameter_mm": 30,
        "start_ts": "2018-11-21T20:05:00+00:00",
        "started": "01:35 IST",
        "requested_at": "2026-11-21T14:20:09.492609+00:00",
        "flow_m3h": 57.3,
    }
    pending = view.leak_view({"leak_id": "demo-1", "node_id": "n100", "diameter_m": 0.02}, TZ)
    assert (
        pending["started"] is None and pending["flow_m3h"] is None and pending["diameter_mm"] == 20
    )


def test_leaks_view_separates_pending_active_and_failed():
    state = {
        "pending_leaks": [{"leak_id": "p1", "node_id": "n100", "diameter_m": 0.03}],
        "leaks": [LEAK],
    }
    out = view.leaks_view(state, [{"leak_id": "f1", "node_id": "n300", "diameter_m": 0.03}], TZ)
    assert [x["node_id"] for x in out["pending"]] == ["n100"]
    assert [x["node_id"] for x in out["active"]] == ["n470"]
    assert [x["node_id"] for x in out["failed"]] == ["n300"]
    assert view.leaks_view({}, [], TZ) == {"pending": [], "active": [], "failed": []}
    both = view.leaks_view(state, [{"leak_id": "p1", "node_id": "n100"}], TZ)
    assert both["pending"] == []  # a failed leak is not also shown as being simulated


def st(pending=(), leaks=(), sim="2018-11-21T20:00:00+00:00", status="running") -> dict:
    return {"status": status, "sim_ts": sim, "pending_leaks": list(pending), "leaks": list(leaks)}


def test_a_leak_that_starts_is_not_a_failure():
    track = api.FailedLeaks(60)
    pending = {"leak_id": "a", "node_id": "n470", "diameter_m": 0.03}
    assert track.update(st([pending]), NOW) == []
    assert track.update(st(leaks=[{**pending, "start_ts": "x"}]), NOW) == []


def test_a_leak_that_vanishes_without_starting_is_shown_for_a_while():
    track = api.FailedLeaks(60)
    pending = {"leak_id": "a", "node_id": "n470", "diameter_m": 0.03}
    track.update(st([pending]), NOW)
    failed = track.update(st(), NOW + timedelta(seconds=2))
    assert [f["node_id"] for f in failed] == ["n470"]
    assert track.update(st(), NOW + timedelta(seconds=40)) == failed
    assert track.update(st(), NOW + timedelta(seconds=90)) == []


def test_a_leak_with_an_error_is_failed_at_once_and_listed_once():
    track = api.FailedLeaks(60)
    broken = {"leak_id": "b", "node_id": "n100", "diameter_m": 0.03, "error": "EPANET error 110"}
    assert [f["leak_id"] for f in track.update(st([broken]), NOW)] == ["b"]
    assert [f["leak_id"] for f in track.update(st(), NOW + timedelta(seconds=1))] == ["b"]


def test_a_restart_clears_pending_leaks_without_calling_them_failed():
    track = api.FailedLeaks(60)
    pending = {"leak_id": "a", "node_id": "n470", "diameter_m": 0.03}
    track.update(st([pending], sim="2018-11-21T20:00:00+00:00"), NOW)
    assert track.update(st(sim="2018-11-19T00:10:00+00:00"), NOW) == []  # the replay rewound
    track.update(st([pending]), NOW)
    assert track.update(st(status="starting"), NOW) == []
    track.update(st([pending]), NOW)
    assert track.update({}, NOW) == []


def test_unavailable_view_has_every_key_the_page_reads():
    out = view.unavailable_view(NOW)
    assert out["banner"]["kind"] == "unknown" and out["runner"]["code"] == "error"
    assert out["leaks"] == {"pending": [], "active": [], "failed": []} and out["clock"] is None


# ------------------------------------------------------------ commands come from the page only


def test_a_command_without_the_page_header_is_refused_and_nothing_is_sent():
    sent: list = []
    c = app(sent)
    res = c.post("/api/demo/restart", headers={"X-Requested-With": ""})
    assert res.status_code == 403 and res.json() == {"ok": False, "message": api.NOT_FROM_PAGE}
    assert sent == []


@pytest.mark.parametrize(
    "extra",
    [
        {"Origin": "http://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://localhost:9"},
    ],
)
def test_a_command_from_another_site_is_refused(extra):
    sent: list = []
    res = app(sent).post(
        "/api/demo/inject", json={"node_id": "n470", "diameter_mm": 30}, headers=extra
    )
    assert res.status_code == 403 and sent == []


def test_the_page_itself_may_send_commands_and_anyone_may_read():
    sent: list = []
    c = app(sent)
    same = {"Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"}
    assert c.post("/api/demo/restart", headers=same).status_code == 200
    assert sent == [("reset", {})]
    assert (
        c.get(
            "/api/live", headers={"X-Requested-With": "", "Sec-Fetch-Site": "cross-site"}
        ).status_code
        == 200
    )


def test_only_local_host_names_are_served():
    c = app([])
    assert c.get("/api/health", headers={"Host": "127.0.0.1:8501"}).status_code == 200
    assert c.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
    assert c.post("/api/demo/restart", headers={"Host": "evil.example"}).status_code == 400


def test_from_the_page_rules():
    ok = {"x-requested-with": "leakwatch", "host": "localhost:8501"}
    assert api.from_the_page("GET", {})
    assert api.from_the_page("POST", ok)
    assert api.from_the_page(
        "POST", {**ok, "origin": "http://localhost:8501", "sec-fetch-site": "same-origin"}
    )
    assert not api.from_the_page("POST", {"host": "localhost:8501"})
    assert not api.from_the_page("POST", {**ok, "origin": "http://localhost:8502"})
    assert not api.from_the_page("DELETE", {**ok, "sec-fetch-site": "same-site"})
