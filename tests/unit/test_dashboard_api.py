"""Control-room screen, server side (D-052): routes, snapshots, lake failure paths."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from leakwatch.common.config import load_config
from leakwatch.dashboard import api

CFG = load_config(environ={})
VM = {
    "runner": {"code": "running", "label": "Running", "detail": "", "live": True},
    "clock": {"sim_time": "Thu 22 Nov, 11:00 IST", "speed": "2,880×", "replay_fraction": 0.46},
    "banner": {"kind": "normal", "title": "No open alerts", "sub": "Watching 33 pressure sensors"},
    "open_alerts": 0,
    "error": None,
    "generated_at": datetime(2026, 11, 21, 14, 20, tzinfo=UTC),
}


def client(source=lambda: VM) -> TestClient:
    return TestClient(
        api.create_app(source, background=False, cfg=CFG),
        base_url="http://localhost",
        headers={"X-Requested-With": "leakwatch"},
    )


def test_live_returns_the_view_model_as_json():
    res = client().get("/api/live")
    assert res.status_code == 200
    body = res.json()
    assert body["banner"]["title"] == "No open alerts"
    assert body["generated_at"] == "2026-11-21T14:20:00Z"


def test_the_page_and_its_files_are_served_and_revalidated():
    c = client()
    page = c.get("/")
    assert page.status_code == 200 and "<title>LeakWatch control room</title>" in page.text
    assert page.headers["cache-control"] == "no-cache"
    for path in ("app.js", "styles.css", "tokens.css", "vendor/htm-preact.module.js"):
        res = c.get(f"/static/{path}")
        assert res.status_code == 200, path
        assert res.headers["cache-control"] == "no-cache"
    assert (
        c.get("/static/vendor/deck.gl.min.js").headers["content-type"].startswith("text/javascript")
    )
    assert c.get("/api/health").json() == {"ok": True}
    assert c.get("/api/settings").json()["poll_ms"] == CFG["dashboard"]["poll_ms"]


def test_a_failing_source_becomes_a_visible_state_not_an_error_page():
    def broken() -> dict:
        raise RuntimeError("boom")

    res = client(broken).get("/api/live")
    assert res.status_code == 200
    body = res.json()
    assert body["banner"]["kind"] == "unknown"
    assert body["banner"]["title"] == "Screen data unavailable"
    assert "boom" not in res.text  # no internals on screen


def test_background_snapshots_serve_the_last_result():
    calls = []

    def source() -> dict:
        calls.append(1)
        return {**VM, "open_alerts": len(calls)}

    with TestClient(
        api.create_app(source, background=True, cfg=CFG),
        base_url="http://localhost",
        headers={"X-Requested-With": "leakwatch"},
    ) as c:
        first = c.get("/api/live").json()["open_alerts"]
        again = c.get("/api/live").json()["open_alerts"]
    assert first >= 1 and again >= first


def test_plain_turns_lake_values_into_json_ready_python():
    row = {
        "top_sensors": np.array(
            [{"sensor_id": "pressure__n469", "contribution": np.float64(0.79)}], dtype=object
        ),
        "est_loss_m3h": np.float64("nan"),
        "start_ts": pd.Timestamp("2018-11-21 20:45", tz="UTC"),
        "end_ts": pd.NaT,
        "windows_above": np.int64(7),
        "area": "A",
        "empty": np.array([], dtype=object),
    }
    out = api.plain(row)
    assert out["top_sensors"] == [{"sensor_id": "pressure__n469", "contribution": 0.79}]
    assert out["est_loss_m3h"] is None and out["end_ts"] is None and out["empty"] == []
    assert out["start_ts"] == datetime(2018, 11, 21, 20, 45, tzinfo=UTC)
    assert out["windows_above"] == 7 and isinstance(out["windows_above"], int)
    assert api.json_response(out).body  # serialisable


def test_live_source_reports_an_unreadable_lake_instead_of_an_empty_one(monkeypatch):
    from leakwatch.demo import runner

    source = api.LiveSource(CFG)
    now = datetime.now(UTC).isoformat()
    monkeypatch.setattr(runner, "read_state", lambda: {"status": "running", "updated_at": now})
    monkeypatch.setattr(source, "_pressure_sensors", lambda: 33)

    def unreadable() -> list:
        raise OSError("No files found")

    monkeypatch.setattr(source, "_alerts", unreadable)
    out = source()
    assert out["error"] and out["banner"] == {
        "kind": "unknown",
        "title": "Alerts cannot be read",
        "sub": "The demo lake could not be read. It is retried every second.",
    }

    monkeypatch.setattr(source, "_alerts", lambda: [])
    out = source()
    assert out["error"] is None and out["banner"]["title"] == "No open alerts"
    assert out["build_ms"] >= 0


def test_live_source_reads_nothing_from_a_lake_without_alerts(monkeypatch, tmp_path):
    from leakwatch.demo import runner

    monkeypatch.setattr(runner, "demo_lake", lambda: tmp_path / "lake")
    source = api.LiveSource(CFG)
    assert source._alerts() == []
    assert source._con is None  # it reconnects once the first alert exists


def test_files_signature_counts_parquet_files_and_notices_changes(tmp_path):
    events, orders = tmp_path / "alerts", tmp_path / "work_orders"
    assert api.files_signature(events, orders) == (0, 0)
    (events / "date=2018-11-21").mkdir(parents=True)
    first = events / "date=2018-11-21" / "part-0.parquet"
    first.write_bytes(b"x")
    (events / "_commit.json").write_text("{}")  # not a data file
    one = api.files_signature(events, orders)
    assert one[0] == 1 and one[1] > 0
    orders.mkdir()
    (orders / "part-0.parquet").write_bytes(b"y")
    two = api.files_signature(events, orders)
    assert two[0] == 2 and two != one


def test_alerts_are_queried_again_only_when_the_alert_files_change(monkeypatch, tmp_path):
    from leakwatch.common import lake
    from leakwatch.demo import runner

    root = tmp_path / "lake"
    monkeypatch.setattr(runner, "demo_lake", lambda: root)
    folder = lake.layer_dirs(root)["alert_events"]
    folder.mkdir(parents=True)
    (folder / "part-0.parquet").write_bytes(b"x")
    queries = []

    def fake_query(con, sql):
        queries.append(sql)
        return pd.DataFrame([{"alert_id": "a", "status": "open", "end_ts": pd.NaT}])

    source = api.LiveSource(CFG)
    monkeypatch.setattr(source, "_connection", lambda: object())
    monkeypatch.setattr(lake, "query_df", fake_query)
    assert source._alerts() == [{"alert_id": "a", "status": "open", "end_ts": None}]
    assert source._alerts() == source._alerts() and len(queries) == 1  # unchanged files: no query
    (folder / "part-1.parquet").write_bytes(b"x")
    source._alerts()
    assert len(queries) == 2
    for path in folder.glob("*.parquet"):
        path.unlink()
    assert source._alerts() == [] and len(queries) == 2  # a wiped lake is not queried
