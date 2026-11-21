"""Control-room screen, server side (D-052): routes, snapshots, lake failure paths."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

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

    def unreadable() -> dict:
        raise OSError("No files found")

    monkeypatch.setattr(source, "_read_lake", unreadable)
    out = source()
    assert out["error"] and out["banner"] == {
        "kind": "unknown",
        "title": "Alerts cannot be read",
        "sub": "The demo lake could not be read. It is retried every second.",
    }

    monkeypatch.setattr(source, "_read_lake", lambda: {})
    out = source()
    assert out["error"] is None and out["banner"]["title"] == "No open alerts"
    assert out["incident"] is None and out["drops"] is None
    assert out["build_ms"] >= 0


def test_live_source_reads_nothing_from_an_empty_lake(monkeypatch, tmp_path):
    from leakwatch.demo import runner

    monkeypatch.setattr(runner, "demo_lake", lambda: tmp_path / "lake")
    source = api.LiveSource(CFG)
    assert source._read_lake() == {
        "alerts": [],
        "versions": [],
        "orders": [],
        "residuals": [],
        "scores": [],
    }
    assert source._con is None  # it connects once the first files exist


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


def test_each_lake_read_runs_again_only_when_its_files_change(monkeypatch, tmp_path):
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
    rows = source._read_lake()
    assert rows["alerts"] == [{"alert_id": "a", "status": "open", "end_ts": None}]
    assert rows["residuals"] == [] and rows["scores"] == []  # no files: not queried
    assert len(queries) == 3  # alerts, their versions, their orders
    assert source._read_lake() == rows and len(queries) == 3  # unchanged files: no query
    (folder / "part-1.parquet").write_bytes(b"x")
    source._read_lake()
    assert len(queries) == 6
    for path in folder.glob("*.parquet"):
        path.unlink()
    assert source._read_lake()["alerts"] == [] and len(queries) == 6  # a wiped lake is not queried


def test_residuals_are_read_per_sensor_for_the_chart_hours(monkeypatch, tmp_path):
    """Real DuckDB over a small features_stream layer: sensor ids, time order, the time window."""
    from leakwatch.common import lake
    from leakwatch.demo import runner

    root = tmp_path / "lake"
    monkeypatch.setattr(runner, "demo_lake", lambda: root)
    folder = lake.layer_dirs(root)["features_stream"] / "source=battledim" / "window=1h"
    folder = folder / "date=2018-11-21"
    folder.mkdir(parents=True)
    ends = pd.date_range("2018-11-20 00:00", periods=30 * 4, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "window_end": ends,
            "pressure__n469__residual": -np.arange(len(ends)) / 100,
            "pressure__n469__mean": 50.0,
            "pressure__n114__residual": 0.1,
        }
    ).to_parquet(folder / "part-0.parquet")
    rows = api.LiveSource(CFG)._read_lake()["residuals"]
    hours = CFG["dashboard"]["chart_hours"]
    assert len(rows) == hours * 4  # 15-minute steps, strictly after the newest minus the hours
    assert set(rows[0]) == {"window_end", "n469", "n114"}
    assert rows[-1]["window_end"] == ends[-1].to_pydatetime()
    assert rows[0]["window_end"] > ends[-1].to_pydatetime() - pd.Timedelta(hours=hours)
    assert rows[-1]["n469"] == pytest.approx(-(len(ends) - 1) / 100)


def test_a_schema_read_without_residual_columns_is_read_again():
    """A DESCRIBE that raced a file replacement must not hide the rings for the whole run."""
    source = api.LiveSource(CFG)
    schemas = [[("window_end",)], [("window_end",), ("pressure__n469__residual",)]]

    class Result:
        def __init__(self, rows: list) -> None:
            self.rows = rows

        def fetchall(self) -> list:
            return self.rows

    class Con:
        def execute(self, sql: str) -> Result:
            return Result(schemas.pop(0))

    con = Con()
    assert source._residual_columns(con) == []
    assert source._residual_columns(con) == [("pressure__n469__residual", "n469")]
    assert source._residual_columns(con) == [("pressure__n469__residual", "n469")]  # now kept


def test_snapshot_builds_never_overlap():
    """The first page request and the refresh thread must not share the connection at once."""
    import threading
    import time

    inside, most = [0], [0]

    def source() -> dict:
        inside[0] += 1
        most[0] = max(most[0], inside[0])
        time.sleep(0.05)
        inside[0] -= 1
        return VM

    snaps = api.Snapshots(source, 60)
    threads = [threading.Thread(target=snaps.refresh) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert most[0] == 1 and snaps.latest() == VM


INCIDENT_VM = {
    **VM,
    "incident": {"alert_id": "5bab5c3a", "area": "A", "start_ts": "2018-11-19T20:15:00+00:00"},
}


class FakeCopilot:
    def __init__(self, configured=True, fail=False):
        self.is_configured, self.fail, self.sent = configured, fail, []

    def configured(self):
        return self.is_configured

    def loss(self, alert_id):
        self.sent.append(("loss", alert_id))
        return {"available": True, "est_volume_m3": 120.0, "est_loss_m3h": 40.0, "duration_h": 3.0}

    def ask(self, question):
        if self.fail:
            raise RuntimeError("connection refused by api.example")
        self.sent.append(("ask", question))
        return SimpleNamespace(
            answer="Fine.",
            tool_calls=[],
            charts=[],
            refused=False,
            latency_s=1.0,
            guardrail=None,
            error=None,
            model="m",
        )


def copilot_client(copilot, vm=INCIDENT_VM) -> TestClient:
    return TestClient(
        api.create_app(lambda: vm, background=False, cfg=CFG, copilot=copilot),
        base_url="http://localhost",
        headers={"X-Requested-With": "leakwatch"},
    )


def test_the_loss_chip_needs_no_model_and_names_the_alert_on_screen():
    fake = FakeCopilot(configured=False)
    res = copilot_client(fake).post(
        "/api/copilot/ask", json={"alert_id": "5bab5c3a", "direct": True}
    )
    assert res.status_code == 200 and res.json()["text"].startswith("About 120 m³")
    assert fake.sent == [("loss", "5bab5c3a")]
    gone = copilot_client(fake).post("/api/copilot/ask", json={"alert_id": "older", "direct": True})
    assert gone.status_code == 409 and "no longer on screen" in gone.json()["message"]
    none = copilot_client(fake, VM).post("/api/copilot/ask", json={"direct": True})
    assert none.status_code == 409


def test_questions_go_to_the_model_with_the_visible_context_line():
    fake = FakeCopilot()
    res = copilot_client(fake).post(
        "/api/copilot/ask", json={"alert_id": "5bab5c3a", "question": " Which pipe first? "}
    )
    assert res.json()["state"] == "answer" and res.json()["question"] == "Which pipe first?"
    assert fake.sent == [
        (
            "ask",
            "Context: the alert on screen is 5bab5c3a in Area A, "
            "which fired at 2018-11-19 20:15 UTC.\n\nWhich pipe first?",
        )
    ]
    plain = FakeCopilot()
    copilot_client(plain).post("/api/copilot/ask", json={"question": "What is L-Town?"})
    assert plain.sent == [("ask", "What is L-Town?")]  # no alert named: no context line


def test_copilot_states_without_a_model_with_a_failure_and_without_a_question():
    res = copilot_client(FakeCopilot(configured=False)).post(
        "/api/copilot/ask", json={"question": "Hi"}
    )
    assert res.json()["state"] == "no_model" and "LLM_MODEL" in res.json()["text"]
    res = copilot_client(FakeCopilot(fail=True)).post("/api/copilot/ask", json={"question": "Hi"})
    assert res.json()["state"] == "error" and "api.example" not in res.text
    empty = copilot_client(FakeCopilot()).post("/api/copilot/ask", json={"question": "  "})
    assert empty.status_code == 422
    long = copilot_client(FakeCopilot()).post("/api/copilot/ask", json={"question": "x" * 501})
    assert long.status_code == 422
    other_site = TestClient(
        api.create_app(lambda: VM, background=False, cfg=CFG, copilot=FakeCopilot()),
        base_url="http://localhost",
    )  # no page header
    assert other_site.post("/api/copilot/ask", json={"question": "Hi"}).status_code == 403
