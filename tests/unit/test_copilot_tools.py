from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest

from leakwatch.copilot import testing
from leakwatch.copilot.tools import TOOLS, Tool, ToolRegistry, fmt_ts, jsonable, parse_ts


@pytest.fixture
def ctx(tmp_path):
    return testing.tiny_context(testing.build_tiny_lake(tmp_path / "lake"))


@pytest.fixture
def reg(ctx):
    return ToolRegistry(ctx)


def call(reg, name, **args):
    result = reg.call(name, args)
    json.dumps(result.output)  # every output is JSON-ready
    return result


def test_views_are_the_lake_layers_only(ctx):
    assert ctx.views() == {
        "alert_events",
        "alerts",
        "work_orders",
        "clean_readings",
        "baselines",
        "features",
    }


def test_data_clock_is_the_latest_window_or_alert(ctx):
    assert ctx.data_clock() == testing.ts("2019-03-12 14:00")
    ctx.clock = parse_ts("2019-01-01 00:00")
    assert ctx.data_clock() == testing.ts("2019-01-01 00:00")


def test_describe_schema_summarises_feature_columns(reg):
    out = call(reg, "describe_schema").output
    tables = {t["name"]: t for t in out["tables"]}
    assert "alerts" in tables and "features" in tables
    assert any(c.startswith("top_sensors STRUCT") for c in tables["alerts"]["columns"])
    assert tables["features"]["feature_columns"]["series"] == ["pressure__n1"]
    assert not any("__mean" in c for c in tables["features"]["columns"])
    assert "2019-03-12 14:00 UTC" in " ".join(out["notes"])


def test_list_alerts_filters_and_orders(reg):
    out = call(reg, "list_alerts").output
    assert out["total"] == 3
    assert out["alerts"][0]["alert_id"] == testing.ALERT_C2  # newest first
    area_a = call(reg, "list_alerts", area="a", order="score").output
    assert area_a["total"] == 2 and area_a["alerts"][0]["kmeans_score_max"] == 9.25
    assert call(reg, "list_alerts", status="open").output["total"] == 1
    window = call(reg, "list_alerts", start="2019-01-01", end="2019-03-11").output
    assert [a["alert_id"] for a in window["alerts"]] == [testing.ALERT_A1]
    assert "error" in call(reg, "list_alerts", area="D").output
    assert "error" in call(reg, "list_alerts", start="not a date").output


def test_get_alert_returns_latest_version_by_prefix(reg):
    out = call(reg, "get_alert", alert_id=testing.ALERT_A1[:8]).output
    assert out["version"] == 2 and out["status"] == "closed"
    assert out["duration_h"] == 12.0
    assert out["score_to_threshold"] == pytest.approx(3.08, abs=0.01)
    assert out["top_sensors"][0] == {
        "series_id": "pressure__n2",
        "sensor_id": "n2",
        "sensor_type": "pressure",
        "contribution": 0.5,
    }
    assert "error" in call(reg, "get_alert", alert_id="1111").output  # too short
    assert "error" in call(reg, "get_alert", alert_id="99999999").output


def test_run_sql_returns_rows_and_executed_sql(reg):
    pytest.importorskip("sqlglot")
    out = call(
        reg, "run_sql", query="SELECT area, count(*) AS n FROM alerts GROUP BY area ORDER BY n DESC"
    ).output
    assert out["rows"] == [["A", 2], ["C", 1]]
    assert out["sql"].endswith("LIMIT 1000")
    bad = call(reg, "run_sql", query="SELECT nope FROM alerts")
    assert "nope" in bad.output["error"] and "Columns: alerts(" in bad.output["hint"]
    rejected = call(reg, "run_sql", query="DROP TABLE alerts").output
    assert rejected["error"].startswith("rejected")


def test_run_sql_caps_rows_and_times_out(ctx):
    pytest.importorskip("sqlglot")
    ctx.full_cfg["copilot"]["sql"]["max_rows"] = 2
    reg = ToolRegistry(ctx)
    out = call(reg, "run_sql", query="SELECT * FROM range(10) t(i)").output
    assert out["row_count"] == 2 and out["truncated"]
    ctx.full_cfg["copilot"]["sql"]["timeout_s"] = 0.2
    slow = call(reg, "run_sql", query="SELECT sum(i * i) FROM range(100000000000) t(i)").output
    assert "stopped after" in slow["error"]


def test_connection_is_sandboxed_even_without_the_guard(ctx):
    with pytest.raises(duckdb.Error):
        ctx.query("SELECT * FROM read_csv('/etc/passwd')")
    with pytest.raises(duckdb.Error):
        ctx.query("SET enable_external_access = true")


def test_sensor_series_summary_chart_and_baseline(reg):
    res = call(
        reg, "get_sensor_series", sensor_id="n2", start="2019-03-10 00:00", end="2019-03-10 23:55"
    )
    out = res.output
    assert out["n_readings"] == 288 and out["unit"] == "m" and out["table"] == "clean_readings"
    assert out["min"] <= out["mean"] <= out["max"]
    assert out["baseline_mean"] is not None and out["mean_residual"] is not None
    assert len(out["sample"]) == 24
    assert res.chart is not None and "_chart" not in out
    assert [s["name"] for s in res.chart["series"]] == ["pressure n2", "hour-of-week baseline"]
    assert len(res.chart["series"][0]["x"]) == 288


def test_sensor_series_around_alert_and_downsampling(ctx):
    ctx.full_cfg["copilot"]["series"]["max_points"] = 100
    res = call(
        ToolRegistry(ctx),
        "get_sensor_series",
        sensor_id="pressure__n1",
        around_alert_id=testing.ALERT_A1,
    )
    assert res.output["chart_resolution_minutes"] > 5
    assert len(res.chart["series"][0]["x"]) <= 100
    assert [m["label"] for m in res.chart["markers"]] == ["alert fired", "alert closed"]


def test_sensor_series_errors(reg):
    assert "error" in call(reg, "get_sensor_series", sensor_id="n999", start="2019-03-10").output
    assert "error" in call(reg, "get_sensor_series", sensor_id="amr_sum__C").output
    assert "error" in call(reg, "get_sensor_series", sensor_id="n1", sensor_type="volume").output


def test_gnn_candidates_and_fallback_pipes(reg):
    none = call(reg, "get_gnn_candidates", alert_id=testing.ALERT_A1).output
    assert none["available"] is False
    assert none["fallback_pipes"][:3] == ["p2", "p3", "p5"]
    some = call(reg, "get_gnn_candidates", alert_id=testing.ALERT_C2).output
    assert [c["node_id"] for c in some["candidates"]] == ["n7", "n6"]
    assert some["candidates"][0]["pipes"] == ["p6", "p7"] and some["candidates"][0]["area"] == "C"


def test_estimate_loss_is_rate_times_duration(reg):
    out = call(reg, "estimate_loss", alert_id=testing.ALERT_A1).output
    assert out["is_estimate"] and out["est_volume_m3"] == 144.0 and out["duration_h"] == 12.0
    open_alert = call(reg, "estimate_loss", alert_id=testing.ALERT_C2).output
    assert open_alert["est_volume_m3"] == 8.0  # 2 m3/h x 4 h to the latest window
    assert call(reg, "estimate_loss", alert_id=testing.ALERT_A3).output["available"] is False


def test_search_manual_returns_citations(reg, ctx):
    out = call(
        reg, "search_manual", query="confirm the leak before digging, listening stick"
    ).output
    assert out["passages"][0]["citation"] == "[LeakWatch SOP, 6. Confirm the leak before digging]"
    ctx.retriever = None
    ctx._retriever_error = "no index"
    assert "make index" in call(reg, "search_manual", query="x").output["error"]


def test_log_complaint_matches_alert(reg, ctx):
    out = call(
        reg,
        "log_complaint",
        text="No water in area A since this morning",
        reported_at="2019-03-10 09:00",
    ).output
    assert out["match"]["status"] == "matched" and out["match"]["alert_id"] == testing.ALERT_A1
    assert out["extracted"]["issue_type"] == "no_water"
    assert len(ctx.complaints) == 1


def test_registry_reports_bad_calls(reg, monkeypatch):
    assert "unknown tool" in reg.call("drop_everything", {}).output["error"]
    assert "missing" in reg.call("get_alert", {}).output["error"]
    assert "unknown" in reg.call("get_alert", {"alert_id": "x" * 8, "foo": 1}).output["error"]
    assert "not valid JSON" in reg.call("get_alert", {"__invalid_json__": "{"}).output["error"]

    def boom(ctx, args):
        raise RuntimeError("kaput")

    params = {"type": "object", "properties": {}}
    monkeypatch.setitem(TOOLS, "describe_schema", Tool("describe_schema", "x", params, boom))
    result = ToolRegistry(reg.ctx).call("describe_schema", {})
    assert result.error == "RuntimeError: kaput"


def test_schemas_are_openai_function_tools(reg):
    schemas = reg.schemas()
    assert {s["function"]["name"] for s in schemas} == set(TOOLS)
    assert all(
        s["type"] == "function" and s["function"]["parameters"]["type"] == "object" for s in schemas
    )


def test_tool_content_is_truncated(reg):
    res = call(reg, "list_alerts")
    assert res.content(80).endswith("more characters]")


def test_formatting_helpers():
    assert fmt_ts(pd.Timestamp("2019-03-10 06:00", tz="UTC")) == "2019-03-10 06:00 UTC"
    assert fmt_ts(pd.Timestamp("2019-03-10 06:00:30", tz="UTC")) == "2019-03-10 06:00:30 UTC"
    assert jsonable({"a": float("nan"), "b": 1.23456789, "c": [pd.NaT]}) == {
        "a": None,
        "b": 1.23457,
        "c": [None],
    }
    assert parse_ts("2019-03-10 06:00 UTC") == testing.ts("2019-03-10 06:00")


def test_connection_is_kept_until_a_new_layer_appears(ctx, monkeypatch):
    """D-045: no periodic rebuild (it costs tens of seconds on the full lake)."""
    from leakwatch.copilot import tools as tools_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(tools_mod.time, "monotonic", lambda: clock["t"])
    first = ctx.con()
    tables = ctx.schema()["tables"]
    clock["t"] += 10 * float(ctx.cfg["agent"]["connection_ttl_s"])
    assert ctx.con() is first  # TTL passed, but no layer changed
    assert ctx.schema()["tables"] is tables  # schema cache kept
    assert "scores" not in ctx.views()
    scores_dir = ctx.lake_dir() / "scores" / "model=x" / "origin=batch" / "date=2019-03-10"
    scores_dir.mkdir(parents=True)
    pd.DataFrame({"window_start": [testing.ts("2019-03-10 00:00")], "score": [1.0]}).to_parquet(
        scores_dir / "part-0.parquet"
    )
    assert ctx.con() is first  # checked only once per TTL
    clock["t"] += 10 * float(ctx.cfg["agent"]["connection_ttl_s"])
    assert ctx.con() is not first and "scores" in ctx.views()


def test_data_clock_is_cached_for_its_ttl_only(ctx, monkeypatch):
    from leakwatch.copilot import tools as tools_mod

    clock = {"t": 50.0}
    monkeypatch.setattr(tools_mod.time, "monotonic", lambda: clock["t"])
    calls = []
    real_query = ctx.query

    def counting(sql, *a, **k):
        calls.append(sql)
        return real_query(sql, *a, **k)

    monkeypatch.setattr(ctx, "query", counting)
    first = ctx.data_clock()
    n = len(calls)
    assert ctx.data_clock() == first and len(calls) == n  # cached
    clock["t"] += float(ctx.cfg["agent"]["clock_ttl_s"]) + 1
    assert ctx.data_clock() == first and len(calls) > n  # recomputed after the TTL
    assert f"Data ends at {fmt_ts(first)}" in ctx.schema_text()
