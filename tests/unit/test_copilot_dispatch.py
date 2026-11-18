from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from leakwatch.common import lake
from leakwatch.common.config import load_config
from leakwatch.common.schemas import WORK_ORDER_COLUMNS, WorkOrder
from leakwatch.copilot import dispatch, testing
from leakwatch.copilot.guardrails import check_numbers
from leakwatch.copilot.llm import ScriptedLLM, reply
from leakwatch.copilot.network_index import plan_pipes

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
POLICY = load_config(environ={})["copilot"]["dispatch"]["priority"]


@pytest.fixture
def ctx(tmp_path):
    return testing.tiny_context(testing.build_tiny_lake(tmp_path / "lake"))


@pytest.mark.parametrize(
    ("ratio", "loss", "priority"),
    [(3.1, 1.0, "P1"), (1.0, 12.0, "P1"), (1.6, 0.5, "P2"), (1.0, 3.0, "P2"), (1.2, None, "P3")],
)
def test_priority_policy(ratio, loss, priority):
    got, reason = dispatch.priority_for({"score_to_threshold": ratio}, loss, POLICY)
    assert got == priority and reason


def test_pipe_plans():
    net = testing.toy_network()
    gnn = {"gnn_top5": [{"node_id": "n6", "prob": 0.3}, {"node_id": "n7", "prob": 0.6}]}
    pipes, basis = plan_pipes(gnn, net, 10, 4)
    assert pipes == ["p6", "p7"] and "localizer" in basis
    fallback = {"top_sensors": [{"sensor_id": "pressure__n2"}, {"sensor_id": "amr_sum__C"}]}
    pipes, basis = plan_pipes(fallback, net, 3, 4)
    assert pipes == ["p2", "p3", "p5"] and "nearest the top sensors" in basis
    assert plan_pipes(fallback, None, 3, 4)[0] == []
    # the valve has length 0, but the search stays in the sensor's area
    assert "p6" not in [p for p, _ in net.nearest_pipes("n4", 6)]


def test_template_work_order_is_valid_labelled_and_grounded(ctx):
    gen = dispatch.generate_work_order(ctx, testing.ALERT_A1, now=NOW)
    order = gen.order
    assert gen.generator == "template" and order.summary.startswith(dispatch.TEMPLATE_LABEL)
    assert order.priority == "P1" and order.area == "A" and order.estimated_loss_m3_per_hour == 12.0
    assert order.pipes_to_inspect == ["p2", "p3", "p5", "p4", "p1"]
    assert order.manual_citations and order.manual_citations[0].doc
    WorkOrder.model_validate_json(order.model_dump_json())
    text = "\n".join([order.summary, *order.evidence])
    assert check_numbers(text, [gen.evidence.as_json()]).ok


def test_gnn_alert_uses_candidate_pipes(ctx):
    order = dispatch.generate_work_order(ctx, testing.ALERT_C2, now=NOW).order
    assert order.pipes_to_inspect == ["p6", "p7"]
    assert any("get_gnn_candidates: top candidate nodes n7 (0.6)" in e for e in order.evidence)


def llm_json(**overrides) -> str:
    doc = {
        "priority": "P3",
        "summary": (
            "Likely leak in area A: score 9.25 against a threshold of 3.0, an estimated 12.0 m3/h."
        ),
        "pipes_to_inspect": ["p99", "p2"],
        "evidence": [
            "get_alert: score 9.25 vs threshold 3.0",
            "estimate_loss: 12.0 m3/h (estimate)",
        ],
        "recommended_actions": ["Walk the pipes in order."],
        "manual_citations": [
            {"doc": "LeakWatch SOP", "chapter": "6. Confirm the leak before digging", "page": "n/a"}
        ],
    }
    doc.update(overrides)
    return json.dumps(doc)


def test_llm_work_order_is_grounded_in_the_evidence(ctx):
    llm = ScriptedLLM([reply(llm_json())])
    gen = dispatch.generate_work_order(ctx, testing.ALERT_A1, llm=llm, now=NOW)
    order = gen.order
    assert gen.generator == "llm"
    assert order.priority == "P1"  # policy, not the model's P3
    assert order.pipes_to_inspect == ["p2", "p3", "p5", "p4", "p1"]  # evidence order, p99 dropped
    assert order.estimated_loss_m3_per_hour == 12.0 and str(order.alert_id) == testing.ALERT_A1
    assert [c.chapter for c in order.manual_citations] == ["6. Confirm the leak before digging"]
    assert llm.requests[0]["json_mode"] is True


def test_llm_number_problems_get_one_correction_then_the_template(ctx):
    fixed = ScriptedLLM(
        [reply(llm_json(summary="Leak losing 77.7 m3/h in area A.")), reply(llm_json())]
    )
    gen = dispatch.generate_work_order(ctx, testing.ALERT_A1, llm=fixed, now=NOW)
    assert gen.generator == "llm" and "77.7" in fixed.requests[1]["messages"][-1]["content"]
    broken = ScriptedLLM(
        [reply(llm_json(summary="Losing 77.7 m3/h.")), reply(llm_json(summary="Losing 66.6 m3/h."))]
    )
    gen = dispatch.generate_work_order(ctx, testing.ALERT_A1, llm=broken, now=NOW)
    assert gen.generator == "template" and "fallback" in gen.notes[0]


def test_invalid_model_output_uses_template_or_raises_without_safe_mode(ctx):
    gen = dispatch.generate_work_order(
        ctx, testing.ALERT_A1, llm=ScriptedLLM([reply("not json")]), now=NOW
    )
    assert gen.generator == "template"
    ctx.full_cfg["copilot"]["dispatch"]["demo_safe_mode"] = False
    with pytest.raises(dispatch.WorkOrderError):
        dispatch.generate_work_order(ctx, testing.ALERT_A1, llm=ScriptedLLM([reply("{}")]), now=NOW)


def test_written_orders_appear_in_the_alerts_view(ctx):
    assert dispatch.alerts_needing_orders(ctx) == [
        testing.ALERT_A3,
        testing.ALERT_A1,
        testing.ALERT_C2,
    ]
    gen = dispatch.generate_work_order(ctx, testing.ALERT_A1, now=NOW)
    path = dispatch.write_work_order(gen, ctx.lake_root)
    assert path.name.startswith(f"wo_{testing.ALERT_A1}_") and path.suffix == ".parquet"
    con = lake.connect(lake_root=ctx.lake_root)
    wo = con.execute("SELECT * FROM work_orders").df()
    assert list(wo.columns) == list(WORK_ORDER_COLUMNS)
    row = con.execute(
        "SELECT work_order_json, work_order_generator FROM alerts WHERE alert_id = ?",
        [testing.ALERT_A1],
    ).fetchone()
    assert row[1] == "template" and json.loads(row[0])["priority"] == "P1"
    ctx.refresh()
    assert testing.ALERT_A1 not in dispatch.alerts_needing_orders(ctx)
    # a newer alert version makes the order stale
    newer = [
        r | {"version": 3, "updated_at": pd.Timestamp("2027-01-01", tz="UTC")}
        for r in testing.alert_rows()
        if r["alert_id"] == testing.ALERT_A1 and r["version"] == 2
    ]
    testing.write_alerts(ctx.lake_root, newer, name="part-1")
    ctx.refresh()
    assert testing.ALERT_A1 in dispatch.alerts_needing_orders(ctx)


def test_run_dispatch_counts_and_dry_run(ctx):
    dry = dispatch.run_dispatch(ctx, None, dry_run=True)
    assert dry == {"alerts": 3, "llm": 0, "template": 3, "fallbacks": 0, "written": 0, "failed": 0}
    assert not (ctx.lake_root / "work_orders").exists()
    done = dispatch.run_dispatch(ctx, None, origin="batch", limit=2)
    assert done["written"] == 2
    assert len(dispatch.alerts_needing_orders(ctx)) == 1


def test_cli_without_alerts_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert dispatch.main(["--dry-run"]) == 0
    assert not (tmp_path / "lake" / "work_orders").exists()


def test_work_order_schema_rejects_bad_priority():
    with pytest.raises(ValidationError):
        dispatch.WorkOrderDraft.model_validate(
            {"priority": "P9", "summary": "x" * 20, "evidence": ["a"], "recommended_actions": ["b"]}
        )
