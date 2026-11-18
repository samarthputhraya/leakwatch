from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import pytest

from leakwatch.copilot import evaluate, testing
from leakwatch.copilot.agent import CopilotAgent
from leakwatch.copilot.llm import ScriptedLLM, calls, reply, tool_call
from leakwatch.copilot.tools import VIEW_ORDER


def test_question_set_is_complete_and_valid():
    items = evaluate.load_items()
    counts = Counter(i["category"] for i in items)
    assert len(items) >= 60
    assert counts["data"] >= 25 and counts["procedure"] >= 10
    assert counts["out_of_scope"] >= 8 and counts["complaint"] >= 8
    assert all(i["synthetic"] for i in items if i["category"] == "complaint")
    assert all(
        i.get("sql")
        for i in items
        if i["category"] == "data" and i["check"] != "chart" and i["id"] != "alerts-14"
    )


def test_reference_sql_passes_the_guard():
    pytest.importorskip("sqlglot")
    from leakwatch.copilot.sql_guard import guard_sql

    for item in evaluate.load_items():
        for key in ("reference_sql", "reported_at_sql"):
            if item.get(key):
                guard_sql(item[key], VIEW_ORDER)


def test_validate_item_rejects_incomplete_items():
    with pytest.raises(ValueError):
        evaluate.validate_item(
            {
                "id": "x",
                "category": "data",
                "question": "q",
                "expected_tools_any": [],
                "check": "number",
            }
        )
    with pytest.raises(ValueError):
        evaluate.validate_item(
            {
                "id": "x",
                "category": "complaint",
                "question": "q",
                "expected_tools_any": [],
                "text": "t",
                "reported_at": "2019-01-01",
            }
        )


def test_result_matching_and_answer_checks():
    ref = pd.DataFrame({"area": ["A"], "n": [7]})
    assert evaluate.result_matches(["n", "area"], [[7, "A"]], ref)
    assert evaluate.result_matches(["area", "n", "extra"], [["A", 7, 1]], ref)
    assert not evaluate.result_matches(["area", "n"], [["A", 7], ["C", 1]], ref)
    assert not evaluate.result_matches(["area", "n"], [["A", 8]], ref)
    assert evaluate._number_in("There were 7 alerts.", 7, 0.01)
    assert evaluate._number_in("There were no alerts.", 0, 0.01)
    assert evaluate._number_in("About 45% of them.", 0.45, 0.01)
    assert not evaluate._number_in("There were 8 alerts.", 7, 0.01)
    assert evaluate._date_in(
        "It started on 2019-03-10 06:00 UTC.", pd.Timestamp("2019-03-10 06:00", tz="UTC")
    )
    assert evaluate._token_in("the top sensor is n54.", "n54") and not evaluate._token_in(
        "n540", "n54"
    )


@pytest.fixture
def ctx(tmp_path):
    return testing.tiny_context(testing.build_tiny_lake(tmp_path / "lake"))


def item(id_: str, category: str, question: str, tools: list[str], **fields) -> dict:
    return {
        "id": id_,
        "category": category,
        "question": question,
        "expected_tools_any": tools,
        **fields,
    }


ITEMS = [
    item(
        "d1", "data", "How many alerts in area A?", ["run_sql"],
        reference_sql="SELECT count(*) AS n FROM alerts WHERE area = 'A'",
        check="number", sql=True, requires=["alerts"],
    ),
    item(
        "d2", "data", "Which area had most alerts?", ["run_sql"],
        reference_sql="SELECT area, count(*) n FROM alerts GROUP BY area ORDER BY n DESC LIMIT 1",
        check="area", column="area", sql=True,
    ),
    item(
        "d3", "data", "Mean score?", ["run_sql"],
        reference_sql="SELECT avg(score) FROM scores", check="number", requires=["scores"],
    ),
    item(
        "c1", "data", "Show n1 on 2019-03-10", ["get_sensor_series"],
        check="chart", chart_sensor="n1",
        mock_call={
            "tool": "get_sensor_series",
            "args": {"sensor_id": "n1", "start": "2019-03-10", "end": "2019-03-10 23:55"},
        },
    ),
    item(
        "p1", "procedure", "How do we confirm a leak before digging?", ["search_manual"],
        expected_sources=[{"doc": "LeakWatch SOP", "section": "Confirm the leak before digging"}],
    ),
    item("o1", "out_of_scope", "Write a poem about rain.", []),
    item(
        "k1", "complaint", "-", ["log_complaint"], synthetic=True,
        text="No water in area A since this morning.",
        reported_at_sql=(
            "SELECT start_ts + INTERVAL 3 HOUR FROM alerts WHERE area = 'A' "
            "ORDER BY start_ts DESC LIMIT 1"
        ),
        expected={"area": "A", "issue_type": "no_water"},
    ),
    item(
        "k2", "complaint", "-", ["log_complaint"], synthetic=True,
        text="Our taps are dry.", reported_at="2019-03-10 09:00",
        expected={"area": None, "issue_type": "no_water"},
    ),
]  # fmt: skip


def test_mock_oracle_run_scores_every_metric(ctx):
    summary, outcomes = evaluate.run_eval("v3", ITEMS, ctx, evaluate.mock_llm_for)
    by_id = {o.id: o for o in outcomes}
    assert by_id["d3"].skipped == "lake has no scores"
    assert all(by_id[i].correct for i in ("d1", "d2", "c1", "p1", "o1", "k1", "k2")), [
        (o.id, o.answer) for o in outcomes if not o.correct
    ]
    assert by_id["o1"].scope_prefilter and by_id["o1"].tools == []
    assert by_id["k1"].complaint_match_ok and by_id["k2"].complaint_match_ok
    assert summary["items"]["evaluated"] == 7 and summary["items"]["skipped"] == 1
    assert summary["sql_execution_accuracy"] == {"value": 1.0, "n": 2}
    assert summary["refusal_accuracy"]["value"] == 1.0
    assert summary["hallucinated_number_rate"]["after_guardrail"]["value"] == 0.0
    assert summary["latency_s"]["p50"] is not None


def test_wrong_answers_are_scored_wrong(ctx):
    prep = evaluate.prepare(ITEMS[0], ctx, None)
    llm = ScriptedLLM(
        [
            calls(tool_call("run_sql", query="SELECT count(*) FROM alerts")),
            reply("There are 3 alerts."),
        ]
    )
    out = evaluate.score(prep, CopilotAgent(llm, ctx, "v1").ask(prep.question))
    assert out.correct is False and out.sql_exec_match is False and out.tool_ok is True
    assert out.hallucinated_before is False


def test_main_without_a_model_writes_nothing(monkeypatch):
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("metrics must not be written")

    monkeypatch.setattr(evaluate, "write_metrics", forbidden)
    assert evaluate.main(["--version", "v3"]) == 2


def test_main_mock_never_writes_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    testing.build_tiny_lake(tmp_path / "lake")
    items = tmp_path / "items.jsonl"
    items.write_text(
        "\n".join(json.dumps(i) for i in ITEMS[:2] + ITEMS[5:6]) + "\n", encoding="utf-8"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("metrics must not be written")

    monkeypatch.setattr(evaluate, "write_metrics", forbidden)
    assert evaluate.main(["--version", "all", "--mock", "--items", str(items)]) == 0
