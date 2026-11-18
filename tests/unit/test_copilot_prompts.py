from __future__ import annotations

import pytest

from leakwatch.copilot.prompts import VERSIONS, few_shot_sql, fill, load_named_prompt, load_prompt
from leakwatch.copilot.tools import VIEW_ORDER


def test_versions_differ_in_technique():
    v1, v2, v3 = (load_prompt(v) for v in VERSIONS)
    assert not v1.examples and not v1.include_schema and not v1.guardrails
    assert v2.examples and not v2.include_schema and not v2.guardrails and not v2.sql_hints
    assert v3.include_schema and v3.guardrails and v3.sql_hints and v3.max_sql_retries > 0
    assert len(v3.examples) > len(v2.examples)  # self-correction examples are v3 only
    with pytest.raises(ValueError):
        load_prompt("v9")


def test_render_fills_placeholders_and_keeps_json_braces():
    spec = load_prompt("v3")
    text = spec.render_system(
        {
            "data_clock": "2019-12-31 23:00 UTC",
            "schema": "- alerts: ...",
            "refusal": "R.",
            "not_covered": "N.",
            "max_sql_retries": 2,
        }
    )
    assert "{data_clock}" not in text and "{schema}" not in text and "2019-12-31 23:00 UTC" in text
    assert '{"columns": ["area", "n_alerts"]' in text  # example tool results keep their braces
    assert fill("a {x} {y}", {"x": 1}) == "a 1 {y}"


def test_task_prompts_load():
    wo = load_named_prompt("work_order")
    assert "{evidence_json}" in wo["user"] and wo["template_actions"]
    assert "{text}" in load_named_prompt("complaint")["user"]


def test_few_shot_sql_passes_the_guard():
    pytest.importorskip("sqlglot")
    from leakwatch.copilot.sql_guard import guard_sql

    queries = few_shot_sql()
    assert len(queries) >= 4
    for query in queries:
        guard_sql(query, VIEW_ORDER)
