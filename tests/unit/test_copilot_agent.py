from __future__ import annotations

import pytest

from leakwatch.copilot import guardrails, testing
from leakwatch.copilot.agent import CopilotAgent, render_tool_fallback
from leakwatch.copilot.llm import LLMError, LLMResponse, ScriptedLLM, calls, reply, tool_call
from leakwatch.copilot.settings import NOT_CONFIGURED_MESSAGE
from leakwatch.copilot.tools import ToolRegistry


@pytest.fixture
def ctx(tmp_path):
    return testing.tiny_context(testing.build_tiny_lake(tmp_path / "lake"))


def test_answer_with_supported_numbers_passes(ctx):
    llm = ScriptedLLM([
        calls(tool_call("list_alerts", area="A")),
        reply("Area A has 2 alerts; the strongest scored 9.25 against 3.0."),
    ])  # fmt: skip
    ans = CopilotAgent(llm, ctx, "v3").ask("How many alerts in area A?")
    assert ans.answer.startswith("Area A has 2 alerts")
    assert [c.name for c in ans.tool_calls] == ["list_alerts"]
    assert not ans.guardrail.regenerated and ans.guardrail.unsupported_before == []
    assert ans.refused is False and ans.error is None and ans.model == "scripted"
    text, tool_calls, charts = ans
    assert text == ans.answer and tool_calls == ans.tool_calls and charts == []


def test_unsupported_number_is_regenerated_once(ctx):
    llm = ScriptedLLM([
        calls(tool_call("list_alerts", area="A")),
        reply("Area A has 7 alerts."),
        reply("Area A has 2 alerts."),
    ])  # fmt: skip
    ans = CopilotAgent(llm, ctx, "v3").ask("How many alerts in area A?")
    assert ans.answer == "Area A has 2 alerts."
    assert ans.guardrail.regenerated and ans.guardrail.unsupported_before == ["7"]
    assert ans.guardrail.unsupported_after == [] and ans.guardrail.fallback is None
    correction = llm.requests[-1]["messages"][-1]
    assert correction["role"] == "user" and "7" in correction["content"]


def test_still_unsupported_falls_back_to_tool_output(ctx):
    llm = ScriptedLLM([
        calls(tool_call("estimate_loss", alert_id=testing.ALERT_A1)),
        reply("We lost about 150 m3."),
        reply("We lost about 160 m3."),
    ])  # fmt: skip
    ans = CopilotAgent(llm, ctx, "v3").ask("How much water did alert A1 lose?")
    assert ans.guardrail.fallback == "tool_output"
    assert "est_volume_m3: 144.0" in ans.answer
    assert ans.guardrail.unsupported_after == []


def test_v1_measures_but_does_not_enforce(ctx):
    llm = ScriptedLLM([calls(tool_call("list_alerts")), reply("There are 11 alerts.")])
    ans = CopilotAgent(llm, ctx, "v1").ask("How many alerts?")
    assert ans.answer == "There are 11 alerts."
    assert ans.guardrail.enforced is False
    assert ans.guardrail.unsupported_before == ["11"] == ans.guardrail.unsupported_after
    assert llm.remaining == 0


def test_out_of_scope_is_refused_without_a_model_call(ctx):
    llm = ScriptedLLM([])
    ans = CopilotAgent(llm, ctx, "v3").ask("Write a poem about the monsoon")
    assert ans.answer == guardrails.REFUSAL_TEXT and ans.refused and ans.guardrail.scope_prefilter
    assert llm.requests == []


def test_a_context_line_does_not_let_an_off_topic_question_through(ctx):
    """The dashboard sends a line about the alert on screen ("alert", "Area" are network words);
    the pre-filter reads the operator's question alone."""
    line = (
        "Context: the alert on screen is 5bab5c3a in Area A, which fired at 2018-11-19 20:15 UTC."
    )
    llm = ScriptedLLM([])
    ans = CopilotAgent(llm, ctx, "v3").ask("Write a poem about the monsoon", context=line)
    assert ans.refused and ans.guardrail.scope_prefilter and llm.requests == []
    llm = ScriptedLLM([reply("It fired at 2018-11-19 20:15 UTC.")])
    ans = CopilotAgent(llm, ctx, "v3").ask("When did it fire?", context=line)
    assert llm.requests[0]["messages"][-1]["content"] == f"{line}\n\nWhen did it fire?"
    assert ans.guardrail.unsupported_after == []  # the time is in the line that was sent


def test_v2_passes_off_topic_questions_to_the_model(ctx):
    llm = ScriptedLLM([reply("Sorry, I can only help with LeakWatch questions.")])
    ans = CopilotAgent(llm, ctx, "v2").ask("Write a poem about the monsoon")
    assert ans.refused and len(llm.requests) == 1


def test_missing_citation_is_corrected_or_answered_not_covered(ctx):
    llm = ScriptedLLM([
        calls(tool_call("search_manual", query="confirm a leak")),
        reply("Listen at valves first."),
        reply("Listen at valves first [LeakWatch SOP, 6. Confirm the leak before digging]."),
    ])  # fmt: skip
    ans = CopilotAgent(llm, ctx, "v3").ask("How do we confirm a leak?")
    assert ans.guardrail.citation_ok_before is False and ans.guardrail.citation_ok_after is True
    assert ans.answer.endswith("digging].")

    llm = ScriptedLLM([
        calls(tool_call("search_manual", query="confirm a leak")),
        reply("Listen at valves first."),
        reply("Just listen."),
    ])  # fmt: skip
    ans = CopilotAgent(llm, ctx, "v3").ask("How do we confirm a leak?")
    assert ans.guardrail.fallback == "not_covered"
    assert ans.answer.startswith(guardrails.NOT_COVERED_TEXT)


def test_charts_are_collected(ctx):
    llm = ScriptedLLM([
        calls(
            tool_call("get_sensor_series", sensor_id="n1", start="2019-03-10", end="2019-03-11")
        ),
        reply("The chart shows n1 on that day."),
    ])  # fmt: skip
    ans = CopilotAgent(llm, ctx, "v3").ask("Show pressure at n1 on 2019-03-10")
    assert len(ans.charts) == 1 and ans.charts[0]["kind"] == "line"


def test_no_model_configured(ctx):
    ans = CopilotAgent(None, ctx, "v3").ask("How many alerts?")
    assert ans.answer == NOT_CONFIGURED_MESSAGE and ans.error == "llm_not_configured"


def test_model_failure_is_reported_not_raised(ctx):
    def fail(messages, tools):
        raise LLMError("timeout")

    ans = CopilotAgent(ScriptedLLM([fail]), ctx, "v3").ask("How many alerts?")
    assert ans.error and "did not respond" in ans.answer


def test_system_prompts_differ_by_version(ctx):
    v1 = CopilotAgent(None, ctx, "v1").system_prompt()
    v3 = CopilotAgent(None, ctx, "v3").system_prompt()
    assert "2019-03-12 14:00 UTC" in v1 and "Example" not in v1 and "columns:" not in v1
    assert "Example 1" in v3 and "columns: alert_id" in v3 and guardrails.REFUSAL_TEXT in v3


def test_history_is_trimmed_and_cleaned(ctx):
    ctx.full_cfg["copilot"]["agent"]["history_messages"] = 2
    llm = ScriptedLLM([reply("Hello.")])
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "x"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    CopilotAgent(llm, ctx, "v1").ask("q3", history)
    sent = llm.requests[0]["messages"]
    assert [m["content"] for m in sent[1:]] == ["q2", "a2", "q3"]


def test_tool_loop_round_limit_forces_a_final_answer(ctx):
    ctx.full_cfg["copilot"]["agent"]["max_tool_rounds"] = 1
    llm = ScriptedLLM([calls(tool_call("list_alerts")), LLMResponse(content="Done.")])
    ans = CopilotAgent(llm, ctx, "v1").ask("List alerts")
    assert ans.answer == "Done." and llm.requests[-1]["tools"] is None


def test_fallback_renders_tables(ctx):
    reg = ToolRegistry(ctx)
    results = [reg.call("list_alerts", {"area": "A"}), reg.call("describe_schema", {})]
    text = render_tool_fallback(results)
    assert "| alert_id |" in text and testing.ALERT_A1 in text and "describe_schema" not in text
    sources = [r.content() for r in results]
    assert guardrails.check_numbers(text, sources).ok
