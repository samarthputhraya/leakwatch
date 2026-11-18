from __future__ import annotations

from types import SimpleNamespace

import pytest

from leakwatch.copilot.llm import (
    LLMError,
    ScriptedLLM,
    build_llm,
    calls,
    json_object_from_text,
    last_tool_output,
    parse_arguments,
    reply,
    run_tool_loop,
    tool_call,
)
from leakwatch.copilot.settings import LLMSettings, llm_settings


def test_scripted_llm_returns_steps_and_records_requests():
    llm = ScriptedLLM([reply("a"), lambda messages, tools: reply(f"{len(messages)} messages")])
    assert llm.chat([{"role": "user", "content": "x"}]).content == "a"
    assert (
        llm.chat([{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]).content
        == "2 messages"
    )
    assert len(llm.requests) == 2 and llm.remaining == 0
    with pytest.raises(LLMError, match="exhausted"):
        llm.chat([])


def test_run_tool_loop_executes_calls_and_appends_messages():
    llm = ScriptedLLM([calls(tool_call("t", x=1), tool_call("t", x=2)), reply("done")])
    seen = []
    messages = [{"role": "user", "content": "go"}]
    out = run_tool_loop(
        llm, messages, [{"type": "function"}], lambda c: seen.append(c.arguments) or '{"ok": 1}', 5
    )
    assert out.content == "done" and out.rounds == 2 and not out.forced_final
    assert seen == [{"x": 1}, {"x": 2}]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    assert last_tool_output(messages) == {"ok": 1}


def test_argument_and_json_parsing():
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("") == {}
    assert "__invalid_json__" in parse_arguments("{oops")
    assert "__invalid_json__" in parse_arguments("[1, 2]")
    assert json_object_from_text('```json\n{"area": "A"}\n```') == {"area": "A"}
    assert json_object_from_text('Here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        json_object_from_text("no json here")
    with pytest.raises(ValueError):
        json_object_from_text("[1, 2]")


def test_settings_configured_and_safe_description():
    assert not llm_settings({}).configured
    assert not llm_settings({"LLM_MODEL": "m"}).configured
    local = llm_settings(
        {"LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "qwen", "LLM_API_KEY": " "}
    )
    assert local.configured and local.api_key is None
    hosted = LLMSettings(None, "sk-secret", "gpt")
    assert hosted.configured and "sk-secret" not in str(hosted.describe())
    assert build_llm(llm_settings({}), {}) is None


def test_openai_client_parses_tool_calls_and_wraps_errors():
    openai = pytest.importorskip("openai")
    from leakwatch.copilot.llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        LLMSettings("http://127.0.0.1:9/v1", "k", "m"), {"timeout_s": 1, "max_retries": 0}
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        tc = SimpleNamespace(
            id="c1", function=SimpleNamespace(name="run_sql", arguments='{"query": "SELECT 1"}')
        )
        msg = SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    resp = llm.chat([{"role": "user", "content": "q"}], [{"type": "function"}], json_mode=True)
    assert resp.tool_calls[0].name == "run_sql" and resp.tool_calls[0].arguments == {
        "query": "SELECT 1"
    }
    assert captured["tool_choice"] == "auto" and captured["response_format"] == {
        "type": "json_object"
    }
    assert "seed" not in captured

    def fail(**kwargs):
        raise openai.APIConnectionError(request=SimpleNamespace(method="POST", url="http://x"))

    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "q"}])
