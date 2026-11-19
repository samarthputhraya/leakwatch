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
    with pytest.raises(LLMError) as err:
        llm.chat([{"role": "user", "content": "q"}])
    assert err.value.transient  # a retry later may work


def _reply_ok(**kwargs):
    msg = SimpleNamespace(content="done", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def test_rate_limits_are_waited_out_then_reported_as_transient(monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    from leakwatch.copilot import llm as llm_mod

    sleeps: list[float] = []
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    request = httpx.Request("POST", "http://x/v1/chat/completions")

    def limited(headers):
        response = httpx.Response(429, headers=headers, request=request)
        return openai.RateLimitError("slow down", response=response, body=None)

    attempts = iter([limited({"retry-after": "2"}), limited({}), None])

    def create(**kwargs):
        exc = next(attempts)
        if exc is not None:
            raise exc
        return _reply_ok()

    cfg = {"max_retries": 0, "rate_limit_retries": 3, "rate_limit_max_wait_s": 60}
    llm = llm_mod.OpenAICompatibleLLM(LLMSettings("http://x/v1", "k", "m"), cfg)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert llm.chat([{"role": "user", "content": "q"}]).content == "done"
    assert sleeps == [2.0, 10.0] and llm.wait_s == 12.0  # Retry-After, then 5 s doubled

    def always(**kwargs):
        raise limited({"retry-after": "500"})

    llm = llm_mod.OpenAICompatibleLLM(
        LLMSettings("http://x/v1", "k", "m"), {**cfg, "rate_limit_retries": 1}
    )
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=always)))
    sleeps.clear()
    with pytest.raises(LLMError) as err:
        llm.chat([{"role": "user", "content": "q"}])
    assert err.value.transient and sleeps == [60.0]  # capped at rate_limit_max_wait_s


def test_requests_are_paced_to_the_configured_rate(monkeypatch):
    pytest.importorskip("openai")
    from leakwatch.copilot import llm as llm_mod

    clock = {"t": 1000.0}
    sleeps: list[float] = []

    def sleep(s):
        sleeps.append(round(s, 3))
        clock["t"] += s

    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(llm_mod.time, "sleep", sleep)
    llm = llm_mod.OpenAICompatibleLLM(LLMSettings("http://x/v1", "k", "m", rpm=30), {})
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_reply_ok))
    )
    llm.chat([{"role": "user", "content": "q"}])
    clock["t"] += 0.5  # the model answered in 0.5 s
    llm.chat([{"role": "user", "content": "q"}])
    assert sleeps == [1.5] and llm.wait_s == 1.5  # 30/min = one request every 2 s


def test_rpm_setting_is_parsed():
    base = {"LLM_BASE_URL": "http://x/v1", "LLM_MODEL": "m"}
    assert llm_settings({**base, "LLM_RPM": "30"}).rpm == 30.0
    assert llm_settings({**base, "LLM_RPM": ""}).rpm is None
    assert llm_settings({**base, "LLM_RPM": "0"}).rpm is None
    with pytest.raises(ValueError, match="LLM_RPM"):
        llm_settings({**base, "LLM_RPM": "fast"})
