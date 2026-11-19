"""Command-line chat: no model means a clear exit, never a fake answer."""

from __future__ import annotations

import json

import pytest

from leakwatch.copilot import chat, testing
from leakwatch.copilot.llm import ScriptedLLM, calls, reply, tool_call
from leakwatch.copilot.settings import NOT_CONFIGURED_MESSAGE


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    context = testing.tiny_context(testing.build_tiny_lake(tmp_path / "lake"))
    monkeypatch.setattr(chat, "ToolContext", lambda: context)
    monkeypatch.setattr(chat, "llm_settings", lambda: None)
    return context


def scripted(monkeypatch):
    llm = ScriptedLLM([
        calls(tool_call("list_alerts", area="A")),
        reply("Area A has 2 alerts; the strongest scored 9.25 against 3.0."),
    ])  # fmt: skip
    monkeypatch.setattr(chat, "build_llm", lambda settings, cfg: llm)
    return llm


def test_without_a_model_it_exits_with_the_setup_message(ctx, monkeypatch, capsys):
    monkeypatch.setattr(chat, "build_llm", lambda settings, cfg: None)
    assert chat.main(["How many alerts?"]) == 2
    assert NOT_CONFIGURED_MESSAGE in capsys.readouterr().err


def test_answer_and_tool_calls_are_printed(ctx, monkeypatch, capsys):
    scripted(monkeypatch)
    assert chat.main(["How many alerts in area A?", "--tools", "--version", "v3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Area A has 2 alerts")
    assert "- list_alerts(" in out and "[ok," in out


def test_json_output_carries_the_guardrail_result(ctx, monkeypatch, capsys):
    scripted(monkeypatch)
    assert chat.main(["How many alerts in area A?", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["model"] == "scripted" and doc["refused"] is False
    assert [c["name"] for c in doc["tool_calls"]] == ["list_alerts"]
    assert doc["guardrail"]["unsupported_after"] == []
