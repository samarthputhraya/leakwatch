"""Command-line chat: ``python -m leakwatch.copilot.chat "Which area had most alerts?"``.

Prints the answer, then (``--tools``) the tool calls with the SQL that ran. Exits with status 2
and a clear message when no model is configured (no fake answers).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from leakwatch.copilot.agent import CopilotAgent
from leakwatch.copilot.llm import build_llm
from leakwatch.copilot.prompts import VERSIONS
from leakwatch.copilot.settings import NOT_CONFIGURED_MESSAGE, llm_settings
from leakwatch.copilot.tools import ToolContext


def main(argv: list[str] | None = None) -> int:
    """Ask one question and print the answer."""
    parser = argparse.ArgumentParser(description="Ask the LeakWatch copilot one question.")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("--version", choices=VERSIONS, help="prompt version (default: config)")
    parser.add_argument("--tools", action="store_true", help="also print the tool calls")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)
    settings = llm_settings()
    ctx = ToolContext()
    llm = build_llm(settings, ctx.cfg["llm"])
    if llm is None:
        sys.stderr.write(NOT_CONFIGURED_MESSAGE + "\n")
        return 2
    result = CopilotAgent(llm, ctx, args.version).ask(args.question)
    if args.json:
        doc = {
            "answer": result.answer,
            "prompt_version": result.prompt_version,
            "model": result.model,
            "latency_s": result.latency_s,
            "refused": result.refused,
            "tool_calls": [asdict(c) for c in result.tool_calls],
            "charts": len(result.charts),
            "guardrail": asdict(result.guardrail),
            "error": result.error,
        }
        sys.stdout.write(json.dumps(doc, indent=2, default=str) + "\n")
        return 0 if result.error is None else 1
    sys.stdout.write(result.answer.rstrip() + "\n")
    if args.tools:
        for call in result.tool_calls:
            status = "ok" if call.ok else f"error: {call.error}"
            sys.stdout.write(
                f"- {call.name}({json.dumps(call.arguments)}) [{status}, {call.elapsed_ms} ms]\n"
            )
            if call.sql:
                sys.stdout.write(f"  sql: {call.sql}\n")
        if result.charts:
            sys.stdout.write(f"({len(result.charts)} chart(s) available in the dashboard)\n")
    return 0 if result.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
