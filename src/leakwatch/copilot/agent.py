"""The copilot loop (PRD §9): system prompt + tools + guardrails, and the API the dashboard calls.

``ask(question, history)`` returns a :class:`CopilotAnswer` (``answer``, ``tool_calls``, ``charts``
and the guardrail report); it also unpacks as ``answer, tool_calls, charts = ask(...)``.

One turn:

1. (v3) clear off-topic requests and prompt injection get the fixed refusal without a model call;
2. the model answers through the tool loop (``llm.run_tool_loop``);
3. (v3) guardrails: every number, date and time in the draft must appear in this turn's tool
   outputs or the question, and an answer that used manual passages must cite one or say the
   manuals do not cover it. On failure the model gets one correction message and may call tools
   again; if the regenerated answer still fails, the answer is built from the tool outputs
   themselves (numbers) or says the manuals do not cover the question (citations).
   For v1/v2 the same checks are measured but not enforced (for the evaluation).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from leakwatch.common.logging import get_logger
from leakwatch.copilot import guardrails
from leakwatch.copilot.llm import LLMClient, LLMError, ToolCall, build_llm, run_tool_loop
from leakwatch.copilot.prompts import PromptSpec, load_prompt
from leakwatch.copilot.settings import NOT_CONFIGURED_MESSAGE, llm_settings
from leakwatch.copilot.tools import ToolContext, ToolRegistry, ToolResult, fmt_ts

log = get_logger("leakwatch.copilot.agent")


@dataclass
class ToolCallRecord:
    """A tool call as shown to the operator."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None
    elapsed_ms: float
    sql: str | None = None


@dataclass
class GuardrailReport:
    """What the guardrails found and did in one turn."""

    enforced: bool
    scope_prefilter: bool = False
    draft_answer: str | None = None
    numbers_checked: list[str] = field(default_factory=list)
    unsupported_before: list[str] = field(default_factory=list)
    unsupported_after: list[str] = field(default_factory=list)
    citation_required: bool = False
    citation_ok_before: bool | None = None
    citation_ok_after: bool | None = None
    regenerated: bool = False
    fallback: str | None = None  # "tool_output" | "not_covered"


@dataclass
class CopilotAnswer:
    """The copilot's reply to one question."""

    answer: str
    tool_calls: list[ToolCallRecord]
    charts: list[dict[str, Any]]
    refused: bool
    prompt_version: str
    latency_s: float
    guardrail: GuardrailReport
    results: list[ToolResult] = field(default_factory=list, repr=False)
    error: str | None = None
    model: str | None = None

    def __iter__(self) -> Iterator[Any]:
        """``answer, tool_calls, charts = ask(...)``."""
        yield self.answer
        yield self.tool_calls
        yield self.charts


def _record(result: ToolResult) -> ToolCallRecord:
    sql = result.output.get("sql") if result.name == "run_sql" else None
    return ToolCallRecord(
        result.name, result.arguments, result.error is None, result.error, result.elapsed_ms, sql
    )


def _table(columns: Sequence[str], rows: Sequence[Sequence[Any]], limit: int = 10) -> str:
    head = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows[:limit]
    ]
    more = [f"({len(rows) - limit} more rows)"] if len(rows) > limit else []
    return "\n".join([head, sep, *body, *more])


def render_tool_fallback(results: Sequence[ToolResult]) -> str:
    """An answer made only of tool output (used when the regenerated answer still fails)."""
    parts = ["I could not phrase an answer with verified numbers, so here are the tool results:"]
    for res in results:
        if res.error or res.name in ("describe_schema", "search_manual"):
            continue
        out = res.output
        if res.name == "run_sql" and "rows" in out:
            parts.append(f"run_sql: {out.get('sql')}\n{_table(out['columns'], out['rows'])}")
            continue
        if res.name == "log_complaint":
            parts.append(f"log_complaint: {out.get('reply')}")
            continue
        scalars = {
            k: v for k, v in out.items() if isinstance(v, str | int | float | bool) and k != "chart"
        }
        listing = "; ".join(f"{k}: {v}" for k, v in scalars.items())
        block = [f"{res.name}: {listing}" if listing else f"{res.name}:"]
        for key, value in out.items():
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                cols = [c for c, v in value[0].items() if isinstance(v, str | int | float | bool)]
                block.append(
                    f"{key}:\n" + _table(cols, [[row.get(c) for c in cols] for row in value])
                )
        parts.append("\n".join(block))
    if len(parts) == 1:
        return "I could not answer this with verified numbers from the tools."
    return "\n\n".join(parts)


def _not_covered(passages: Sequence[Mapping[str, Any]]) -> str:
    text = guardrails.NOT_COVERED_TEXT
    if passages:
        closest = "; ".join(str(p.get("citation")) for p in passages[:3])
        text += f" Closest passages found: {closest}."
    return text


class CopilotAgent:
    """Answers operator questions with a model, the tools and the guardrails."""

    def __init__(
        self,
        llm: LLMClient | None,
        ctx: ToolContext | None = None,
        prompt_version: str | None = None,
    ) -> None:
        """Create the agent.

        Args:
            llm: the model (None: answers explain that no model is configured).
            ctx: tool context (default: the configured lake).
            prompt_version: v1, v2 or v3 (default: ``copilot.prompt_version``).

        """
        self.ctx = ctx or ToolContext()
        self.llm = llm
        if self.ctx.llm is None:
            self.ctx.llm = llm
        self.spec: PromptSpec = load_prompt(prompt_version or str(self.ctx.cfg["prompt_version"]))
        self.ctx.sql_hints = self.spec.sql_hints
        self.registry = ToolRegistry(self.ctx)

    @property
    def acfg(self) -> dict[str, Any]:
        """``copilot.agent`` config."""
        return self.ctx.cfg["agent"]

    def system_prompt(self) -> str:
        """The rendered system prompt for this version."""
        values = {
            "data_clock": fmt_ts(self.ctx.data_clock()) or "unknown",
            "refusal": guardrails.REFUSAL_TEXT,
            "not_covered": guardrails.NOT_COVERED_TEXT,
            "max_sql_retries": self.spec.max_sql_retries,
            "schema": self.ctx.schema_text() if self.spec.include_schema else "",
        }
        return self.spec.render_system(values)

    def _history(self, history: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
        keep = int(self.acfg.get("history_messages", 12))
        msgs = [
            {"role": m["role"], "content": str(m.get("content") or "")}
            for m in (history or [])
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        return msgs[-keep:] if keep > 0 else []

    def ask(
        self, question: str, history: Sequence[Mapping[str, Any]] | None = None
    ) -> CopilotAnswer:
        """Answer one question (see module docstring)."""
        t0 = time.perf_counter()
        spec = self.spec
        report = GuardrailReport(enforced=spec.guardrails)
        results: list[ToolResult] = []

        def finish(answer: str, error: str | None = None) -> CopilotAnswer:
            return CopilotAnswer(
                answer=answer,
                tool_calls=[_record(r) for r in results],
                charts=[r.chart for r in results if r.chart],
                refused=guardrails.is_refusal(answer),
                prompt_version=spec.version,
                latency_s=round(time.perf_counter() - t0, 3),
                guardrail=report,
                results=results,
                error=error,
                model=getattr(self.llm, "model", None),
            )

        if spec.guardrails and guardrails.is_out_of_scope(question):
            report.scope_prefilter = True
            return finish(guardrails.REFUSAL_TEXT)
        if self.llm is None:
            return finish(NOT_CONFIGURED_MESSAGE, error="llm_not_configured")

        max_chars = int(self.acfg.get("max_tool_output_chars", 12000))

        def execute(call: ToolCall) -> str:
            result = self.registry.call(call.name, call.arguments)
            results.append(result)
            return result.content(max_chars)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            *self._history(history),
            {"role": "user", "content": question},
        ]
        tools = self.registry.schemas()
        try:
            outcome = run_tool_loop(
                self.llm, messages, tools, execute, int(self.acfg["max_tool_rounds"])
            )
            draft = outcome.content.strip()
            report.draft_answer = draft
            answer = self._guard(question, draft, messages, tools, execute, results, report)
        except LLMError as exc:
            log.warning("model call failed: %s", exc)
            # llm_unavailable: rate limit, timeout or server error (a retry may work);
            # llm_error: the request was rejected or the reply was unusable
            kind = "llm_unavailable" if getattr(exc, "transient", False) else "llm_error"
            return finish(
                "The language model did not respond, so I cannot answer right now. Alerts, charts "
                "and template work orders still work; please try again in a moment.",
                error=f"{kind}: {exc}",
            )
        return finish(answer)

    # -- guardrails ----------------------------------------------------------------------------

    def _checks(
        self, question: str, answer: str, results: Sequence[ToolResult]
    ) -> tuple[guardrails.NumberCheck, guardrails.CitationCheck | None, list[dict[str, Any]]]:
        max_chars = int(self.acfg.get("max_tool_output_chars", 12000))
        sources = [r.content(max_chars) for r in results]
        trivial = [float(x) for x in self.ctx.cfg["guardrails"].get("trivial_numbers", [0, 1])]
        if guardrails.is_refusal(answer):
            numbers = guardrails.NumberCheck(ok=True, checked=[], unsupported=[])
        else:
            numbers = guardrails.check_numbers(answer, sources, question, trivial)
        passages = [
            p for r in results if r.name == "search_manual" for p in r.output.get("passages", [])
        ]
        manual_used = any(r.name == "search_manual" and not r.error for r in results)
        citations = guardrails.check_citations(answer, passages) if manual_used else None
        return numbers, citations, passages

    def _guard(
        self,
        question: str,
        draft: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        execute: Any,
        results: list[ToolResult],
        report: GuardrailReport,
    ) -> str:
        numbers, citations, passages = self._checks(question, draft, results)
        report.numbers_checked = numbers.checked
        report.unsupported_before = list(numbers.unsupported)
        report.citation_required = citations is not None
        report.citation_ok_before = citations.ok if citations is not None else None
        if not self.spec.guardrails:
            report.unsupported_after = list(numbers.unsupported)
            report.citation_ok_after = report.citation_ok_before
            return draft
        answer = draft
        regenerations = int(self.ctx.cfg["guardrails"].get("max_regenerations", 1))
        for _ in range(regenerations):
            if numbers.ok and (citations is None or citations.ok):
                break
            corrections = []
            if not numbers.ok:
                corrections.append(guardrails.number_correction_message(numbers.unsupported))
            if citations is not None and not citations.ok:
                corrections.append(guardrails.citation_correction_message(passages))
            messages.append({"role": "user", "content": "\n\n".join(corrections)})
            report.regenerated = True
            outcome = run_tool_loop(
                self.llm,  # type: ignore[arg-type]
                messages,
                tools,
                execute,
                int(self.acfg.get("max_correction_rounds", 3)),
            )
            answer = outcome.content.strip()
            numbers, citations, passages = self._checks(question, answer, results)
        if citations is not None and not citations.ok:
            report.fallback = "not_covered"
            answer = _not_covered(passages)
            numbers, citations, passages = self._checks(question, answer, results)
        if not numbers.ok:
            report.fallback = "tool_output"
            answer = render_tool_fallback(results)
            numbers, citations, _ = self._checks(question, answer, results)
        report.unsupported_after = list(numbers.unsupported)
        report.citation_ok_after = citations.ok if citations is not None else None
        return answer


_default_agent: CopilotAgent | None = None


def get_default_agent(prompt_version: str | None = None) -> CopilotAgent:
    """The agent for the configured model and lake (created once per process)."""
    global _default_agent
    if _default_agent is None or (prompt_version and _default_agent.spec.version != prompt_version):
        ctx = ToolContext()
        llm = build_llm(llm_settings(), ctx.cfg["llm"])
        _default_agent = CopilotAgent(llm, ctx, prompt_version)
    return _default_agent


def ask(
    question: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    prompt_version: str | None = None,
) -> CopilotAnswer:
    """Dashboard API: answer ``question`` given earlier ``history`` (OpenAI-style messages)."""
    return get_default_agent(prompt_version).ask(question, history)
