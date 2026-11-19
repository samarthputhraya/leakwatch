"""``make pe-eval``: evaluate the copilot's prompt versions (PRD §9.6).

Questions: ``evals/copilot_questions.jsonl``.

Expected answers are never stored: data items carry a reference SQL query that runs on the lake at
evaluation time, complaint items compute their expected match with SQL, procedure items name the
document sections that should be cited, and out-of-scope items expect a refusal.

Item fields (one JSON object per line):

- common: ``id``, ``category`` (``data`` | ``procedure`` | ``out_of_scope`` | ``complaint``),
  ``question``, ``expected_tools_any`` (at least one must be called; ``[]`` = no tool),
  ``requires`` (lake views that must exist, else the item is skipped);
- data: ``reference_sql``, ``check`` (``number`` | ``string`` | ``area`` | ``sensor`` | ``date`` |
  ``dates`` | ``chart``), ``column`` (default: first), ``tolerance`` (relative, numbers),
  ``sql`` (true: counts towards SQL execution accuracy), ``chart_sensor`` and ``mock_call``
  (chart items);
- procedure: ``expected_sources`` (``[{doc, section}]``; any one cited is correct),
  ``requires_docs`` (index documents that must exist);
- complaint (synthetic, ``"synthetic": true``): ``text``, ``reported_at`` or ``reported_at_sql``,
  ``expected`` (``area``, ``issue_type``).

Metrics written to ``reports/metrics/pe_eval_<version>.json``: SQL execution accuracy (the model's
result set equals the reference result set, column order ignored), answer correctness (overall and
per category), hallucinated-number rate before and after the guardrail (share of non-refused
answers with a number not found in the turn's tool outputs), tool-call correctness, citation
correctness (an expected section is cited) and grounding (every citation matches a retrieved
passage), refusal accuracy, complaint extraction and match accuracy, latency.

With no model configured the command exits with status 2 and writes nothing. ``--mock`` runs a
scripted oracle model (reference SQL, first passage, fixed refusal) to test the harness; it never
writes metrics.

Free, rate-limited endpoints: every scored answer is appended to
``paths.pe_eval_progress_dir() / <version>.progress.jsonl``. When the model becomes unavailable
(rate limit or daily quota after the client's retries, timeout, server error) the run stops
with status 3 instead of scoring that item as wrong; running the same command again continues
with the next item, and a finished version is not asked again (``--fresh`` asks everything
again). Answers are reused only for the same model, code commit, prompt files and item text.
Latency excludes time spent waiting for the rate limit (reported as ``wait_s``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import git_commit, latency_stats, write_metrics
from leakwatch.copilot import guardrails
from leakwatch.copilot.agent import CopilotAgent, CopilotAnswer
from leakwatch.copilot.llm import (
    LLMResponse,
    ScriptedLLM,
    build_llm,
    calls,
    last_tool_output,
    reply,
    tool_call,
)
from leakwatch.copilot.prompts import PROMPTS_DIR, VERSIONS
from leakwatch.copilot.settings import NOT_CONFIGURED_MESSAGE, llm_settings
from leakwatch.copilot.tools import ToolContext, fmt_ts, jsonable

log = get_logger("leakwatch.copilot.evaluate")

ITEMS_PATH = paths.REPO_ROOT / "evals" / "copilot_questions.jsonl"
CATEGORIES = ("data", "procedure", "out_of_scope", "complaint")
CHECKS = ("number", "string", "area", "sensor", "date", "dates", "chart")
#: PRD §10 aims (design targets, not results).
TARGETS = {
    "sql_execution_accuracy": 0.85,
    "hallucinated_number_rate_after": 0.0,
    "refusal_accuracy": 0.90,
}
FAR_FUTURE = "TIMESTAMPTZ '9999-12-31 00:00:00+00'"


def load_items(path: Path = ITEMS_PATH) -> list[dict[str, Any]]:
    """Read and validate the question set."""
    items = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if line.strip():
                item = json.loads(line)
                validate_item(item, n)
                items.append(item)
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item ids")
    return items


def validate_item(item: Mapping[str, Any], line: int = 0) -> None:
    """Raise ValueError if an item misses a field its category needs."""
    where = f"item {item.get('id', '?')} (line {line})"
    if item.get("category") not in CATEGORIES:
        raise ValueError(f"{where}: unknown category {item.get('category')!r}")
    for key in ("id", "question", "expected_tools_any"):
        if key not in item:
            raise ValueError(f"{where}: missing {key}")
    cat = item["category"]
    if cat == "data":
        if item.get("check") not in CHECKS:
            raise ValueError(f"{where}: check must be one of {CHECKS}")
        if item["check"] != "chart" and not item.get("reference_sql"):
            raise ValueError(f"{where}: data items need reference_sql")
        if item["check"] == "chart" and not (item.get("chart_sensor") and item.get("mock_call")):
            raise ValueError(f"{where}: chart items need chart_sensor and mock_call")
    if cat == "procedure" and not item.get("expected_sources"):
        raise ValueError(f"{where}: procedure items need expected_sources")
    if cat == "complaint":
        if not item.get("synthetic"):
            raise ValueError(f"{where}: complaints must be labelled synthetic")
        if not item.get("text") or not (item.get("reported_at") or item.get("reported_at_sql")):
            raise ValueError(f"{where}: complaints need text and reported_at(_sql)")
        if "expected" not in item:
            raise ValueError(f"{where}: complaints need expected")


# ------------------------------------------------------------------------------------------
# Preparing items (expected values computed now)
# ------------------------------------------------------------------------------------------


@dataclass
class Prepared:
    """An item with its question text and expected values for this lake."""

    item: dict[str, Any]
    question: str
    skip: str | None = None
    reference: pd.DataFrame | None = None
    reported_at: pd.Timestamp | None = None
    expected_strict: set[str] = field(default_factory=set)
    expected_wide: set[str] = field(default_factory=set)


def _complaint_sets(
    ctx: ToolContext, area: str | None, reported: pd.Timestamp
) -> tuple[set[str], set[str]]:
    if area is None or "alerts" not in ctx.views():
        return set(), set()
    window = pd.Timedelta(hours=float(ctx.cfg["complaints"]["match_window_hours"]))
    sql = (
        "SELECT alert_id FROM alerts WHERE area = ? AND start_ts <= ? "
        f"AND coalesce(end_ts, {FAR_FUTURE}) >= ?"
    )
    strict = set(ctx.query(sql, [area, reported + window, reported - window])["alert_id"])
    wide_lo = reported - window - pd.Timedelta(hours=48)
    wide = set(ctx.query(sql, [area, reported + window, wide_lo])["alert_id"])
    return strict, wide


def prepare(item: Mapping[str, Any], ctx: ToolContext, index_docs: set[str] | None) -> Prepared:
    """Compute the expected values of one item (or the reason it is skipped)."""
    prep = Prepared(item=dict(item), question=str(item["question"]))
    missing = [v for v in item.get("requires", []) if v not in ctx.views()]
    if missing:
        prep.skip = f"lake has no {', '.join(missing)}"
        return prep
    need_docs = [
        d for d in item.get("requires_docs", []) if index_docs is not None and d not in index_docs
    ]
    if need_docs:
        prep.skip = f"index has no {', '.join(need_docs)}"
        return prep
    try:
        if item.get("reference_sql"):
            prep.reference = ctx.query(str(item["reference_sql"]), timeout_s=120)
            if item["category"] == "data" and item.get("check") != "chart":
                if prep.reference.empty:
                    prep.skip = "reference result is empty"
                    return prep
                col = item.get("column") or prep.reference.columns[0]
                first = prep.reference[col].iloc[0]
                if first is None or (not isinstance(first, str) and pd.isna(first)):
                    prep.skip = "reference value is null"
                    return prep
        if item["category"] == "complaint":
            if item.get("reported_at_sql"):
                got = ctx.query(str(item["reported_at_sql"]))
                if got.empty or pd.isna(got.iloc[0, 0]):
                    prep.skip = "reported_at_sql returned nothing"
                    return prep
                prep.reported_at = pd.Timestamp(got.iloc[0, 0]).tz_convert("UTC").floor("min")
            else:
                prep.reported_at = pd.Timestamp(item["reported_at"], tz="UTC")
            when = fmt_ts(prep.reported_at)
            prep.question = f'Resident complaint (synthetic) received {when}: "{item["text"]}"'
            prep.expected_strict, prep.expected_wide = _complaint_sets(
                ctx, item["expected"].get("area"), prep.reported_at
            )
    except Exception as exc:
        prep.skip = f"reference failed: {type(exc).__name__}: {str(exc)[:200]}"
    return prep


# ------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------


def _number_in(answer: str, expected: float, rel_tol: float) -> bool:
    found = guardrails.extract(answer).numbers  # inclusive: a count may look like a year
    if expected == 0 and re.search(r"\b(?:no|none|zero|not any)\b", answer, re.I):
        return True
    for m in found:
        targets = [abs(expected)] + ([abs(expected) * 100] if m.percent else [])
        if any(abs(m.value - t) <= max(m.tolerance, rel_tol * abs(t)) for t in targets):
            return True
    return False


def _date_in(answer: str, value: Any) -> bool:
    day = (
        str(value)[:10]
        if isinstance(value, str)
        else pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d")
    )
    found = guardrails.extract(answer, answer=True)
    return day in found.dates or day in answer


def _token_in(answer: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", answer, re.I))


def answer_correct(item: Mapping[str, Any], prep: Prepared, ans: CopilotAnswer) -> bool:
    """Data items: does the answer state the reference result?"""
    check = item["check"]
    if check == "chart":
        sensor = str(item["chart_sensor"])
        return any(
            r.name == "get_sensor_series" and r.chart and r.output.get("sensor_id") == sensor
            for r in ans.results
        )
    ref = prep.reference
    assert ref is not None
    col = item.get("column") or ref.columns[0]
    values = ref[col].tolist()
    text = ans.answer
    if check == "number":
        return _number_in(text, float(values[0]), float(item.get("tolerance", 0.01)))
    if check == "area":
        return bool(
            re.search(rf"\b(?:area|zone)\b\W{{0,4}}{re.escape(str(values[0]))}\b", text, re.I)
        )
    if check == "sensor":
        value = str(values[0])
        return _token_in(text, value) or _token_in(text, value.split("__")[-1])
    if check == "string":
        return str(values[0]).lower() in text.lower()
    if check == "date":
        return _date_in(text, values[0])
    if check == "dates":
        return all(_date_in(text, v) for v in values)
    raise ValueError(check)


def _norm_cell(value: Any) -> str:
    v = jsonable(value)
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int | float):
        return f"{float(v):.4g}"
    return str(v)


def result_matches(
    columns: Sequence[str], rows: Sequence[Sequence[Any]], ref: pd.DataFrame
) -> bool:
    """Execution match: same row count and every reference column equals some result column."""
    if len(rows) != len(ref):
        return False
    got = [sorted(_norm_cell(r[j]) for r in rows) for j in range(len(columns))]
    for col in ref.columns:
        want = sorted(_norm_cell(v) for v in ref[col].tolist())
        if want not in got:
            return False
    return True


def _source_cited(citation: str, source: Mapping[str, Any]) -> bool:
    c = citation.lower()
    if str(source["doc"]).lower() not in c:
        return False
    section = str(source["section"])
    number = guardrails.section_number(section)
    return guardrails.section_cited(citation, number, None if number else section)


@dataclass
class Outcome:
    """Scores of one item."""

    id: str
    category: str
    skipped: str | None = None
    question: str = ""
    answer: str = ""
    tools: list[str] = field(default_factory=list)
    latency_s: float | None = None
    refused: bool | None = None
    error: str | None = None
    correct: bool | None = None
    sql_exec_match: bool | None = None
    tool_ok: bool | None = None
    citation_ok: bool | None = None
    citation_grounded: bool | None = None
    refusal_ok: bool | None = None
    hallucinated_before: bool | None = None
    hallucinated_after: bool | None = None
    unsupported_before: list[str] = field(default_factory=list)
    unsupported_after: list[str] = field(default_factory=list)
    regenerated: bool = False
    fallback: str | None = None
    scope_prefilter: bool = False
    complaint_extraction_ok: bool | None = None
    complaint_match_ok: bool | None = None
    wait_s: float = 0.0  # rate-limit pacing/waiting, excluded from latency_s


def score(prep: Prepared, ans: CopilotAnswer) -> Outcome:
    """Score one answered item."""
    item = prep.item
    cat = item["category"]
    tools = [c.name for c in ans.tool_calls]
    out = Outcome(
        id=item["id"],
        category=cat,
        question=prep.question,
        answer=ans.answer[:800],
        tools=tools,
        latency_s=ans.latency_s,
        refused=ans.refused,
        error=ans.error,
        regenerated=ans.guardrail.regenerated,
        fallback=ans.guardrail.fallback,
        scope_prefilter=ans.guardrail.scope_prefilter,
    )
    expected_tools = list(item.get("expected_tools_any") or [])
    out.tool_ok = (not tools) if not expected_tools else any(t in tools for t in expected_tools)
    out.refusal_ok = ans.refused == (cat == "out_of_scope")
    if not ans.refused:
        out.unsupported_before = list(ans.guardrail.unsupported_before)
        out.unsupported_after = list(ans.guardrail.unsupported_after)
        out.hallucinated_before = bool(out.unsupported_before)
        out.hallucinated_after = bool(out.unsupported_after)
    if cat == "out_of_scope":
        out.correct = ans.refused
    elif cat == "data":
        out.correct = (not ans.refused) and answer_correct(item, prep, ans)
        if item.get("sql") and prep.reference is not None:
            out.sql_exec_match = any(
                r.name == "run_sql"
                and not r.error
                and result_matches(r.output["columns"], r.output["rows"], prep.reference)
                for r in ans.results
            )
    elif cat == "procedure":
        cited = guardrails.find_citations(ans.answer)
        passages = [
            p
            for r in ans.results
            if r.name == "search_manual"
            for p in r.output.get("passages", [])
        ]
        out.citation_ok = any(_source_cited(c, s) for c in cited for s in item["expected_sources"])
        out.citation_grounded = bool(cited) and all(
            any(guardrails.citation_matches(c, p) for p in passages) for c in cited
        )
        out.correct = bool(out.citation_ok) and not ans.refused
    elif cat == "complaint":
        logged = [r for r in ans.results if r.name == "log_complaint" and not r.error]
        exp = item["expected"]
        if logged:
            got = logged[-1].output
            ext, match = got.get("extracted", {}), got.get("match", {})
            out.complaint_extraction_ok = ext.get("area") == exp.get("area") and ext.get(
                "issue_type"
            ) == exp.get("issue_type")
            status = match.get("status")
            if exp.get("area") is None:
                ok = status == "need_area"
            elif prep.expected_strict:
                ok = status == "matched" and match.get("alert_id") in prep.expected_wide
            elif not prep.expected_wide:
                ok = status == "no_matching_alert"
            else:  # only alerts in the onset-widened window: either answer is acceptable
                ok = status == "no_matching_alert" or (
                    status == "matched" and match.get("alert_id") in prep.expected_wide
                )
            out.complaint_match_ok = ok
        else:
            out.complaint_extraction_ok = False
            out.complaint_match_ok = False
        out.correct = bool(out.complaint_extraction_ok and out.complaint_match_ok)
    return out


def _rate(values: Sequence[bool | None]) -> dict[str, Any]:
    vals = [bool(v) for v in values if v is not None]
    return {"value": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}


def summarize(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Aggregate metrics over the evaluated items."""
    done = [o for o in outcomes if o.skipped is None]
    by_cat = {c: _rate([o.correct for o in done if o.category == c]) for c in CATEGORIES}
    return {
        "items": {
            "total": len(outcomes),
            "evaluated": len(done),
            "skipped": len(outcomes) - len(done),
            "by_category": {c: sum(1 for o in done if o.category == c) for c in CATEGORIES},
            "skip_reasons": {o.id: o.skipped for o in outcomes if o.skipped},
        },
        "sql_execution_accuracy": _rate([o.sql_exec_match for o in done]),
        "answer_correctness": {"overall": _rate([o.correct for o in done]), "by_category": by_cat},
        "hallucinated_number_rate": {
            "before_guardrail": _rate([o.hallucinated_before for o in done]),
            "after_guardrail": _rate([o.hallucinated_after for o in done]),
        },
        "tool_call_correctness": _rate([o.tool_ok for o in done]),
        "citation_correctness": _rate([o.citation_ok for o in done]),
        "citation_grounded_rate": _rate([o.citation_grounded for o in done]),
        "refusal_accuracy": _rate([o.refusal_ok for o in done]),
        "complaint_extraction_accuracy": _rate([o.complaint_extraction_ok for o in done]),
        "complaint_match_accuracy": _rate([o.complaint_match_ok for o in done]),
        "latency_s": latency_stats(o.latency_s for o in done),
        "latency_by_category_s": {
            c: latency_stats(o.latency_s for o in done if o.category == c) for c in CATEGORIES
        },
        "guardrail_actions": {
            "regenerated": sum(o.regenerated for o in done),
            "fallback_tool_output": sum(o.fallback == "tool_output" for o in done),
            "fallback_not_covered": sum(o.fallback == "not_covered" for o in done),
            "scope_prefilter_refusals": sum(o.scope_prefilter for o in done),
        },
        "errors": sum(1 for o in done if o.error),
        "rate_limit_wait_s": round(sum(o.wait_s for o in done), 1),
    }


# ------------------------------------------------------------------------------------------
# Scripted oracle (harness tests only)
# ------------------------------------------------------------------------------------------


def _answer_from_rows(messages: list[dict[str, Any]], _tools: Any) -> LLMResponse:
    out = last_tool_output(messages)
    if out.get("rows"):
        cells = [f"{c} = {v}" for c, v in zip(out["columns"], out["rows"][0], strict=False)]
        extra = [" | ".join(str(v) for v in row) for row in out["rows"][1:5]]
        return reply("Result: " + "; ".join(cells) + ("; " + "; ".join(extra) if extra else ""))
    if out.get("sensor_id"):
        return reply(
            f"The chart shows {out['sensor_id']} ({out.get('unit')}); mean {out.get('mean')}."
        )
    return reply("No rows were returned.")


def _answer_from_passage(messages: list[dict[str, Any]], _tools: Any) -> LLMResponse:
    passages = last_tool_output(messages).get("passages") or []
    if not passages:
        return reply(guardrails.NOT_COVERED_TEXT)
    return reply(f"See {passages[0]['citation']}.")


def _answer_from_complaint(messages: list[dict[str, Any]], _tools: Any) -> LLMResponse:
    return reply(str(last_tool_output(messages).get("reply") or "Logged."))


def mock_llm_for(prep: Prepared) -> ScriptedLLM:
    """An oracle script for one item (``--mock`` and tests)."""
    item = prep.item
    cat = item["category"]
    if cat == "out_of_scope":
        return ScriptedLLM([reply(guardrails.REFUSAL_TEXT)], model="mock-oracle")
    if cat == "procedure":
        return ScriptedLLM(
            [calls(tool_call("search_manual", query=item["question"])), _answer_from_passage],
            model="mock-oracle",
        )
    if cat == "complaint":
        when = prep.reported_at.strftime("%Y-%m-%d %H:%M") if prep.reported_at is not None else None
        return ScriptedLLM(
            [
                calls(tool_call("log_complaint", text=item["text"], reported_at=when)),
                _answer_from_complaint,
            ],
            model="mock-oracle",
        )
    if item.get("mock_call"):
        mc = item["mock_call"]
        return ScriptedLLM(
            [calls(tool_call(mc["tool"], **mc["args"])), _answer_from_rows], model="mock-oracle"
        )
    return ScriptedLLM(
        [calls(tool_call("run_sql", query=item["reference_sql"])), _answer_from_rows],
        model="mock-oracle",
    )


# ------------------------------------------------------------------------------------------
# Running
# ------------------------------------------------------------------------------------------


class EvalInterruptedError(RuntimeError):
    """The model became unavailable mid-run; scored answers are kept for the next run."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def prompts_fingerprint() -> str:
    """Hash of every prompt file (a changed prompt invalidates saved answers)."""
    files = sorted(PROMPTS_DIR.glob("*.yaml"))
    return _sha("".join(f.name + f.read_text(encoding="utf-8") for f in files))


@dataclass
class Progress:
    """Scored answers of an unfinished run, one JSON line each, behind a header line."""

    path: Path
    meta: dict[str, Any]

    def load(self) -> dict[str, tuple[str, Outcome]]:
        """Item id -> (item hash, outcome) when the header matches ``meta``, else ``{}``."""
        if not self.path.exists():
            return {}
        lines = self.path.read_text(encoding="utf-8").splitlines()
        try:
            if not lines or json.loads(lines[0]).get("meta") != self.meta:
                log.info("%s was made by another model, commit or prompt; starting over", self.path)
                return {}
        except json.JSONDecodeError:
            return {}
        saved: dict[str, tuple[str, Outcome]] = {}
        for line in lines[1:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # cut off by an interrupted write
            saved[row["outcome"]["id"]] = (row["item_sha"], Outcome(**row["outcome"]))
        return saved

    def start(self, keep: bool) -> None:
        """Write a fresh header unless continuing a matching file."""
        if not keep:
            paths.ensure_dir(self.path.parent)
            self.path.write_text(json.dumps({"meta": self.meta}) + "\n", encoding="utf-8")

    def add(self, item_sha: str, outcome: Outcome) -> None:
        """Append one scored answer."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"item_sha": item_sha, "outcome": jsonable(asdict(outcome))}) + "\n"
            )

    def clear(self) -> None:
        """Forget saved answers (``--fresh``)."""
        self.path.unlink(missing_ok=True)


def run_eval(
    version: str,
    items: Sequence[Mapping[str, Any]],
    ctx: ToolContext,
    llm_for: Callable[[Prepared], Any],
    progress: Progress | None = None,
) -> tuple[dict[str, Any], list[Outcome]]:
    """Ask every item with prompt ``version``; ``llm_for(prepared)`` gives the model per item.

    With ``progress``, answers saved by an interrupted run of the same setup are reused and new
    ones are saved; :class:`EvalInterruptedError` is raised when the model becomes unavailable.
    """
    retriever, _ = ctx.get_retriever()
    index_docs = set(retriever.documents) if retriever is not None else set()
    saved = progress.load() if progress is not None else {}
    if progress is not None:
        progress.start(keep=bool(saved))
        if saved:
            log.info("%s: continuing; %d answers saved by an earlier run", version, len(saved))
    outcomes: list[Outcome] = []
    for item in items:
        item_sha = _sha(json.dumps(item, sort_keys=True))
        if item["id"] in saved and saved[item["id"]][0] == item_sha:
            outcomes.append(saved[item["id"]][1])
            continue
        prep = prepare(item, ctx, index_docs)
        if prep.skip:
            outcomes.append(
                Outcome(
                    id=item["id"],
                    category=item["category"],
                    skipped=prep.skip,
                    question=prep.question,
                )
            )
            log.info("%s skipped: %s", item["id"], prep.skip)
            continue
        llm = llm_for(prep)
        waited_before = float(getattr(llm, "wait_s", 0.0))
        ans = CopilotAgent(llm, ctx, version).ask(prep.question)
        if ans.error and ans.error.startswith("llm_unavailable"):
            done = sum(o.skipped is None for o in outcomes)
            raise EvalInterruptedError(
                f"{version} {item['id']}: {ans.error} ({done} answers kept; run the same "
                "command again to continue)"
            )
        outcome = score(prep, ans)
        waited = float(getattr(llm, "wait_s", 0.0)) - waited_before
        if waited > 0 and outcome.latency_s is not None:
            outcome.latency_s = round(max(0.0, outcome.latency_s - waited), 3)
            outcome.wait_s = round(waited, 3)
        outcomes.append(outcome)
        if progress is not None:
            progress.add(item_sha, outcome)
        log.info(
            "%s %s correct=%s tools=%s %.1fs",
            version,
            item["id"],
            outcome.correct,
            outcome.tools,
            ans.latency_s,
        )
    return summarize(outcomes), outcomes


def main(argv: list[str] | None = None) -> int:
    """Evaluate one prompt version (or all) and write ``pe_eval_<version>.json``."""
    parser = argparse.ArgumentParser(description="Evaluate copilot prompt versions (PRD §9.6).")
    parser.add_argument("--version", required=True, choices=[*VERSIONS, "all"])
    parser.add_argument("--items", type=Path, default=ITEMS_PATH)
    parser.add_argument("--only", help="comma-separated item ids or categories")
    parser.add_argument(
        "--mock", action="store_true", help="scripted oracle model; never writes metrics"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="ask every item again (ignore saved answers)"
    )
    args = parser.parse_args(argv)
    items = load_items(args.items)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        items = [i for i in items if i["id"] in keep or i["category"] in keep]
    versions = list(VERSIONS) if args.version == "all" else [args.version]
    settings = llm_settings()
    if not args.mock and not settings.configured:
        log.error(
            "%s Nothing was evaluated and no metrics file was written.", NOT_CONFIGURED_MESSAGE
        )
        return 2
    cfg = load_config()
    if args.mock:
        cfg = copy.deepcopy(cfg)
        cfg["copilot"]["complaints"]["use_llm_extraction"] = False
    for version in versions:
        ctx = ToolContext(full_cfg=cfg)
        if args.mock:
            summary, outcomes = run_eval(version, items, ctx, mock_llm_for)
            log.info(
                "mock run (not written): %s",
                json.dumps(jsonable({k: v for k, v in summary.items() if k != "items"}))[:2000],
            )
            continue
        llm_cfg = dict(ctx.cfg["llm"])  # an unattended run can wait longer for a rate limit
        llm_cfg["rate_limit_retries"] = llm_cfg.get("eval_rate_limit_retries", 6)
        llm = build_llm(settings, llm_cfg)
        ctx.llm = llm
        progress = Progress(
            paths.pe_eval_progress_dir() / f"{version}.progress.jsonl",
            {
                "version": version,
                **settings.describe(),
                "git_commit": git_commit(),
                "prompts": prompts_fingerprint(),
            },
        )
        if args.fresh:
            progress.clear()
        try:
            summary, outcomes = run_eval(
                version, items, ctx, lambda _p, _llm=llm: _llm, progress=progress
            )
        except EvalInterruptedError as exc:
            log.error("model unavailable, stopped: %s", exc)
            return 3
        payload = {
            "version": version,
            "model": settings.describe(),
            "items_file": str(args.items.relative_to(paths.REPO_ROOT))
            if args.items.is_relative_to(paths.REPO_ROOT)
            else str(args.items),
            "data_clock": fmt_ts(ctx.data_clock()),
            **summary,
            "targets": TARGETS,
            "per_item": [asdict(o) for o in outcomes],
        }
        path = write_metrics(f"pe_eval_{version}", payload)
        log.info("%s -> %s", version, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
