"""Work orders for alerts (PRD §9.3 "Explain and dispatch", §9.1 ``demo_safe_mode``).

CLI: ``make dispatch``.

For one alert:

1. **Evidence** from the tools: ``get_alert``, ``estimate_loss``, ``get_gnn_candidates``, the pipe
   list and passages from ``search_manual`` (if the index exists).
2. **Pipes to inspect**: with localizer candidates (``gnn_top5``) the pipes at those nodes, most
   probable first; otherwise the pipes nearest the top sensors (``network_index.plan_pipes``).
3. **Priority** by operator policy (``copilot.dispatch.priority``): P1 if the estimated loss or the
   score-to-threshold ratio reaches the P1 level, P2 likewise, else P3.
4. **Text**: with a model, structured output (JSON validated with pydantic). The result is grounded:
   alert id, area, loss rate, pipe order and priority are taken from the evidence and the policy,
   citations must be retrieved passages, and every number in the text must appear in the evidence
   (one correction round, then the template). Without a model, or when the model fails and
   ``demo_safe_mode`` is on, a fixed template renders the same evidence, labelled
   "template fallback".
5. **Write** one Parquet file per order to ``lake/work_orders/`` with ``WORK_ORDER_COLUMNS``
   (pyarrow). The ``alerts`` view joins the latest order per alert (``common.lake``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, ValidationError

from leakwatch.common import lake
from leakwatch.common.logging import get_logger
from leakwatch.common.schemas import WORK_ORDER_COLUMNS, ManualCitation, Priority, WorkOrder
from leakwatch.copilot.guardrails import check_numbers, citation_matches
from leakwatch.copilot.network_index import plan_pipes
from leakwatch.copilot.prompts import fill, load_named_prompt
from leakwatch.copilot.tools import (
    ToolContext,
    alert_view,
    estimate_loss,
    fetch_alert,
    get_gnn_candidates,
    jsonable,
    search_manual,
)

log = get_logger("leakwatch.copilot.dispatch")

TEMPLATE_LABEL = "[template fallback]"


class WorkOrderError(RuntimeError):
    """The model's work order could not be used."""


class WorkOrderDraft(BaseModel):
    """The JSON the model must return (fields it may write)."""

    model_config = ConfigDict(extra="ignore")

    priority: Priority | None = None
    summary: str
    pipes_to_inspect: list[str] = []
    evidence: list[str]
    recommended_actions: list[str]
    manual_citations: list[ManualCitation] = []


@dataclass
class Evidence:
    """Tool outputs a work order is built from."""

    alert: dict[str, Any]
    loss: dict[str, Any]
    candidates: dict[str, Any]
    pipes: list[str]
    pipes_basis: str
    passages: list[dict[str, Any]]
    priority: str
    priority_reason: str
    policy: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """The evidence the model sees (and the number check uses)."""
        return {
            "alert": self.alert,
            "estimate_loss": self.loss,
            "localizer": self.candidates,
            "pipes_to_inspect": self.pipes,
            "pipes_basis": self.pipes_basis,
            "priority": self.priority,
            "priority_reason": self.priority_reason,
            "priority_policy": self.policy,
            "manual_passages": [
                {k: p.get(k) for k in ("citation", "doc_title", "section", "page", "text")}
                for p in self.passages
            ],
        }

    def as_json(self) -> str:
        """``as_dict`` as JSON text."""
        return json.dumps(self.as_dict(), ensure_ascii=False, default=str)


def priority_for(
    alert: Mapping[str, Any], loss_m3h: float | None, policy: Mapping[str, Any]
) -> tuple[str, str]:
    """P1/P2/P3 and the reason, from the loss estimate and the score-to-threshold ratio."""
    ratio = alert.get("score_to_threshold")
    loss = loss_m3h if loss_m3h is not None else None

    def reason(level: str) -> str | None:
        parts = []
        if loss is not None and loss >= float(policy[f"{level}_loss_m3h"]):
            parts.append(f"estimated loss {loss} m3/h >= {policy[f'{level}_loss_m3h']} m3/h")
        if ratio is not None and ratio >= float(policy[f"{level}_score_ratio"]):
            parts.append(f"score {ratio}x the threshold >= {policy[f'{level}_score_ratio']}x")
        return " and ".join(parts) or None

    for level, name in (("p1", "P1"), ("p2", "P2")):
        why = reason(level)
        if why:
            return name, why
    return "P3", "below the P2 levels for estimated loss and score"


def gather_evidence(ctx: ToolContext, alert_id: str) -> Evidence:
    """Run the tools for one alert."""
    dcfg = ctx.cfg["dispatch"]
    row = fetch_alert(ctx, alert_id)
    alert = alert_view(row)
    loss = estimate_loss(ctx, {"alert_id": row["alert_id"]})
    candidates = get_gnn_candidates(ctx, {"alert_id": row["alert_id"]})
    pipes, basis = plan_pipes(
        row, ctx.get_network(), int(dcfg["max_pipes"]), int(dcfg["pipes_per_sensor"])
    )
    passages: list[dict[str, Any]] = []
    manual = search_manual(ctx, {"query": str(dcfg["manual_query"])})
    if manual.get("passages"):
        passages = list(manual["passages"])[: int(dcfg["manual_citations"])]
    rate = loss.get("est_loss_m3h") if loss.get("available") else None
    policy = dict(dcfg["priority"])
    priority, reason = priority_for(alert, rate, policy)
    return Evidence(alert, loss, candidates, pipes, basis, passages, priority, reason, policy)


def citations_from(passages: Sequence[Mapping[str, Any]]) -> list[ManualCitation]:
    """Manual citations for retrieved passages."""
    return [
        ManualCitation(
            doc=str(p.get("doc_title") or p.get("doc_short")),
            chapter=str(p.get("section")),
            page=p["page"] if p.get("page") is not None else "n/a",
        )
        for p in passages
    ]


def _top_sensors_text(alert: Mapping[str, Any]) -> str:
    parts = [
        f"{s['series_id']} (share {s['contribution']})" for s in alert.get("top_sensors") or []
    ]
    return ", ".join(parts) or "none recorded"


def template_work_order(ev: Evidence, actions: Sequence[str], now: datetime) -> WorkOrder:
    """The deterministic work order (``demo_safe_mode``, no model), from the evidence only."""
    a = ev.alert
    loss = ev.loss
    if loss.get("available"):
        loss_text = (
            f"estimate_loss: {loss['est_loss_m3h']} m3/h of excess inflow (an estimate), about "
            f"{loss['est_volume_m3']} m3 over {loss['duration_h']} h"
        )
        size = f"an estimated {loss['est_loss_m3h']} m3/h of excess inflow"
    else:
        loss_text = "estimate_loss: no loss estimate for this alert"
        size = "no loss estimate"
    if ev.candidates.get("available"):
        nodes = ", ".join(
            f"{c['node_id']} ({c['prob']})" for c in ev.candidates.get("candidates", [])
        )
        cand_text = f"get_gnn_candidates: top candidate nodes {nodes}"
    else:
        cand_text = f"get_gnn_candidates: none; {ev.pipes_basis}"
    evidence = [
        (
            f"get_alert: fired {a['start_ts']}, status {a['status']}, K-Means score max "
            f"{a['kmeans_score_max']} vs threshold {a['threshold']}, {a['windows_above']} windows "
            "above the threshold"
        ),
        f"get_alert: top sensors {_top_sensors_text(a)}",
        loss_text,
        cand_text,
        *[f"search_manual: {p['citation']}" for p in ev.passages],
    ]
    summary = (
        f"{TEMPLATE_LABEL} Possible leak in area {a['area']}: the K-Means score reached "
        f"{a['kmeans_score_max']} against a threshold of {a['threshold']} (alert fired "
        f"{a['start_ts']}), with {size}. Priority {ev.priority}: {ev.priority_reason}."
    )
    return WorkOrder(
        alert_id=a["alert_id"],
        priority=ev.priority,  # type: ignore[arg-type]
        summary=summary,
        area=a["area"],
        pipes_to_inspect=list(ev.pipes),
        estimated_loss_m3_per_hour=loss.get("est_loss_m3h") if loss.get("available") else None,
        evidence=evidence,
        recommended_actions=list(actions),
        manual_citations=citations_from(ev.passages),
        generated_at=now,
    )


def _ask_model(llm: Any, messages: list[dict[str, Any]]) -> WorkOrderDraft:
    from leakwatch.copilot.llm import LLMError, json_object_from_text

    try:
        resp = llm.chat(messages, json_mode=True)
    except LLMError as exc:
        if "response_format" not in str(exc) and "json" not in str(exc).lower():
            raise
        resp = llm.chat(messages)  # server without JSON mode
    messages.append({"role": "assistant", "content": resp.content or ""})
    try:
        return WorkOrderDraft.model_validate(json_object_from_text(resp.content or ""))
    except (ValueError, ValidationError) as exc:
        raise WorkOrderError(f"invalid work order JSON: {str(exc)[:300]}") from exc


def llm_work_order(ev: Evidence, llm: Any, now: datetime, max_corrections: int = 1) -> WorkOrder:
    """Structured output from the model, validated and grounded in the evidence."""
    prompt = load_named_prompt("work_order")
    evidence_json = ev.as_json()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": fill(
                prompt["user"], {"alert_id": ev.alert["alert_id"], "evidence_json": evidence_json}
            ),
        },
    ]
    draft = _ask_model(llm, messages)
    for attempt in range(max_corrections + 1):
        text = "\n".join([draft.summary, *draft.evidence, *draft.recommended_actions])
        check = check_numbers(text, [evidence_json])
        if check.ok:
            break
        if attempt == max_corrections:
            raise WorkOrderError(f"numbers not in the evidence: {', '.join(check.unsupported)}")
        messages.append(
            {
                "role": "user",
                "content": fill(
                    prompt["correction"], {"unsupported": ", ".join(check.unsupported)}
                ),
            }
        )
        draft = _ask_model(llm, messages)
    cited = [
        c
        for c in draft.manual_citations
        if any(citation_matches(f"{c.doc}, {c.chapter}, p. {c.page}", p) for p in ev.passages)
    ]
    if draft.priority and draft.priority != ev.priority:
        log.info("model suggested %s; policy priority %s kept", draft.priority, ev.priority)
    loss = ev.loss
    return WorkOrder(
        alert_id=ev.alert["alert_id"],
        priority=ev.priority,  # type: ignore[arg-type]
        summary=draft.summary.strip(),
        area=ev.alert["area"],
        pipes_to_inspect=list(ev.pipes),
        estimated_loss_m3_per_hour=loss.get("est_loss_m3h") if loss.get("available") else None,
        evidence=[e.strip() for e in draft.evidence if e.strip()],
        recommended_actions=[r.strip() for r in draft.recommended_actions if r.strip()],
        manual_citations=cited or citations_from(ev.passages),
        generated_at=now,
    )


@dataclass
class GeneratedOrder:
    """A work order and how it was made."""

    order: WorkOrder
    generator: str  # "llm" | "template"
    prompt_version: str
    notes: list[str]
    evidence: Evidence


def generate_work_order(
    ctx: ToolContext,
    alert_id: str,
    llm: Any | None = None,
    prompt_version: str = "work_order",
    template_only: bool = False,
    now: datetime | None = None,
) -> GeneratedOrder:
    """Work order for one alert: the model when available, else (or on failure) the template."""
    from leakwatch.copilot.llm import LLMError

    now = now or datetime.now(UTC)
    ev = gather_evidence(ctx, alert_id)
    actions = list(load_named_prompt("work_order")["template_actions"])
    notes: list[str] = []
    if llm is not None and not template_only:
        try:
            order = llm_work_order(ev, llm, now)
            return GeneratedOrder(order, "llm", prompt_version, notes, ev)
        except (LLMError, WorkOrderError, ValidationError) as exc:
            if not ctx.cfg["dispatch"].get("demo_safe_mode", True):
                raise
            notes.append(
                f"model failed ({type(exc).__name__}: {str(exc)[:200]}); template fallback"
            )
            log.warning("alert %s: %s", alert_id, notes[-1])
    elif llm is None:
        notes.append("no model configured; template")
    return GeneratedOrder(template_work_order(ev, actions, now), "template", "template", notes, ev)


def work_order_row(gen: GeneratedOrder) -> dict[str, Any]:
    """One ``lake/work_orders`` row (``WORK_ORDER_COLUMNS``)."""
    return {
        "alert_id": str(gen.order.alert_id),
        "work_order_json": gen.order.model_dump_json(),
        "generator": gen.generator,
        "prompt_version": gen.prompt_version,
        "generated_at": gen.order.generated_at,
    }


def write_work_order(gen: GeneratedOrder, lake_root: Path | None = None) -> Path:
    """Write one order as its own Parquet file (written to a temp name, then renamed)."""
    row = work_order_row(gen)
    table = pa.table(
        {
            "alert_id": pa.array([row["alert_id"]], pa.string()),
            "work_order_json": pa.array([row["work_order_json"]], pa.string()),
            "generator": pa.array([row["generator"]], pa.string()),
            "prompt_version": pa.array([row["prompt_version"]], pa.string()),
            "generated_at": pa.array(
                [pd.Timestamp(row["generated_at"]).tz_convert("UTC")], pa.timestamp("us", tz="UTC")
            ),
        }
    )
    if table.column_names != list(WORK_ORDER_COLUMNS):
        raise RuntimeError("work order columns out of sync with schemas.WORK_ORDER_COLUMNS")
    out_dir = lake.layer_dirs(lake_root)["work_orders"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(row["generated_at"]).strftime("%Y%m%dT%H%M%S%f")
    target = out_dir / f"wo_{row['alert_id']}_{stamp}.parquet"
    tmp = out_dir / f".{target.name}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, target)
    return target


def alerts_needing_orders(
    ctx: ToolContext, origin: str | None = None, force: bool = False, limit: int | None = None
) -> list[str]:
    """Alerts with no work order, or whose alert changed after its latest order (or all: force)."""
    if "alerts" not in ctx.views():
        return []
    where, params = [], []
    if not force:
        where.append("(w.g IS NULL OR a.updated_at > w.g)")
    if origin:
        where.append("a.origin = ?")
        params.append(origin)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""SELECT a.alert_id FROM alerts a
              LEFT JOIN (SELECT alert_id, max(generated_at) AS g
                         FROM work_orders GROUP BY alert_id) w
              ON w.alert_id = a.alert_id {clause} ORDER BY a.start_ts, a.alert_id"""
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    return ctx.query(sql, params)["alert_id"].tolist()


def run_dispatch(
    ctx: ToolContext,
    llm: Any | None,
    origin: str | None = None,
    force: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    alert_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate (and unless ``dry_run``, write) work orders; returns counts."""
    ids = list(alert_ids) if alert_ids else alerts_needing_orders(ctx, origin, force, limit)
    counts = {
        "alerts": len(ids),
        "llm": 0,
        "template": 0,
        "fallbacks": 0,
        "written": 0,
        "failed": 0,
    }
    for alert_id in ids:
        try:
            gen = generate_work_order(ctx, alert_id, llm)
        except (LookupError, ValidationError, RuntimeError) as exc:
            counts["failed"] += 1
            log.error("alert %s: no work order (%s)", alert_id, exc)
            continue
        counts[gen.generator] += 1
        counts["fallbacks"] += int(any("fallback" in n for n in gen.notes))
        if not dry_run:
            write_work_order(gen, ctx.lake_root)
            counts["written"] += 1
    if counts["written"]:
        ctx.refresh()
    return counts


def main(argv: list[str] | None = None) -> int:
    """``make dispatch``: work orders for alerts that have none (template when no model is set)."""
    from leakwatch.copilot.llm import build_llm
    from leakwatch.copilot.settings import llm_settings

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--origin", choices=["batch", "stream"], help="only alerts of this origin")
    parser.add_argument("--alert-id", action="append", help="this alert only (repeatable)")
    parser.add_argument("--limit", type=int, help="at most this many alerts (default: config)")
    parser.add_argument("--force", action="store_true", help="regenerate every alert's order")
    parser.add_argument("--template", action="store_true", help="never call the model")
    parser.add_argument("--dry-run", action="store_true", help="generate but do not write")
    args = parser.parse_args(argv)
    ctx = ToolContext()
    settings = llm_settings()
    llm = None if args.template else build_llm(settings, ctx.cfg["llm"])
    ctx.llm = llm
    log.info("work orders by %s", "model " + str(settings.model) if llm else "template (no model)")
    if "alerts" not in ctx.views():
        log.info("no alerts in the lake yet (%s): nothing to do", ctx.lake_dir())
        return 0
    limit = args.limit or int(ctx.cfg["dispatch"]["max_per_run"])
    counts = run_dispatch(ctx, llm, args.origin, args.force, limit, args.dry_run, args.alert_id)
    log.info("work orders: %s", json.dumps(jsonable(counts)))
    return 1 if counts["failed"] and not (counts["llm"] or counts["template"]) else 0


if __name__ == "__main__":
    sys.exit(main())
