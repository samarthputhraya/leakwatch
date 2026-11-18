"""Resident complaints (PRD §9.3 "Take complaints", user story 6). Synthetic: L-Town is fictional.

1. **Extract** ``{area, issue_type, time_reported}`` (plus when the problem started, if said):
   rules always run (no model needed); with a model configured, the model fills the fields as JSON
   (validated with pydantic) and the rules fill anything it leaves out or gets invalid.
2. **Match** against alerts in that area: an alert matches when it was active within
   ``copilot.complaints.match_window_hours`` of the problem, i.e.
   ``start_ts <= time_reported + window`` and ``active_until >= problem_start - window``, where
   ``active_until`` is ``end_ts`` (open alerts: still active) and ``problem_start`` is the onset or,
   if unknown, the time reported. Alerts active when the complaint arrived rank first, then by how
   close the alert start is to the problem start.
3. **Reply** with a status: ``matched`` (alert id), ``no_matching_alert`` (logged for review) or
   ``need_area`` (ask which area / street).

Times are UTC ("since this morning" = the configured morning hour, UTC).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from leakwatch.common.logging import get_logger
from leakwatch.common.schemas import Area

if TYPE_CHECKING:
    from leakwatch.copilot.network_index import NetworkIndex
    from leakwatch.copilot.tools import ToolContext

log = get_logger("leakwatch.copilot.complaints")

IssueType = Literal["no_water", "low_pressure", "visible_leak", "noise", "water_quality", "other"]
ISSUE_TYPES: tuple[str, ...] = (
    "no_water",
    "low_pressure",
    "visible_leak",
    "noise",
    "water_quality",
    "other",
)
MatchStatus = Literal["matched", "no_matching_alert", "need_area"]
SYNTHETIC_NOTE = (
    "L-Town is a fictional benchmark network: resident complaints are synthetic test cases."
)

_AREA = re.compile(r"\b(?:area|zone|sector|district|dma)\s*[-:#]?\s*([abc])\b", re.I)
_NODE = re.compile(r"\b(n\d{1,3}|T1)\b")
_PIPE = re.compile(r"\b(p\d{1,4})\b")
_LOCATION = re.compile(
    r"\b(?:near|at|on|outside|behind|opposite|next to|in front of)\s+([^.,;!?\"]{3,60})", re.I
)
_ISSUES: list[tuple[str, re.Pattern[str]]] = [
    (
        "visible_leak",
        re.compile(
            r"\bburst\w*|\bgush\w*|\bleak\w*|water (?:is )?(?:flowing|running|pouring|coming out|"
            r"spraying|bubbling|pooling)|water on the (?:road|street)|\bpuddles?\b|"
            r"\bpool(?:ing|s)? of water|\bflood\w*|wet patch|\bsinkhole|\bfountain",
            re.I,
        ),
    ),
    (
        "no_water",
        re.compile(
            r"\bno (?:water|supply|pressure)\b|\bzero pressure|water (?:is |has )?(?:not coming|"
            r"cut|stopped)|\b(?:taps?|pipes?) (?:are |is |have |has )?"
            r"(?:been |gone |run |running )?dry\b|\bdry taps?\b|\bwithout water\b|"
            r"\bnothing comes out",
            re.I,
        ),
    ),
    (
        "low_pressure",
        re.compile(
            r"\blow (?:water )?pressure|\bweak (?:flow|pressure|supply)|\btrickl\w*|"
            r"pressure (?:is )?(?:very )?(?:low|weak|poor)|\bpoor pressure|"
            r"\bslow(?:ly)? (?:flow|fill\w*)",
            re.I,
        ),
    ),
    ("noise", re.compile(r"\bhiss\w*|\bnois\w*|\bhumming|\bvibrat\w*|\bknocking|\bgurgl\w*", re.I)),
    (
        "water_quality",
        re.compile(
            r"\bdirty|\bbrown|\bmuddy|\bsmell\w*|\bodou?r|\bdiscolou?r\w*|\btaste|\bturbid|"
            r"\bcloudy|\byellow",
            re.I,
        ),
    ),
]

_COUNT = (
    r"(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"a couple of|a few|several)"
)
_UNIT = r"(minutes?|mins?|hours?|hrs?|days?)"
_DURATION = re.compile(
    rf"\b(?:for|since)\s+(?:the\s+)?(?:last\s+|past\s+)?(?:{_COUNT}\s+)?{_UNIT}\b", re.I
)
_AGO = re.compile(rf"\b{_COUNT}\s+{_UNIT}\s+ago\b", re.I)
_CLOCK = re.compile(
    r"\b(?:since|from)\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\w:])", re.I
)
_PART = re.compile(
    r"\b(?:(since|from)\s+)?(this|yesterday|last|early this|late last)?\s*"
    r"(morning|afternoon|evening|night|noon|midnight|yesterday|today)\b",
    re.I,
)
_WEEKDAY = re.compile(
    r"\bsince\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I
)
_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a few": 3,
    "several": 3,
}  # fmt: skip
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class ComplaintExtraction(BaseModel):
    """Structured fields of a complaint."""

    model_config = ConfigDict(extra="forbid")

    area: Area | None
    issue_type: IssueType
    time_reported: datetime
    onset: datetime | None
    onset_text: str | None
    location_text: str | None
    method: Literal["rules", "llm"]


class LLMComplaintFields(BaseModel):
    """What the model must return for LLM-assisted extraction."""

    model_config = ConfigDict(extra="ignore")

    area: Area | None = None
    issue_type: IssueType = "other"
    onset_text: str | None = None
    location_text: str | None = None


def _amount(text: str | None) -> int:
    text = (text or "").lower().strip()
    return int(text) if text.isdigit() else _WORDS.get(text, 1)


def _delta(count: str | None, unit: str) -> pd.Timedelta:
    n = _amount(count)
    unit = unit.lower()
    if unit.startswith("d"):
        return pd.Timedelta(days=n)
    if unit.startswith("h"):
        return pd.Timedelta(hours=n)
    return pd.Timedelta(minutes=n)


def parse_onset(
    text: str, reported: pd.Timestamp, cfg: Mapping[str, Any]
) -> tuple[pd.Timestamp | None, str | None]:
    """When the problem started, from phrases like "since this morning" or "2 hours ago"."""
    day = reported.normalize()
    for pattern in (_DURATION, _AGO):
        m = pattern.search(text)
        if m:
            return reported - _delta(m.group(1), m.group(2)), m.group(0)
    m = _CLOCK.search(text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if m.group(3):
            hour = hour % 12 + (12 if m.group(3).lower().startswith("p") else 0)
        if hour <= 23 and minute <= 59:
            onset = day + pd.Timedelta(hours=hour, minutes=minute)
            return (onset - pd.Timedelta(days=1) if onset > reported else onset), m.group(0)
    m = _WEEKDAY.search(text)
    if m:
        back = (reported.dayofweek - _DAYS.index(m.group(1).lower())) % 7
        return day - pd.Timedelta(days=back), m.group(0)
    for m in _PART.finditer(text):
        since, when, part = (
            (m.group(1) or "").lower(),
            (m.group(2) or "").lower(),
            m.group(3).lower(),
        )
        hours = {
            "morning": cfg.get("morning_hour", 6),
            "afternoon": cfg.get("afternoon_hour", 12),
            "evening": cfg.get("evening_hour", 18),
            "night": cfg.get("night_hour", 22),
            "noon": 12,
            "midnight": 0,
        }
        if part == "yesterday":
            return day - pd.Timedelta(days=1), m.group(0).strip()
        if part == "today":
            if since or when:
                return day, m.group(0).strip()
            continue
        if not (since or when):
            continue
        previous = when in ("yesterday", "last", "late last") or (
            part == "night" and when != "this"
        )
        onset = (
            day - pd.Timedelta(days=1 if previous else 0) + pd.Timedelta(hours=float(hours[part]))
        )
        if onset > reported:
            onset -= pd.Timedelta(days=1)
        return onset, m.group(0).strip()
    return None, None


def extract_rules(
    text: str,
    reported: pd.Timestamp,
    cfg: Mapping[str, Any],
    network: Callable[[], NetworkIndex | None] | None = None,
) -> ComplaintExtraction:
    """Rule-based extraction (works without a model)."""
    area = None
    m = _AREA.search(text)
    if m:
        area = m.group(1).upper()
    elif network is not None and (_NODE.search(text) or _PIPE.search(text)):
        net = network()
        if net is not None:
            node = _NODE.search(text)
            pipe = _PIPE.search(text)
            where = (
                node.group(1)
                if node
                else net.locate_series(f"flow__{pipe.group(1)}")
                if pipe
                else None
            )
            area = net.area_of(where) if where else None
    issue = next((name for name, pattern in _ISSUES if pattern.search(text)), "other")
    onset, onset_text = parse_onset(text, reported, cfg)
    loc = _LOCATION.search(text)
    return ComplaintExtraction(
        area=area,  # type: ignore[arg-type]
        issue_type=issue,  # type: ignore[arg-type]
        time_reported=reported.to_pydatetime(),
        onset=onset.to_pydatetime() if onset is not None else None,
        onset_text=onset_text,
        location_text=loc.group(1).strip() if loc else None,
        method="rules",
    )


def extract_llm(text: str, llm: Any, prompt: Mapping[str, str]) -> LLMComplaintFields | None:
    """Fields from the model (JSON), or None if the call or validation fails."""
    from leakwatch.copilot.llm import LLMError, json_object_from_text

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].replace("{text}", text)},
    ]
    try:
        resp = llm.chat(messages, json_mode=True)
        return LLMComplaintFields.model_validate(json_object_from_text(resp.content or ""))
    except (LLMError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        log.info("LLM complaint extraction failed (%s); using rules", type(exc).__name__)
        return None


def extract_complaint(
    text: str,
    reported: pd.Timestamp,
    cfg: Mapping[str, Any],
    llm: Any | None = None,
    network: Callable[[], NetworkIndex | None] | None = None,
) -> ComplaintExtraction:
    """Rules, refined by the model when one is given (rules fill what the model leaves out)."""
    rules = extract_rules(text, reported, cfg, network)
    if llm is None:
        return rules
    from leakwatch.copilot.prompts import load_named_prompt

    fields = extract_llm(text, llm, load_named_prompt("complaint"))
    if fields is None:
        return rules
    onset, onset_text = (None, None)
    if fields.onset_text:
        onset, onset_text = parse_onset(fields.onset_text, reported, cfg)
    if onset is None:
        onset, onset_text = (pd.Timestamp(rules.onset) if rules.onset else None), rules.onset_text
    return ComplaintExtraction(
        area=fields.area or rules.area,
        issue_type=fields.issue_type if fields.issue_type != "other" else rules.issue_type,
        time_reported=rules.time_reported,
        onset=onset.to_pydatetime() if onset is not None else None,
        onset_text=onset_text,
        location_text=fields.location_text or rules.location_text,
        method="llm",
    )


def match_complaint(
    ex: ComplaintExtraction, alerts: pd.DataFrame, window_hours: float
) -> dict[str, Any]:
    """Match an extracted complaint to alerts (columns alert_id, status, area, start_ts, end_ts)."""
    out: dict[str, Any] = {"status": "need_area", "area": ex.area, "window_hours": window_hours}
    if ex.area is None:
        return out
    reported = pd.Timestamp(ex.time_reported)
    problem_start = pd.Timestamp(ex.onset) if ex.onset else reported
    window = pd.Timedelta(hours=float(window_hours))
    df = alerts[alerts["area"] == ex.area].copy() if not alerts.empty else alerts
    if df.empty:
        out["status"] = "no_matching_alert"
        out["candidates"] = []
        return out
    far = pd.Timestamp.max.tz_localize("UTC")
    active_until = df["end_ts"].where(df["end_ts"].notna(), far)
    hit = df[
        (df["start_ts"] <= reported + window) & (active_until >= problem_start - window)
    ].copy()
    if hit.empty:
        out["status"] = "no_matching_alert"
        out["candidates"] = []
        return out
    hit["_active"] = (hit["start_ts"] <= reported) & (active_until.loc[hit.index] >= reported)
    hit["_gap"] = (hit["start_ts"] - problem_start).abs()
    hit = hit.sort_values(["_active", "_gap", "alert_id"], ascending=[False, True, True])
    best = hit.iloc[0]
    out.update(
        {
            "status": "matched",
            "alert_id": best["alert_id"],
            "alert_status": best["status"],
            "alert_start_ts": best["start_ts"],
            "alert_end_ts": best["end_ts"] if pd.notna(best["end_ts"]) else None,
            "candidates": hit["alert_id"].tolist(),
        }
    )
    return out


def reply_for(ex: ComplaintExtraction, match: Mapping[str, Any]) -> str:
    """Status message for the resident."""
    from leakwatch.copilot.tools import fmt_ts

    if match["status"] == "need_area":
        return (
            "Thank you for the report. Which area (A, B or C), street or landmark is affected? "
            "We need it to check the network."
        )
    if match["status"] == "no_matching_alert":
        return (
            f"Thank you. Your report for area {ex.area} is logged. There is no alert in that area "
            "around that time, so an operator will review it and may send a crew to check."
        )
    ref = str(match["alert_id"])[:8]
    if match.get("alert_end_ts") is None:
        return (
            f"Thank you. We are already working on a possible leak in area {ex.area}: an alert has "
            f"been open since {fmt_ts(match['alert_start_ts'])} and your report is linked to it "
            f"(reference {ref})."
        )
    return (
        f"Thank you. Your report is linked to an alert in area {ex.area} that ran from "
        f"{fmt_ts(match['alert_start_ts'])} to {fmt_ts(match['alert_end_ts'])} (reference {ref}). "
        "An operator will check whether the problem is still there."
    )


def area_alerts(ctx: ToolContext, area: str | None) -> pd.DataFrame:
    """Alerts of one area (empty frame if none or no alerts table)."""
    cols = ["alert_id", "status", "area", "start_ts", "end_ts"]
    if area is None or "alerts" not in ctx.views():
        return pd.DataFrame(columns=cols)
    return ctx.query(f"SELECT {', '.join(cols)} FROM alerts WHERE area = ?", [area])


def process_complaint(
    ctx: ToolContext, text: str, reported: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Extract, match and reply; the record is kept in ``ctx.complaints`` for the session."""
    from leakwatch.copilot.tools import jsonable

    ccfg = ctx.cfg["complaints"]
    when = reported if reported is not None else ctx.data_clock()
    if when is None:
        when = pd.Timestamp.now(tz="UTC").floor("min")
    llm = ctx.llm if ccfg.get("use_llm_extraction", True) else None
    ex = extract_complaint(text, pd.Timestamp(when), ccfg, llm=llm, network=ctx.get_network)
    match = match_complaint(ex, area_alerts(ctx, ex.area), float(ccfg["match_window_hours"]))
    record = {
        "synthetic_note": SYNTHETIC_NOTE,
        "text": text,
        "extracted": jsonable(ex.model_dump()),
        "match": jsonable(match),
        "reply": reply_for(ex, match),
    }
    ctx.complaints.append(record)
    return record
