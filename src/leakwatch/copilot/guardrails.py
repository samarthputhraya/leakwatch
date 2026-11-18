"""Answer guardrails (PRD §9.4). Pure functions on text: no model, no lake.

1. **No number without a tool result.** :func:`check_numbers` extracts every number in an answer
   and checks that each one appears in that turn's tool outputs (or in the question). Matching
   allows rounding to the precision the answer was written with, thousands separators, percent
   forms (0.45 ↔ 45 %), and the unit factors 1000 (m³ ↔ L), 24 (per hour ↔ per day) and 60
   (hours ↔ minutes). Dates (``YYYY-MM-DD``, ``15 March 2019``) and times (``08:15``) must appear
   as dates/times in the tool outputs. Ignored: years, list markers (``1.``), ranks (``top 5``,
   ``#2``, ``1st``), identifiers (``n54``, ``p227``, ``PUMP_1``, ``pressure__n54``, alert ids),
   bracketed citations (checked separately) and the configured trivial numbers (0, 1).
2. **Scope.** :func:`is_out_of_scope` is a cheap pre-filter for clear off-topic requests and prompt
   injection; :func:`is_refusal` recognises a refusal in a model answer.
3. **Citations.** :func:`check_citations`: an answer that used manual passages must cite one of
   them (``[doc, section, p. N]``) or say that the manuals do not cover the question.

The agent (``copilot.agent``) decides what to do with a failed check: regenerate once with a
correction message, then fall back to the tool output.
"""

from __future__ import annotations

import calendar
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The copilot's refusal (v3 prompts ask for exactly this sentence).
REFUSAL_TEXT = (
    "I can only help with LeakWatch questions: alerts and work orders, sensor data in the lake "
    "(with charts), leak-response procedures from the indexed manuals, and resident complaints "
    "about the water supply."
)
#: Said when no indexed passage supports an answer (PRD §9.4).
NOT_COVERED_TEXT = "The indexed manuals do not cover this question."

#: Unit/format factors between a tool value and a number written in an answer.
FACTORS: tuple[float, ...] = (1.0, 100.0, 0.01, 1000.0, 0.001, 24.0, 1 / 24, 60.0, 1 / 60)
_MAGNITUDE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "lakh": 1e5, "crore": 1e7}
_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

# Things that contain digits but are not quantities.
_URL = re.compile(r"https?://\S+")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_ID_PREFIX = re.compile(r"(?<![\w-])[0-9a-f][0-9a-f-]{5,35}(?![\w-])", re.I)
_HEX_ID = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{6,}\b", re.I)
_SERIES = re.compile(r"\b(?:pressure|flow|level|amr|amr_sum)__[\w-]+", re.I)
_DEVICE = re.compile(r"\b(?:PRV|PUMP|FCV|TCV)[-_ ]?\d+\b", re.I)
_CITATION = re.compile(r"\[[^\[\]]{0,300}\]")
_LIST_MARKER = re.compile(r"^\s*(?:[-*]\s*)?\(?\d{1,2}[.)]\s", re.M)

# Dates and times.
_ISO_DATE = re.compile(
    r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2}|\s?UTC)?"
)
_YEAR_MONTH = re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])\b(?!-\d)")
_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?(?:\s+(\d{{4}}))?\b", re.I
)
_MONTH_DAY = re.compile(
    rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?(?:\s+(\d{{4}}))?", re.I
)
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?(?:\s*([ap])\.?m\.?\b)?", re.I)

_NUMBER = re.compile(
    r"(?<![\w.,#§])"
    r"(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?!\.?\d)"
    r"(?P<ordinal>(?:st|nd|rd|th)\b)?"
    r"(?P<pct>\s*(?:%|percent\b|per\s?cent\b))?"
    r"(?P<mag>\s*(?:thousand|million|billion|lakh|crore)\b)?",
    re.I,
)
_RANK_BEFORE = re.compile(
    r"(?:\btop|\brank|\bstep|\bno\.|\bnumber|\boption|\bpriority)[\s-]*$", re.I
)


@dataclass(frozen=True)
class NumberMention:
    """A number as written in a text."""

    text: str
    value: float  # as written (percent sign not applied; magnitude word applied)
    tolerance: float  # how far a source value may be and still round to ``text``
    percent: bool = False


@dataclass
class Extracted:
    """Numbers, dates and times found in a text."""

    numbers: list[NumberMention] = field(default_factory=list)
    dates: set[str] = field(default_factory=set)  # YYYY-MM-DD
    month_days: set[str] = field(default_factory=set)  # MM-DD (dates written without a year)
    year_months: set[str] = field(default_factory=set)  # YYYY-MM written on its own
    times: set[str] = field(default_factory=set)  # HH:MM (24 h)
    date_mentions: list[str] = field(default_factory=list)  # as written (answers only)
    time_mentions: list[tuple[str, str]] = field(default_factory=list)  # (written, HH:MM)


def _parse_number(raw: str, pct: bool, mag: str | None) -> NumberMention | None:
    clean = raw.replace(",", "")
    try:
        value = float(clean)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    mantissa, _, exponent = clean.lower().partition("e")
    exp10 = int(exponent) if exponent else 0
    if "." in mantissa:
        decimals = len(mantissa.split(".", 1)[1])
        tol = 10.0 ** (-decimals)
    else:
        digits = mantissa.lstrip("+-")
        zeros = len(digits) - len(digits.rstrip("0")) if len(digits) > 1 else 0
        tol = 0.5 * 10.0**zeros if digits.strip("0") else 0.5
    tol *= 10.0**exp10
    scale = _MAGNITUDE.get(mag.strip().lower(), 1.0) if mag else 1.0
    return NumberMention(text=raw, value=abs(value) * scale, tolerance=tol * scale, percent=pct)


def _blank(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def extract(text: str, *, answer: bool = False, known_ids: Iterable[str] = ()) -> Extracted:
    """Find numbers, dates and times in ``text``.

    Args:
        text: an answer or a tool output.
        answer: apply the answer-only rules (skip list markers, ranks, ordinals, citations).
        known_ids: ids (UUIDs) from the tool outputs; their prefixes in ``text`` (such as
            ``11111111`` for ``11111111-aaaa-…``) are identifiers, not numbers.

    """
    out = Extracted()
    work = text or ""
    for pattern in (_URL, _UUID, _HEX_ID, _SERIES, _DEVICE):
        work = _blank(work, pattern)
    ids = [i.lower() for i in known_ids]
    if ids:
        work = _ID_PREFIX.sub(
            lambda m: (
                " " * len(m.group(0))
                if any(i.startswith(m.group(0).lower()) for i in ids)
                else m.group(0)
            ),
            work,
        )
    if answer:
        work = _blank(work, _CITATION)
        work = _LIST_MARKER.sub(lambda m: " " * len(m.group(0)), work)

    def take_iso(m: re.Match[str]) -> str:
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        out.dates.add(date)
        out.month_days.add(date[5:])
        out.date_mentions.append(m.group(0))
        if m.group(4) is not None:
            hhmm = f"{m.group(4)}:{m.group(5)}"
            out.times.add(hhmm)
            out.time_mentions.append((m.group(0), hhmm))
        return " " * len(m.group(0))

    work = _ISO_DATE.sub(take_iso, work)

    def take_year_month(m: re.Match[str]) -> str:
        out.year_months.add(f"{m.group(1)}-{m.group(2)}")
        out.date_mentions.append(m.group(0))
        return " " * len(m.group(0))

    work = _YEAR_MONTH.sub(take_year_month, work)

    def take_named(day: str, month: str, year: str | None, whole: str) -> str:
        mm = _MONTHS[month.lower().rstrip(".")]
        dd = int(day)
        if not 1 <= dd <= 31:
            return whole
        md = f"{mm:02d}-{dd:02d}"
        out.month_days.add(md)
        if year:
            out.dates.add(f"{year}-{md}")
        out.date_mentions.append(whole)
        return " " * len(whole)

    work = _DAY_MONTH.sub(
        lambda m: take_named(m.group(1), m.group(2), m.group(3), m.group(0)), work
    )
    work = _MONTH_DAY.sub(
        lambda m: take_named(m.group(2), m.group(1), m.group(3), m.group(0)), work
    )

    def take_time(m: re.Match[str]) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return m.group(0)
        if m.group(3):
            hour = hour % 12 + (12 if m.group(3).lower() == "p" else 0)
        hhmm = f"{hour:02d}:{minute:02d}"
        out.times.add(hhmm)
        out.time_mentions.append((m.group(0), hhmm))
        return " " * len(m.group(0))

    work = _TIME.sub(take_time, work)

    for m in _NUMBER.finditer(work):
        raw = m.group("num")
        if answer:
            if m.group("ordinal"):
                continue
            if _RANK_BEFORE.search(work[max(0, m.start() - 12) : m.start()]):
                continue
            plain = raw.lstrip("+-")
            if (
                plain.isdigit()
                and len(plain) == 4
                and 1900 <= int(plain) <= 2100
                and not m.group("pct")
            ):
                continue  # a year
        mention = _parse_number(raw, bool(m.group("pct")), m.group("mag"))
        if mention is not None:
            out.numbers.append(mention)
    if not answer:
        out.date_mentions.clear()
        out.time_mentions.clear()
    return out


def number_supported(mention: NumberMention, values: Iterable[float]) -> bool:
    """True if some source value, under one of :data:`FACTORS`, rounds to ``mention``."""
    target = mention.value
    tol = mention.tolerance + 1e-9 * max(1.0, target)
    for v in values:
        av = abs(v)
        for f in FACTORS:
            if abs(av * f - target) <= tol:
                return True
    return False


@dataclass
class NumberCheck:
    """Result of :func:`check_numbers`."""

    ok: bool
    checked: list[str]
    unsupported: list[str]


def check_numbers(
    answer: str,
    sources: Iterable[str],
    question: str = "",
    trivial: Sequence[float] = (0, 1),
) -> NumberCheck:
    """Check that every number, date and time in ``answer`` comes from ``sources``.

    Args:
        answer: the model's answer.
        sources: texts the model saw this turn (tool outputs as sent to it).
        question: the user's question (its numbers may be repeated).
        trivial: numbers that never need a source.

    """
    pool = Extracted()
    sources_list = list(sources)
    for text in [*sources_list, question]:
        found = extract(text)
        pool.numbers.extend(found.numbers)
        pool.dates |= found.dates
        pool.month_days |= found.month_days
        pool.year_months |= found.year_months
        pool.times |= found.times
    values = [n.value for n in pool.numbers]
    ids = {m.group(0) for text in sources_list for m in _UUID.finditer(text)}
    said = extract(answer, answer=True, known_ids=ids)
    checked: list[str] = []
    unsupported: list[str] = []
    for mention in said.numbers:
        checked.append(mention.text)
        if any(abs(mention.value - t) < 1e-12 for t in trivial):
            continue
        if not number_supported(mention, values):
            unsupported.append(mention.text + ("%" if mention.percent else ""))
    for written in said.date_mentions:
        checked.append(written)
        iso = _ISO_DATE.match(written)
        year_month = _YEAR_MONTH.fullmatch(written)
        if iso:
            ok = f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}" in pool.dates
        elif year_month:
            ok = any(d.startswith(written) for d in pool.dates) or written in pool.year_months
        else:
            sub = extract(written)
            ok = (
                bool(sub.dates & pool.dates)
                if sub.dates
                else bool(sub.month_days & pool.month_days)
            )
        if not ok:
            unsupported.append(written)
    for written, hhmm in said.time_mentions:
        checked.append(written)
        if hhmm not in pool.times and written not in unsupported:
            unsupported.append(written)
    return NumberCheck(ok=not unsupported, checked=checked, unsupported=unsupported)


def number_correction_message(unsupported: Sequence[str]) -> str:
    """Correction sent to the model when :func:`check_numbers` fails (PRD §9.4)."""
    listed = ", ".join(dict.fromkeys(unsupported))
    return (
        "Your answer contains numbers, dates or times that do not appear in this turn's tool "
        f"results: {listed}. Rewrite the answer using only values copied from tool results "
        "(dates as YYYY-MM-DD HH:MM UTC, as the tools return them). If you need a value you do not "
        "have, call a tool (for example run_sql) to compute it. Do not estimate or do arithmetic "
        "in your head."
    )


# ------------------------------------------------------------------------------------------
# Scope and refusals
# ------------------------------------------------------------------------------------------

_DOMAIN = re.compile(
    r"\b(?:water|leak\w*|pipe\w*|pressure|flows?|sensor\w*|alerts?|alarm\w*|areas?|zones?|"
    r"network|valves?|pumps?|tanks?|reservoirs?|meters?|amrs?|scada|burst\w*|complain\w*|supply|"
    r"manuals?|cpheeo|work\s?orders?|crews?|repair\w*|dig\w*|excavat\w*|loss\w*|lost|nrw|ufw|"
    r"audit\w*|l-?town|battledim|k-?means|anomal\w*|scores?|threshold\w*|gnn|locali[sz]\w*|"
    r"nodes?|junctions?|sql|tables?|data|lake|readings?|charts?|plot\w*|trends?|baselines?|"
    r"features?|windows?|dispatch\w*|priorit\w*|inspect\w*|demand|consumption|taps?|street|road|"
    r"hiss\w*|noise|dirty|outage|copilot|leakwatch|inflow|outflow|m3|litres?|liters?|operators?|"
    r"incidents?|maintenance|procedures?|sop|n\d+|p\d+|pump_\d+|t1)\b",
    re.I,
)
_OFF_TOPIC = re.compile(
    r"\b(?:poem|poetry|joke|story|song|lyrics|recipe|cook\w*|weather|forecast|stocks?|shares|"
    r"bitcoin|crypto\w*|football|cricket|movies?|films?|celebrit\w*|president|prime\s+minister|"
    r"capital\s+of|translate|homework|essay|horoscope|dating|world\s+cup|elections?|religion)\b",
    re.I,
)
_INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+|above\s+)*"
    r"(?:instructions|rules|prompts?)|system\s+prompt|developer\s+mode|jailbreak|"
    r"reveal\s+(?:your|the)\s+(?:prompt|instructions|api\s+key)|api\s+key|password",
    re.I,
)
_REFUSAL = re.compile(
    r"I can only help with|I can(?:no|')t help with|I(?: am|'m) (?:not able|unable) to "
    r"(?:help|answer|assist)|outside (?:of )?(?:my|the) scope|beyond (?:my|the) scope|"
    r"I can(?:no|')t assist|I only (?:answer|help with)|not (?:something|a question) I can "
    r"(?:help|answer)",
    re.I,
)


def is_out_of_scope(question: str) -> bool:
    """Pre-filter: prompt injection, or an off-topic request with no water-network words.

    Anything else goes to the model, which refuses with :data:`REFUSAL_TEXT` when needed.
    """
    q = question or ""
    if _INJECTION.search(q):
        return True
    return bool(_OFF_TOPIC.search(q)) and not _DOMAIN.search(q)


def is_refusal(answer: str) -> bool:
    """True if the answer declines the request."""
    return bool(_REFUSAL.search(answer or ""))


# ------------------------------------------------------------------------------------------
# Citations
# ------------------------------------------------------------------------------------------

_BRACKET = re.compile(r"\[([^\[\]]{3,300})\]")
_SECTION_NO = re.compile(r"(\d+(?:\.\d+)+)")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def find_citations(answer: str) -> list[str]:
    """Bracketed citations in an answer, e.g. ``[CPHEEO O&M Manual 2005, §15.10 …, p. 383]``."""
    return [m.group(1) for m in _BRACKET.finditer(answer or "")]


def section_number(section: str) -> str | None:
    """Leading section number of a heading (``15.10`` in ``§15.10 Leakage Detection``)."""
    m = _SECTION_NO.search(section or "")
    return m.group(1) if m else None


def section_cited(citation: str, number: str | None, title: str | None) -> bool:
    """True if ``citation`` names a section by its number (exactly, not 15.10 for 15.1) or title."""
    c = _norm(citation)
    if number and re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", c):
        return True
    return bool(title) and _norm(title) in c


def citation_matches(citation: str, passage: Mapping[str, Any]) -> bool:
    """True if ``citation`` names the passage's document and its section or page."""
    c = _norm(citation)
    if passage.get("citation") and _norm(str(passage["citation"])).strip("[]") == c:
        return True
    doc_names = [passage.get("doc_short"), passage.get("doc_title"), passage.get("doc_id")]
    if not any(name and _norm(str(name)) in c for name in doc_names):
        return False
    section = str(passage.get("section") or "")
    title = re.sub(r"^[§\s]*\d+(?:\.\d+)*\.?\s*", "", section).strip()
    if section_cited(c, section_number(section), title or None):
        return True
    page = passage.get("page")
    return page not in (None, "", "n/a") and bool(re.search(rf"\bp(?:age|\.)?\s*{page}\b", c))


@dataclass
class CitationCheck:
    """Result of :func:`check_citations`."""

    ok: bool
    cited: list[str]
    valid: list[str]
    not_covered: bool


def check_citations(answer: str, passages: Sequence[Mapping[str, Any]]) -> CitationCheck:
    """An answer that used manual passages must cite one of them or say they do not cover it."""
    cited = find_citations(answer)
    valid = [c for c in cited if any(citation_matches(c, p) for p in passages)]
    not_covered = _norm(NOT_COVERED_TEXT).rstrip(".") in _norm(answer) or bool(
        re.search(
            r"(?:manuals?|documents?)\s+(?:do(?:es)?\s+not|don't|doesn't)\s+cover", answer, re.I
        )
    )
    return CitationCheck(
        ok=bool(valid) or not_covered, cited=cited, valid=valid, not_covered=not_covered
    )


def citation_correction_message(passages: Sequence[Mapping[str, Any]]) -> str:
    """Correction sent when a manual-based answer has no valid citation."""
    examples = "; ".join(str(p.get("citation")) for p in passages[:3] if p.get("citation"))
    return (
        "Your answer used the manual search but does not cite a returned passage. Cite every "
        "statement from the manuals with the passage's citation exactly as given "
        f"(for example {examples}). If none of the passages answers the question, reply: "
        f"{NOT_COVERED_TEXT}"
    )
