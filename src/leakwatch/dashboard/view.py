"""Shaping for the control-room screen (D-052): runner state and lake rows -> what the page shows.

Pure functions, no I/O. ``api.py`` reads ``state.json`` and the lake and passes plain values in;
the page renders the result as it is, so the unit tests of this module cover every string and
number on screen. Inputs are UTC; times are shown in ``project.display_timezone``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

STATUS_CODES = {
    "starting": "starting",
    "warming up": "warming",
    "running": "running",
    "finished": "finished",
}
STATUS_TEXT = {
    "starting": ("Starting", "Kafka replay and Spark are starting."),
    "warming": ("Warming up", "The first micro-batches are being scored."),
    "running": ("Running", ""),
    "finished": ("Replay finished", "The replayed week is over. Restart the replay to run again."),
    "unknown": ("Status unknown", "The runner reported a status this screen does not know."),
}


def to_utc(value: Any) -> datetime | None:
    """A timestamp (ISO string, ``datetime`` or pandas ``Timestamp``) as an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace(" ", "T", 1)
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value != value:  # NaT is a datetime unequal to itself
        return None
    convert = getattr(value, "to_pydatetime", None)  # pandas Timestamp -> datetime
    if callable(convert):
        value = convert()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def fmt_time(ts: datetime, tz: str, *, date: bool = True) -> str:
    """``Thu 22 Nov, 12:15 IST`` (or ``12:15 IST``) in the display time zone."""
    local = ts.astimezone(ZoneInfo(tz))
    clock = f"{local:%H:%M} {local.tzname()}"
    return f"{local:%a} {local.day} {local:%b}, {clock}" if date else clock


def fmt_duration(seconds: float) -> str:
    """``45 s``, ``1 min 05 s`` or ``2 h 10 min``."""
    total = max(0, round(seconds))
    if total < 60:
        return f"{total} s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def runner_view(
    state: Mapping[str, Any],
    now: datetime,
    stale_after_s: float,
    starting_stale_after_s: float | None = None,
    busy: bool = False,
) -> dict[str, Any]:
    """Status of the demo runner: ``code`` drives the page, ``label`` and ``detail`` are shown.

    The runner rewrites its state file several times a second while it replays, but not while
    it starts Spark, so "starting" is allowed a longer silence (``starting_stale_after_s``).
    The same holds while it simulates a requested leak (``busy``): EPANET can hold it up.
    """
    if not state:
        return {
            "code": "offline",
            "label": "Demo runner is not running",
            "detail": "Start it with `make demo`.",
            "live": False,
        }
    code = STATUS_CODES.get(str(state.get("status", "")), "unknown")
    updated = to_utc(state.get("updated_at"))
    age = None if updated is None else (now - updated).total_seconds()
    limit = stale_after_s
    if (code == "starting" or busy) and starting_stale_after_s is not None:
        limit = starting_stale_after_s
    if code != "finished" and age is not None and age > limit:
        return {
            "code": "stale",
            "label": "Runner not responding",
            "detail": f"Last update {fmt_duration(age)} ago. Check the `make demo` terminal.",
            "live": False,
        }
    label, detail = STATUS_TEXT[code]
    return {"code": code, "label": label, "detail": detail, "live": code == "running"}


def clock_view(
    state: Mapping[str, Any], demo_cfg: Mapping[str, Any], tz: str
) -> dict[str, Any] | None:
    """Simulated time, replay speed and position in the replay (None before it starts)."""
    sim = to_utc(state.get("sim_ts"))
    if sim is None:
        return None
    start = to_utc(str(state.get("background_start") or demo_cfg["background_start"]))
    total = timedelta(hours=float(demo_cfg["background_hours"]))
    speed = float(state.get("speed") or 0.0)
    out: dict[str, Any] = {
        "sim_ts": sim.isoformat(),
        "sim_time": fmt_time(sim, tz),
        "speed": f"{speed:,.0f}×" if speed > 0 else None,
        "replay_fraction": None,
        "replay_left": None,
    }
    if start is not None and total.total_seconds() > 0:
        done = (sim - start) / total
        out["replay_fraction"] = round(min(1.0, max(0.0, done)), 4)
        if speed > 0:
            left_s = max(0.0, (start + total - sim).total_seconds()) / speed
            out["replay_left"] = fmt_duration(left_s)
    return out


def score_ratio(alert: Mapping[str, Any]) -> int | None:
    """How many times the alert threshold the peak score reached (whole number, for prose)."""
    score, threshold = alert.get("kmeans_score_max"), alert.get("threshold")
    if not isinstance(score, int | float) or not isinstance(threshold, int | float):
        return None
    if math.isnan(score) or math.isnan(threshold) or threshold <= 0:
        return None
    return round(score / threshold)


def banner_view(
    runner: Mapping[str, Any],
    open_alerts: Sequence[Mapping[str, Any]],
    n_pressure: int | None,
    lake_error: str | None,
    tz: str,
) -> dict[str, str]:
    """The one-line state over the map. Never says "no alerts" when the screen cannot know."""
    code = runner["code"]
    if code in ("offline", "stale"):
        return {"kind": "unknown", "title": runner["label"], "sub": runner["detail"]}
    if lake_error:
        return {"kind": "unknown", "title": "Alerts cannot be read", "sub": lake_error}
    if open_alerts:
        areas = sorted({str(a["area"]) for a in open_alerts if a.get("area")})
        first = min(
            open_alerts, key=lambda a: to_utc(a.get("start_ts")) or datetime.max.replace(tzinfo=UTC)
        )
        since = to_utc(first.get("start_ts"))
        ratio = score_ratio(first)
        parts = [] if since is None else [f"since {fmt_time(since, tz)}"]
        if ratio is not None:
            parts.append(f"score {ratio} times the threshold")
        title = (
            f"Alert in Area {areas[0]}"
            if len(open_alerts) == 1 and areas
            else f"{len(open_alerts)} open alerts"
            + (f" in Area {', '.join(areas)}" if areas else "")
        )
        return {"kind": "alert", "title": title, "sub": ", ".join(parts)}
    if code in ("starting", "warming"):
        return {"kind": "waiting", "title": runner["label"], "sub": runner["detail"]}
    if code == "finished":
        return {"kind": "waiting", "title": runner["label"], "sub": runner["detail"]}
    if code == "unknown":
        return {"kind": "unknown", "title": runner["label"], "sub": runner["detail"]}
    watching = "" if n_pressure is None else f"Watching {n_pressure} pressure sensors"
    return {"kind": "normal", "title": "No open alerts", "sub": watching}


def leak_view(leak: Mapping[str, Any], tz: str) -> dict[str, Any]:
    """One requested leak: where, how large, when it began (simulated) and when it was asked for."""
    start = to_utc(leak.get("start_ts"))
    asked = to_utc(leak.get("requested_at"))
    diameter, flow = leak.get("diameter_m"), leak.get("est_flow_m3h")
    return {
        "leak_id": leak.get("leak_id"),
        "node_id": str(leak.get("node_id")),
        "diameter_mm": round(float(diameter) * 1000) if isinstance(diameter, int | float) else None,
        "start_ts": None if start is None else start.isoformat(),
        "started": None if start is None else fmt_time(start, tz, date=False),
        "requested_at": None if asked is None else asked.isoformat(),
        "flow_m3h": round(float(flow), 1) if isinstance(flow, int | float) else None,
    }


def leaks_view(
    state: Mapping[str, Any], failed: Sequence[Mapping[str, Any]], tz: str
) -> dict[str, list[dict[str, Any]]]:
    """Requested leaks: being simulated, running, and those that could not be simulated."""
    failed_ids = {str(f.get("leak_id")) for f in failed}
    pending = [
        p for p in state.get("pending_leaks") or [] if str(p.get("leak_id")) not in failed_ids
    ]
    return {
        "pending": [leak_view(p, tz) for p in pending],
        "active": [leak_view(lk, tz) for lk in state.get("leaks") or []],
        "failed": [leak_view(f, tz) for f in failed],
    }


def unavailable_view(now: datetime) -> dict[str, Any]:
    """What the page shows when the server could not build the live view at all."""
    title = "Screen data unavailable"
    return {
        "runner": {"code": "error", "label": title, "detail": "", "live": False},
        "clock": None,
        "banner": {
            "kind": "unknown",
            "title": title,
            "sub": "The server could not build the live view. See its log.",
        },
        "open_alerts": 0,
        "leaks": {"pending": [], "active": [], "failed": []},
        "error": "snapshot failed",
        "generated_at": now,
    }


def live_view(
    *,
    state: Mapping[str, Any],
    alerts: Sequence[Mapping[str, Any]],
    now: datetime,
    cfg: Mapping[str, Any],
    n_pressure: int | None = None,
    lake_error: str | None = None,
    failed_leaks: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Everything the live page shows, from the runner state and the newest alerts."""
    tz = str(cfg["project"]["display_timezone"])
    dash = cfg["dashboard"]
    runner = runner_view(
        state,
        now,
        float(dash["stale_after_s"]),
        float(dash["starting_stale_after_s"]),
        busy=bool(state.get("pending_leaks")),
    )
    if runner["code"] == "starting":
        alerts = []  # the lake still holds the previous replay; the runner wipes it as it starts
    open_alerts = [a for a in alerts if a.get("status") == "open"]
    return {
        "runner": runner,
        "clock": clock_view(state, cfg["demo"], tz) if state else None,
        "banner": banner_view(runner, open_alerts, n_pressure, lake_error, tz),
        "open_alerts": len(open_alerts),
        "leaks": leaks_view(state, failed_leaks, tz),
        "error": lake_error,
    }
