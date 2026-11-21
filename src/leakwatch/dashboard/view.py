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


def number(value: Any) -> float | None:
    """A finite float, or None for anything else (None, NaN, text, booleans)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def sensor_id(name: Any) -> str:
    """``pressure__n469`` or ``pressure__n469__residual`` -> ``n469``; a plain id stays as it is."""
    parts = str(name).split("__")
    return parts[1] if len(parts) > 1 else parts[0]


def seconds_between(later: Any, earlier: Any) -> int | None:
    """Whole seconds from ``earlier`` to ``later``; None if one is missing or the order is wrong."""
    a, b = to_utc(later), to_utc(earlier)
    if a is None or b is None or a < b:
        return None
    return round((a - b).total_seconds())


def drops_view(
    window_end: Any,
    residuals: Mapping[str, Any],
    floor_m: float,
    tz: str,
    label_sensors: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Pressure drop per sensor in one window: metres below the usual level, 0 under the floor.

    ``residuals`` maps sensor id -> window mean minus the hour-of-week baseline (negative = below
    the usual level). A missing value draws no ring. ``label_sensors`` (the alert's top sensors)
    get a text label on the map while their drop is shown.
    """
    end = to_utc(window_end)
    if end is None:
        return None
    values: dict[str, float] = {}
    for sensor, residual in residuals.items():
        r = number(residual)
        drop = 0.0 if r is None else max(0.0, -r)
        values[str(sensor)] = round(drop, 2) if drop >= floor_m else 0.0
    labels = [
        {"sensor": s, "text": f"{s}  −{values[s]:.1f} m"}
        for s in label_sensors
        if values.get(s, 0.0) > 0
    ]
    return {
        "window_end": end.isoformat(),
        "time": fmt_time(end, tz, date=False),
        "floor_m": floor_m,
        "values": values,
        "shown": sum(1 for v in values.values() if v > 0),
        "labels": labels,
    }


def match_leak(
    alert: Mapping[str, Any], leaks: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """The leak an alert belongs to: the newest leak that began at or before the alert did."""
    start = to_utc(alert.get("start_ts"))
    if start is None:
        return None
    before = [
        (began, leak)
        for leak in leaks
        if (began := to_utc(leak.get("start_ts"))) is not None and began <= start
    ]
    return max(before, key=lambda pair: pair[0])[1] if before else None


def incident_view(
    alert: Mapping[str, Any],
    leaks: Sequence[Mapping[str, Any]],
    versions: Sequence[Mapping[str, Any]],
    tz: str,
) -> dict[str, Any]:
    """The open alert, shaped for rail steps 2 (alert) and 3 (located) and for the map.

    ``versions`` are this alert's rows in ``alert_events`` (``updated_at``, ``located``: whether
    that version had localizer candidates). Seconds "after the click" are measured from the
    matching leak's ``requested_at``; candidates are given by rank only.
    """
    leak = match_leak(alert, leaks)
    start = to_utc(alert.get("start_ts"))
    began = None if leak is None else to_utc(leak.get("start_ts"))
    asked = None if leak is None else leak.get("requested_at")
    sensors = []
    for item in alert.get("top_sensors") or []:
        if isinstance(item, Mapping) and item.get("sensor_id"):
            share = number(item.get("contribution"))
            sensors.append(
                {
                    "id": sensor_id(item["sensor_id"]),
                    "share_pct": None if share is None else round(share * 100),
                }
            )
    sensors = sensors[:3]
    candidates = [
        str(c["node_id"])
        for c in alert.get("gnn_top5") or []
        if isinstance(c, Mapping) and c.get("node_id")
    ][:5]
    located = [t for v in versions if v.get("located") and (t := to_utc(v.get("updated_at")))]
    located_at = min(located) if located else (alert.get("updated_at") if candidates else None)
    loss = number(alert.get("est_loss_m3h"))
    return {
        "alert_id": str(alert.get("alert_id")),
        "area": alert.get("area"),
        "status": alert.get("status"),
        "start_ts": None if start is None else start.isoformat(),
        "started": None if start is None else fmt_time(start, tz, date=False),
        "leak_id": None if leak is None else leak.get("leak_id"),
        "after_click_s": seconds_between(alert.get("created_at"), asked),
        "after_leak_min": None
        if start is None or began is None
        else round((start - began).total_seconds() / 60),
        "ratio": score_ratio(alert),
        "sensors": sensors,
        "candidates": candidates,
        "located_after_s": seconds_between(located_at, asked) if candidates else None,
        "loss_now_m3h": None if loss is None else round(loss, 1),
        "focus": list(candidates),  # the camera frames these (the sensors may be far away)
    }


def newest(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any] | None:
    """The row with the latest timestamp in ``key`` (rows without one are skipped)."""
    dated = [(t, r) for r in rows if (t := to_utc(r.get(key))) is not None]
    return max(dated, key=lambda pair: pair[0])[1] if dated else None


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
        "incident": None,
        "drops": None,
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
    versions: Sequence[Mapping[str, Any]] = (),
    residuals: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Everything the live page shows, from the runner state and the newest lake rows.

    Args:
        state: the runner's ``state.json``.
        alerts: current alerts (``alerts`` view, newest version of each).
        now: wall-clock time of this snapshot (UTC).
        cfg: configuration (``project``, ``dashboard``, ``demo``).
        n_pressure: number of pressure sensors (for the idle banner).
        lake_error: sentence for the screen when the lake could not be read.
        failed_leaks: leaks that vanished from the state without starting.
        versions: ``alert_events`` rows (``alert_id``, ``updated_at``, ``located``).
        residuals: ``features_stream`` windows, oldest first: ``window_end`` + one residual per
            sensor id.

    """
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
        # the lake still holds the previous replay; the runner wipes it as it starts
        alerts, versions, residuals = [], [], []
    open_alerts = [a for a in alerts if a.get("status") == "open"]
    current = newest(open_alerts, "start_ts")
    incident = None
    if current is not None:
        mine = [v for v in versions if v.get("alert_id") == current.get("alert_id")]
        incident = incident_view(current, state.get("leaks") or [], mine, tz)
    last = residuals[-1] if residuals else None
    drops = None
    if last is not None:
        drops = drops_view(
            last.get("window_end"),
            {k: v for k, v in last.items() if k != "window_end"},
            float(dash["ring_floor_m"]),
            tz,
            [s["id"] for s in incident["sensors"]] if incident else (),
        )
    return {
        "runner": runner,
        "clock": clock_view(state, cfg["demo"], tz) if state else None,
        "banner": banner_view(runner, open_alerts, n_pressure, lake_error, tz),
        "open_alerts": len(open_alerts),
        "leaks": leaks_view(state, failed_leaks, tz),
        "incident": incident,
        "drops": drops,
        "error": lake_error,
    }
