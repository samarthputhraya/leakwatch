"""The copilot's tools (PRD §9.2): functions with JSON schemas over the lake (DuckDB, read-only).

Every tool takes ``(ctx, arguments)`` and returns a JSON-ready dict; errors come back as
``{"error": …}`` so the model can correct itself. Timestamps are written ``YYYY-MM-DD HH:MM UTC``.

The DuckDB connection is the lake's read views (``common.lake.connect``) plus a sandbox: external
access is disabled outside the lake directory and the configuration is locked, so even a query
that slipped past ``sql_guard`` could not read other files, attach databases or load extensions.

Chart spec (from ``get_sensor_series``; given to the UI, never sent to the model)::

    {"kind": "line", "title": str, "x_label": str, "y_label": str, "unit": str,
     "series": [{"name": str, "x": [ISO-8601 UTC], "y": [float | None]}],
     "markers": [{"x": ISO-8601 UTC, "label": str}]}
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import duckdb
import numpy as np
import pandas as pd

from leakwatch.common import lake, paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.schemas import AREAS, SENSOR_TYPES, UNITS
from leakwatch.copilot.guardrails import NOT_COVERED_TEXT
from leakwatch.copilot.network_index import NetworkIndex, load_network_index, plan_pipes

log = get_logger("leakwatch.copilot.tools")

#: What each lake view holds (also the schema text in the v3 prompt).
VIEW_DESCRIPTIONS: dict[str, str] = {
    "alerts": (
        "current alerts, one row per alert (latest version) with the latest work order. "
        "start_ts = when the alert fired; end_ts = null while open; status open/closed/confirmed/"
        "false_alarm; kmeans_score_max vs threshold; top_sensors = list of {sensor_id, "
        "contribution} where sensor_id is a series id such as pressure__n54; area A/B/C (area of "
        "the top sensor); gnn_top5 = list of {node_id, prob} or null; est_loss_m3h = estimated "
        "excess inflow (m3/h, an estimate); windows_above; last_window_end; origin batch|stream"
    ),
    "alert_events": "every version of every alert (append-only); prefer `alerts`",
    "work_orders": "work orders written by the copilot (generator llm|template)",
    "scores": (
        "K-Means anomaly score per 1-hour window: score, threshold, above (score > threshold), "
        "cluster, top_sensor, model, origin batch|stream"
    ),
    "features": (
        "batch feature table: one row per 1-hour window sliding every 15 minutes; columns "
        "<series>__<stat> (series like pressure__n54, flow__p227, level__T1, amr_sum__C; stats "
        "mean, std, min, max, slope per hour, residual = value minus hour-of-week baseline). "
        'Quote the partition column "window" (reserved word).'
    ),
    "features_stream": "same columns as features, computed live by the streaming job",
    "baselines": (
        "hour-of-week median per series from normal 2018 data: series_id, hour_of_week "
        "(0 = Monday 00:00 UTC), baseline, n_readings"
    ),
    "clean_readings": (
        "sensor readings after cleaning, long format, 5-minute steps: ts, sensor_id (n54, p227, "
        "PUMP_1, T1, AMR node ids), sensor_type (pressure m | flow m3/h | level m | amr L/h), "
        "value, "
        "unit, source (battledim|sim), year, date ('YYYY-MM-DD', partition: filter on it), "
        "filled, stuck, out_of_range. Prefer this table for sensor values"
    ),
    "raw_readings": "readings exactly as received from Kafka (same layout, no filled/stuck flags)",
}
VIEW_ORDER: tuple[str, ...] = tuple(VIEW_DESCRIPTIONS)
_FEATURE_COL = re.compile(r"^(?P<series>.+)__(?P<stat>mean|std|min|max|slope|residual)$")
_TYPE_ORDER = {t: i for i, t in enumerate(("pressure", "flow", "level", "amr"))}


class QueryTimeoutError(RuntimeError):
    """A query ran longer than ``copilot.sql.timeout_s`` and was interrupted."""


# ------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------


def fmt_ts(value: Any) -> str | None:
    """``2019-03-14 08:15 UTC`` (seconds shown only when not zero)."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if ts.second or ts.microsecond:
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def jsonable(value: Any) -> Any:
    """Make a query value JSON-ready: timestamps as text, floats to 6 significant digits."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp | datetime):
        return fmt_ts(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.6g}")
    if isinstance(value, bytes):
        return value.hex()
    if value is pd.NaT:
        return None
    return value


def parse_ts(value: Any) -> pd.Timestamp:
    """A timestamp from text or datetime; naive values are taken as UTC."""
    ts = pd.Timestamp(str(value).replace(" UTC", "").strip() if isinstance(value, str) else value)
    if pd.isna(ts):
        raise ValueError(f"not a timestamp: {value!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _hours(start: Any, end: Any) -> float | None:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None
    return max(0.0, (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 3600.0)


# ------------------------------------------------------------------------------------------
# Context
# ------------------------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Shared state of the tools: config, the DuckDB connection, network, retriever, clock."""

    full_cfg: dict[str, Any] = field(default_factory=load_config)
    lake_root: Path | None = None
    clock: pd.Timestamp | None = None
    retriever: Any | None = None
    network: NetworkIndex | None = None
    llm: Any | None = None  # LLMClient for LLM-assisted complaint extraction
    sql_hints: bool = True  # self-correction hints on SQL errors (prompt v3)
    complaints: list[dict[str, Any]] = field(default_factory=list)
    _con: duckdb.DuckDBPyConnection | None = field(default=None, repr=False)
    _con_at: float = field(default=0.0, repr=False)
    _views: set[str] = field(default_factory=set, repr=False)
    _clock_cache: pd.Timestamp | None = field(default=None, repr=False)
    _schema_cache: dict[str, Any] | None = field(default=None, repr=False)
    _network_tried: bool = field(default=False, repr=False)
    _retriever_error: str | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def cfg(self) -> dict[str, Any]:
        """The ``copilot`` config section."""
        return self.full_cfg["copilot"]

    def lake_dir(self) -> Path:
        """Root of the lake the tools read."""
        return (self.lake_root or paths.lake_dir()).resolve()

    # -- connection ------------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        root = self.lake_dir()
        con = lake.connect(lake_root=root)
        sql_cfg = self.cfg["sql"]
        allowed = root.as_posix().replace("'", "''")
        con.execute(f"SET memory_limit = '{sql_cfg['memory_limit']}'")
        con.execute(f"SET threads = {int(sql_cfg['threads'])}")
        con.execute(f"SET allowed_directories = ['{allowed}']")
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        return con

    def con(self) -> duckdb.DuckDBPyConnection:
        """The sandboxed connection, rebuilt after ``agent.connection_ttl_s``."""
        with self._lock:
            ttl = float(self.cfg["agent"].get("connection_ttl_s", 60))
            if self._con is None or time.monotonic() - self._con_at > ttl:
                self.refresh()
                self._con = self._connect()
                self._con_at = time.monotonic()
                self._views = {
                    r[0]
                    for r in self._con.execute(
                        "SELECT view_name FROM duckdb_views() WHERE NOT internal"
                    ).fetchall()
                }
            return self._con

    def refresh(self) -> None:
        """Drop the connection and cached lookups (new lake files become visible)."""
        with self._lock:
            if self._con is not None:
                self._con.close()
            self._con = None
            self._views = set()
            self._clock_cache = None
            self._schema_cache = None

    def views(self) -> set[str]:
        """Lake views that exist (only layers with Parquet files get a view)."""
        self.con()
        return set(self._views)

    def query(
        self, sql: str, params: Sequence[Any] | None = None, timeout_s: float | None = None
    ) -> pd.DataFrame:
        """Run ``sql`` (with ``?`` parameters); timestamps come back as UTC."""
        with self._lock:
            con = self.con()
            timeout = float(timeout_s if timeout_s is not None else self.cfg["sql"]["timeout_s"])
            timer = threading.Timer(timeout, con.interrupt) if timeout > 0 else None
            if timer is not None:
                timer.daemon = True
                timer.start()
            try:
                df = con.execute(sql, list(params or [])).df()
            except duckdb.InterruptException as exc:
                raise QueryTimeoutError(f"stopped after {timeout:g} s") from exc
            finally:
                if timer is not None:
                    timer.cancel()
        for col in df.columns:
            if isinstance(df[col].dtype, pd.DatetimeTZDtype):
                df[col] = df[col].dt.tz_convert("UTC")
        return df

    # -- clock, network, retriever ---------------------------------------------------------

    def data_clock(self) -> pd.Timestamp | None:
        """Return "now" in the data world: ``copilot.clock`` or the end of the feature tables."""
        if self.clock is not None:
            return parse_ts(self.clock)
        if self.cfg.get("clock"):
            return parse_ts(self.cfg["clock"])
        if self._clock_cache is not None:
            return self._clock_cache
        views = self.views()

        def latest(pairs: Sequence[tuple[str, str]]) -> list[pd.Timestamp]:
            found = []
            for view, col in pairs:
                if view in views:
                    value = self.query(f"SELECT max({col}) AS t FROM {view}")["t"].iloc[0]
                    if value is not None and not pd.isna(value):
                        found.append(pd.Timestamp(value))
            return found

        found = latest(
            [
                ("features", "window_end"),
                ("features_stream", "window_end"),
                ("alerts", "last_window_end"),
            ]
        ) or latest([("clean_readings", "ts"), ("raw_readings", "ts")])
        self._clock_cache = max(found) if found else None
        return self._clock_cache

    def get_network(self) -> NetworkIndex | None:
        """L-Town (loaded once); None if the network file is missing."""
        if self.network is None and not self._network_tried:
            self._network_tried = True
            try:
                self.network = load_network_index()
            except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
                log.warning("network model not available: %s", exc)
        return self.network

    def get_retriever(self) -> tuple[Any | None, str | None]:
        """The manual retriever, or None and the reason it is unavailable."""
        if self.retriever is None and self._retriever_error is None:
            try:
                from leakwatch.copilot.rag.index import Retriever

                self.retriever = Retriever.from_config(self.cfg)
            except Exception as exc:
                self._retriever_error = f"{type(exc).__name__}: {exc}"
                log.warning("manual index not available: %s", self._retriever_error)
        return self.retriever, self._retriever_error

    # -- schema ------------------------------------------------------------------------------

    def schema(self) -> dict[str, Any]:
        """Tables and columns (cached until :meth:`refresh`)."""
        if self._schema_cache is not None:
            return self._schema_cache
        views = self.views()
        tables = []
        for view in [v for v in VIEW_ORDER if v in views]:
            desc = self.query(f"DESCRIBE {view}")
            plain, series, stats = [], set(), set()
            for name, ctype in zip(desc["column_name"], desc["column_type"], strict=True):
                m = _FEATURE_COL.match(str(name))
                if m and view in ("features", "features_stream"):
                    series.add(m.group("series"))
                    stats.add(m.group("stat"))
                else:
                    plain.append(f"{name} {ctype}")
            entry: dict[str, Any] = {
                "name": view,
                "description": VIEW_DESCRIPTIONS[view],
                "columns": plain,
            }
            if series:
                entry["feature_columns"] = {
                    "pattern": "<series>__<stat> DOUBLE",
                    "stats": sorted(
                        stats, key=["mean", "std", "min", "max", "slope", "residual"].index
                    ),
                    "series": sorted(series),
                }
            tables.append(entry)
        clock = self.data_clock()
        self._schema_cache = {
            "tables": tables,
            "notes": [
                "All timestamps are TIMESTAMP WITH TIME ZONE in UTC; compare with literals like "
                "TIMESTAMPTZ '2019-03-01 00:00:00+00'.",
                f"Data ends at {fmt_ts(clock) or 'unknown'}: never use now() or current_date; "
                "'last week' means the 7 days before that time.",
                'Quote the column "window" (reserved word).',
                "Readings tables are partitioned by date ('YYYY-MM-DD' text) and sensor_type: "
                "filter on them to keep queries fast.",
                "top_sensors[1].sensor_id is the top sensor of an alert (lists are 1-based); use "
                "unnest(top_sensors) to expand.",
            ],
            "data_clock": fmt_ts(clock),
        }
        return self._schema_cache

    def schema_text(self) -> str:
        """Compact schema for the system prompt."""
        info = self.schema()
        lines = []
        for table in info["tables"]:
            lines.append(f"- {table['name']}: {table['description']}")
            lines.append(f"  columns: {', '.join(table['columns'])}")
            fc = table.get("feature_columns")
            if fc:
                lines.append(
                    f"  plus {len(fc['series']) * len(fc['stats'])} feature columns "
                    f"{fc['pattern']} with stats {', '.join(fc['stats'])} for series "
                    f"{', '.join(fc['series'])}"
                )
        if not info["tables"]:
            lines.append("- (the lake has no tables yet)")
        lines.extend(f"- note: {n}" for n in info["notes"])
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# Alerts helpers
# ------------------------------------------------------------------------------------------


def _end_ref(row: Mapping[str, Any]) -> tuple[Any, str]:
    if row.get("end_ts") is not None and not pd.isna(row.get("end_ts")):
        return row["end_ts"], "end_ts (alert closed)"
    return row.get("last_window_end"), "last_window_end (latest window above the threshold)"


def _structs(value: Any, keys: Sequence[str]) -> list[dict[str, Any]] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    items = list(value) if not isinstance(value, str) else json.loads(value)
    return [{k: item.get(k) if isinstance(item, Mapping) else None for k in keys} for item in items]


def fetch_alert(ctx: ToolContext, alert_id: str) -> dict[str, Any]:
    """The current version of one alert as a plain dict (id or unique prefix)."""
    if "alerts" not in ctx.views():
        raise LookupError("there are no alerts in the lake yet")
    ident = str(alert_id or "").strip().lower()
    min_chars = int(ctx.cfg["alerts"]["id_prefix_min_chars"])
    if len(ident) < min_chars:
        raise LookupError(f"give the alert id (at least {min_chars} characters)")
    df = ctx.query(
        "SELECT * FROM alerts WHERE lower(alert_id) LIKE ? || '%' ORDER BY start_ts LIMIT 6",
        [ident],
    )
    if df.empty:
        raise LookupError(f"no alert with id {alert_id!r}")
    exact = df[df["alert_id"].str.lower() == ident]
    if len(df) > 1 and exact.empty:
        raise LookupError(
            "several alerts start with that id: " + ", ".join(df["alert_id"].tolist())
        )
    row = (exact if not exact.empty else df).iloc[0].to_dict()
    row["top_sensors"] = _structs(row.get("top_sensors"), ("sensor_id", "contribution")) or []
    row["gnn_top5"] = _structs(row.get("gnn_top5"), ("node_id", "prob"))
    return row


def _sensor_parts(series_id: str) -> tuple[str | None, str]:
    kind, sep, ident = str(series_id).partition("__")
    return (kind, ident) if sep else (None, kind)


def alert_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """An alert as the tools report it."""
    end_ref, _ = _end_ref(row)
    hours = _hours(row.get("start_ts"), end_ref)
    score, thr = row.get("kmeans_score_max"), row.get("threshold")
    ratio = (score / thr) if score is not None and thr not in (None, 0) else None
    work_order = None
    if row.get("work_order_json"):
        try:
            work_order = json.loads(row["work_order_json"])
        except (TypeError, json.JSONDecodeError):
            work_order = row["work_order_json"]
    top = []
    for s in row.get("top_sensors") or []:
        kind, ident = _sensor_parts(str(s.get("sensor_id")))
        top.append(
            {
                "series_id": s.get("sensor_id"),
                "sensor_id": ident,
                "sensor_type": kind,
                "contribution": s.get("contribution"),
            }
        )
    out = {
        "alert_id": row.get("alert_id"),
        "status": row.get("status"),
        "area": row.get("area"),
        "origin": row.get("origin"),
        "source": row.get("source"),
        "scenario_id": row.get("scenario_id"),
        "start_ts": row.get("start_ts"),
        "end_ts": row.get("end_ts"),
        "last_window_end": row.get("last_window_end"),
        "duration_h": None if hours is None else round(hours, 2),
        "kmeans_score_max": score,
        "threshold": thr,
        "score_to_threshold": None if ratio is None else round(ratio, 2),
        "windows_above": row.get("windows_above"),
        "top_sensors": top,
        "gnn_top5": row.get("gnn_top5"),
        "lstm_score_max": row.get("lstm_score_max"),
        "est_loss_m3h": row.get("est_loss_m3h"),
        "work_order": work_order,
        "work_order_generator": row.get("work_order_generator"),
        "version": row.get("version"),
        "updated_at": row.get("updated_at"),
    }
    return jsonable(out)


# ------------------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------------------


def describe_schema(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """Tables and columns available in the lake."""
    return ctx.schema()


def _sql_hint(ctx: ToolContext, sql: str) -> str:
    schema = {t["name"]: t for t in ctx.schema()["tables"]}
    used = [name for name in schema if re.search(rf"\b{name}\b", sql, re.I)]
    parts = []
    for name in used or list(schema):
        cols = ", ".join(c.split(" ")[0] for c in schema[name]["columns"])
        extra = " + <series>__<stat> feature columns" if schema[name].get("feature_columns") else ""
        parts.append(f"{name}({cols}{extra})")
    return (
        "Read the error, fix the query and call run_sql again. Columns: "
        + "; ".join(parts)
        + '. Timestamps are UTC TIMESTAMPTZ; quote "window".'
    )


def run_sql(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """One read-only SELECT on the lake views (guarded, LIMIT enforced, timeout)."""
    from leakwatch.copilot.sql_guard import SQLGuardError, guard_sql

    query = str(args.get("query") or "")
    sql_cfg = ctx.cfg["sql"]
    try:
        guarded = guard_sql(
            query, ctx.views(), int(sql_cfg["max_rows"]), int(sql_cfg["max_query_chars"])
        )
    except SQLGuardError as exc:
        out: dict[str, Any] = {"error": f"rejected: {exc}", "sql": query}
        if ctx.sql_hints:
            out["hint"] = _sql_hint(ctx, query)
        return out
    try:
        df = ctx.query(guarded.sql)
    except QueryTimeoutError as exc:
        return {
            "error": f"query {exc}: filter on date/sensor_type partitions or aggregate first",
            "sql": guarded.sql,
        }
    except duckdb.Error as exc:
        out = {"error": str(exc).strip().splitlines()[0][:500], "sql": guarded.sql}
        if ctx.sql_hints:
            out["hint"] = _sql_hint(ctx, guarded.sql)
        return out
    rows = [[jsonable(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return {
        "sql": guarded.sql,
        "columns": [str(c) for c in df.columns],
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) >= guarded.max_rows,
    }


_ORDER_BY = {
    "newest": "start_ts DESC",
    "oldest": "start_ts ASC",
    "score": "kmeans_score_max DESC",
    "loss": "est_loss_m3h DESC NULLS LAST",
}


def list_alerts(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """Alerts filtered by status, area, origin and start time."""
    if "alerts" not in ctx.views():
        return {
            "total": 0,
            "returned": 0,
            "alerts": [],
            "note": "there are no alerts in the lake yet",
        }
    where, params, filters = [], [], {}
    status = args.get("status")
    if status:
        if status not in ("open", "closed", "confirmed", "false_alarm"):
            return {"error": "status must be open, closed, confirmed or false_alarm"}
        where.append("status = ?")
        params.append(status)
        filters["status"] = status
    area = args.get("area")
    if area:
        area = str(area).strip().upper()
        if area not in AREAS:
            return {"error": "area must be A, B or C"}
        where.append("area = ?")
        params.append(area)
        filters["area"] = area
    for key in ("origin", "source"):
        if args.get(key):
            where.append(f"{key} = ?")
            params.append(str(args[key]))
            filters[key] = str(args[key])
    try:
        if args.get("start"):
            start = parse_ts(args["start"])
            where.append("start_ts >= ?")
            params.append(start)
            filters["start"] = fmt_ts(start)
        if args.get("end"):
            end = parse_ts(args["end"])
            where.append("start_ts < ?")
            params.append(end)
            filters["end"] = fmt_ts(end)
    except ValueError as exc:
        return {"error": str(exc)}
    order = str(args.get("order") or "newest")
    if order not in _ORDER_BY:
        return {"error": f"order must be one of {', '.join(_ORDER_BY)}"}
    acfg = ctx.cfg["alerts"]
    limit = max(1, min(int(args.get("limit") or acfg["list_default"]), int(acfg["list_max"])))
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    total = int(ctx.query(f"SELECT count(*) AS n FROM alerts {clause}", params)["n"].iloc[0])
    df = ctx.query(
        f"""SELECT alert_id, status, area, origin, start_ts, end_ts, last_window_end,
                   kmeans_score_max, threshold, est_loss_m3h, windows_above, top_sensors,
                   gnn_top5 IS NOT NULL AS has_gnn, work_order_json IS NOT NULL AS has_work_order
            FROM alerts {clause} ORDER BY {_ORDER_BY[order]}, alert_id LIMIT ?""",
        [*params, limit],
    )
    alerts = []
    for row in df.to_dict("records"):
        end_ref, _ = _end_ref(row)
        top = _structs(row.get("top_sensors"), ("sensor_id", "contribution")) or []
        hours = _hours(row["start_ts"], end_ref)
        alerts.append(
            jsonable(
                {
                    "alert_id": row["alert_id"],
                    "status": row["status"],
                    "area": row["area"],
                    "origin": row["origin"],
                    "start_ts": row["start_ts"],
                    "end_ts": row["end_ts"],
                    "duration_h": None if hours is None else round(hours, 2),
                    "kmeans_score_max": row["kmeans_score_max"],
                    "threshold": row["threshold"],
                    "est_loss_m3h": row["est_loss_m3h"],
                    "windows_above": row["windows_above"],
                    "top_sensors": [s["sensor_id"] for s in top],
                    "has_gnn_candidates": bool(row["has_gnn"]),
                    "has_work_order": bool(row["has_work_order"]),
                }
            )
        )
    return {
        "total": total,
        "returned": len(alerts),
        "filters": filters,
        "order": order,
        "alerts": alerts,
    }


def get_alert(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """One alert with its explanation, candidates and work order."""
    try:
        return alert_view(fetch_alert(ctx, str(args.get("alert_id") or "")))
    except LookupError as exc:
        return {"error": str(exc)}


def _sensor_window(
    ctx: ToolContext, args: Mapping[str, Any]
) -> tuple[pd.Timestamp, pd.Timestamp, list[dict[str, Any]], list[str]]:
    scfg = ctx.cfg["series"]
    notes: list[str] = []
    markers: list[dict[str, Any]] = []
    if args.get("around_alert_id"):
        alert = fetch_alert(ctx, str(args["around_alert_id"]))
        margin = pd.Timedelta(hours=float(scfg["alert_margin_hours"]))
        end_ref, _ = _end_ref(alert)
        start = pd.Timestamp(alert["start_ts"]) - margin
        end = pd.Timestamp(end_ref if end_ref is not None else alert["start_ts"]) + margin
        markers.append({"x": pd.Timestamp(alert["start_ts"]).isoformat(), "label": "alert fired"})
        if alert.get("end_ts") is not None and not pd.isna(alert["end_ts"]):
            markers.append(
                {"x": pd.Timestamp(alert["end_ts"]).isoformat(), "label": "alert closed"}
            )
    else:
        default = pd.Timedelta(hours=float(scfg["default_hours"]))
        start = parse_ts(args["start"]) if args.get("start") else None
        end = parse_ts(args["end"]) if args.get("end") else None
        if end is None:
            end = start + default if start is not None else ctx.data_clock()
            if end is None:
                raise LookupError("no readings in the lake yet")
        if start is None:
            start = end - default
    if end <= start:
        raise ValueError("end must be after start")
    max_span = pd.Timedelta(days=float(scfg["max_days"]))
    if end - start > max_span:
        start = end - max_span
        notes.append(f"range shortened to the last {scfg['max_days']} days before the end")
    return start, end, markers, notes


def get_sensor_series(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """Readings of one sensor over a time range: summary for the model, full chart for the UI."""
    views = ctx.views()
    table = "clean_readings" if "clean_readings" in views else "raw_readings"
    if table not in views:
        return {"error": "there are no readings in the lake yet"}
    kind, sensor = _sensor_parts(str(args.get("sensor_id") or "").strip())
    if kind == "amr_sum":
        return {"error": "amr_sum__C is a feature series: query the features table with run_sql"}
    stype = args.get("sensor_type") or kind
    if stype and stype not in SENSOR_TYPES:
        return {"error": f"sensor_type must be one of {', '.join(SENSOR_TYPES)}"}
    try:
        start, end, markers, notes = _sensor_window(ctx, args)
    except (LookupError, ValueError) as exc:
        return {"error": str(exc)}
    sql = (
        f"SELECT ts, value, sensor_type, source FROM {table} WHERE sensor_id = ? "
        "AND date BETWEEN ? AND ? AND ts >= ? AND ts <= ?"
    )
    params: list[Any] = [sensor, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), start, end]
    if stype:
        sql += " AND sensor_type = ?"
        params.append(stype)
    if args.get("source"):
        sql += " AND source = ?"
        params.append(str(args["source"]))
    try:
        df = ctx.query(sql + " ORDER BY ts", params)
    except QueryTimeoutError as exc:
        return {"error": f"reading the series {exc}; ask for a shorter range"}
    if df.empty:
        return {
            "error": f"no readings for {sensor!r} between {fmt_ts(start)} and {fmt_ts(end)}. "
            "Pressure sensors look like n54, flows p227/p235/PUMP_1, the tank level T1."
        }
    types = sorted(df["sensor_type"].unique(), key=lambda t: _TYPE_ORDER.get(t, 9))
    if len(types) > 1:
        notes.append(f"{sensor} has {', '.join(types)} readings; showing {types[0]}")
        df = df[df["sensor_type"] == types[0]]
    sources = df["source"].unique().tolist()
    if len(sources) > 1:
        keep = "battledim" if "battledim" in sources else sources[0]
        notes.append(f"several sources ({', '.join(sources)}); showing {keep}")
        df = df[df["source"] == keep]
    stype = types[0]
    unit = UNITS[stype]
    series = df.set_index("ts")["value"].astype(float)
    baseline_mean = mean_residual = None
    baseline = None
    if "baselines" in views:
        base = ctx.query(
            "SELECT hour_of_week, baseline FROM baselines WHERE series_id = ?",
            [f"{stype}__{sensor}"],
        )
        if not base.empty:
            lookup = dict(zip(base["hour_of_week"].astype(int), base["baseline"], strict=True))
            how = series.index.dayofweek * 24 + series.index.hour
            baseline = pd.Series([lookup.get(int(h)) for h in how], index=series.index, dtype=float)
            baseline_mean = float(baseline.mean())
            mean_residual = float((series - baseline).mean())
    span_min = max(1.0, (series.index[-1] - series.index[0]).total_seconds() / 60.0)
    max_points = int(ctx.cfg["series"]["max_points"])
    chart_vals, chart_base = series, baseline
    resolution = 5
    if len(series) > max_points:
        resolution = int(math.ceil(span_min / max_points / 5.0) * 5)
        rule = f"{resolution}min"
        chart_vals = series.resample(rule).mean().dropna()
        chart_base = (
            baseline.resample(rule).mean().reindex(chart_vals.index)
            if baseline is not None
            else None
        )
    k = int(ctx.cfg["series"]["sample_points"])
    idx = sorted(
        set(np.linspace(0, len(chart_vals) - 1, min(k, len(chart_vals))).round().astype(int))
    )
    sample = [[fmt_ts(chart_vals.index[i]), jsonable(float(chart_vals.iloc[i]))] for i in idx]
    chart_series = [
        {
            "name": f"{stype} {sensor}",
            "x": [t.isoformat() for t in chart_vals.index],
            "y": [jsonable(v) for v in chart_vals],
        }
    ]
    if chart_base is not None:
        chart_series.append(
            {
                "name": "hour-of-week baseline",
                "x": [t.isoformat() for t in chart_base.index],
                "y": [jsonable(v) for v in chart_base],
            }
        )
    chart = {
        "kind": "line",
        "title": f"{stype.capitalize()} at {sensor} ({unit})",
        "x_label": "time (UTC)",
        "y_label": f"{stype} ({unit})",
        "unit": unit,
        "series": chart_series,
        "markers": markers,
    }
    out = {
        "sensor_id": sensor,
        "sensor_type": stype,
        "unit": unit,
        "table": table,
        "start": fmt_ts(start),
        "end": fmt_ts(end),
        "n_readings": len(series),
        "chart_resolution_minutes": resolution,
        "min": series.min(),
        "min_at": fmt_ts(series.idxmin()),
        "max": series.max(),
        "max_at": fmt_ts(series.idxmax()),
        "mean": series.mean(),
        "first": [fmt_ts(series.index[0]), series.iloc[0]],
        "last": [fmt_ts(series.index[-1]), series.iloc[-1]],
        "baseline_mean": baseline_mean,
        "mean_residual": mean_residual,
        "sample": sample,
        "notes": notes,
        "chart": "a line chart of this series is shown to the user",
    }
    result = jsonable(out)
    result["_chart"] = chart
    return result


def get_gnn_candidates(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """The localizer's top-5 candidate nodes for an alert (with attached pipes)."""
    try:
        alert = fetch_alert(ctx, str(args.get("alert_id") or ""))
    except LookupError as exc:
        return {"error": str(exc)}
    net = ctx.get_network()
    gnn = alert.get("gnn_top5")
    dcfg = ctx.cfg["dispatch"]
    if not gnn:
        out: dict[str, Any] = {
            "alert_id": alert["alert_id"],
            "available": False,
            "note": "the localizer has not written candidates (gnn_top5) for this alert",
        }
        pipes, basis = plan_pipes(alert, net, int(dcfg["max_pipes"]), int(dcfg["pipes_per_sensor"]))
        out["fallback_pipes"] = pipes
        out["fallback_basis"] = basis
        return jsonable(out)
    candidates = []
    for rank, cand in enumerate(sorted(gnn, key=lambda c: -float(c.get("prob") or 0.0)), start=1):
        node = str(cand.get("node_id"))
        candidates.append(
            {
                "rank": rank,
                "node_id": node,
                "prob": cand.get("prob"),
                "area": net.area_of(node) if net else None,
                "pipes": net.pipes_at(node) if net else [],
            }
        )
    return jsonable({"alert_id": alert["alert_id"], "available": True, "candidates": candidates})


def estimate_loss(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """Estimated water lost: est_loss_m3h × alert duration (clearly an estimate)."""
    try:
        alert = fetch_alert(ctx, str(args.get("alert_id") or ""))
    except LookupError as exc:
        return {"error": str(exc)}
    rate = alert.get("est_loss_m3h")
    if rate is None or pd.isna(rate):
        return {
            "alert_id": alert["alert_id"],
            "available": False,
            "note": "no loss estimate for this alert (no inflow residual during its windows)",
        }
    end_ref, end_kind = _end_ref(alert)
    hours = _hours(alert["start_ts"], end_ref) or 0.0
    inflow = ", ".join(ctx.full_cfg.get("bdt", {}).get("inflow_series", [])) or "inflow series"
    return jsonable(
        {
            "alert_id": alert["alert_id"],
            "available": True,
            "is_estimate": True,
            "est_loss_m3h": float(rate),
            "start_ts": alert["start_ts"],
            "end_ref_ts": end_ref,
            "end_ref": end_kind,
            "duration_h": round(hours, 2),
            "est_volume_m3": round(float(rate) * hours, 1),
            "method": (
                f"estimate: mean excess inflow ({inflow}) against the hour-of-week baseline over "
                "the alert's windows above the threshold, times the time from when the alert fired "
                "to its end; water lost before the alert fired is not counted"
            ),
        }
    )


def search_manual(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """Passages from the indexed manuals with document, section and page."""
    query = str(args.get("query") or "").strip()
    if not query:
        return {"error": "give a search query"}
    rag = ctx.cfg["rag"]
    retriever, reason = ctx.get_retriever()
    if retriever is None:
        return {
            "error": f"the manual index is not available ({reason}); run `make index`",
            "passages": [],
        }
    k = max(1, min(int(args.get("k") or rag["top_k"]), 8))
    hits = retriever.search(query, k=k, min_score=float(rag["min_score"]))
    limit = int(rag["passage_chars"])
    passages = [
        {
            "citation": h["citation"],
            "doc_id": h["doc_id"],
            "doc_short": h["doc_short"],
            "doc_title": h["doc_title"],
            "section": h["section"],
            "chapter": h.get("chapter"),
            "page": h.get("page"),
            "printed_page": h.get("printed_page"),
            "score": h["score"],
            "text": h["text"] if len(h["text"]) <= limit else h["text"][:limit] + " …",
        }
        for h in hits
    ]
    out: dict[str, Any] = {"query": query, "passages": passages}
    if not passages:
        out["note"] = f"no passage matched well enough. {NOT_COVERED_TEXT}"
    return out


def log_complaint(ctx: ToolContext, args: Mapping[str, Any]) -> dict[str, Any]:
    """Extract {area, issue_type, time_reported} and match the complaint to alerts."""
    from leakwatch.copilot.complaints import process_complaint

    text = str(args.get("text") or "").strip()
    if not text:
        return {"error": "give the complaint text"}
    try:
        reported = parse_ts(args["reported_at"]) if args.get("reported_at") else None
    except ValueError as exc:
        return {"error": str(exc)}
    return process_complaint(ctx, text, reported)


# ------------------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """A tool: name, description, JSON schema of its arguments, implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[ToolContext, Mapping[str, Any]], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        """OpenAI function-tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _params(properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_TS = {"type": "string", "description": "UTC time, e.g. 2019-03-14 08:00"}
_ALERT_ID = {"type": "string", "description": "alert id (or a unique prefix of 8+ characters)"}

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool(
            "describe_schema",
            "List the lake tables (DuckDB views) and their columns, with notes on querying them.",
            _params({}),
            describe_schema,
        ),
        Tool(
            "run_sql",
            "Run ONE read-only DuckDB SELECT on the lake views and return the rows and the "
            "executed SQL. The row count is capped; file, network and settings functions are "
            "rejected.",
            _params(
                {"query": {"type": "string", "description": "a single SELECT statement"}}, ["query"]
            ),
            run_sql,
        ),
        Tool(
            "list_alerts",
            "List alerts, newest first by default, with optional filters. Returns the total count "
            "and a page of alerts.",
            _params(
                {
                    "status": {
                        "type": "string",
                        "enum": ["open", "closed", "confirmed", "false_alarm"],
                    },
                    "area": {"type": "string", "enum": list(AREAS)},
                    "origin": {"type": "string", "enum": ["batch", "stream"]},
                    "source": {"type": "string", "enum": ["battledim", "sim"]},
                    "start": {**_TS, "description": "alerts that fired at or after this UTC time"},
                    "end": {**_TS, "description": "alerts that fired before this UTC time"},
                    "order": {"type": "string", "enum": list(_ORDER_BY)},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                }
            ),
            list_alerts,
        ),
        Tool(
            "get_alert",
            "Details of one alert: timing, score vs threshold, top sensors, area, localizer "
            "candidates, estimated loss rate and its work order.",
            _params({"alert_id": _ALERT_ID}, ["alert_id"]),
            get_alert,
        ),
        Tool(
            "get_sensor_series",
            "Readings of one sensor over a time range (or around an alert), with min/max/mean, "
            "the hour-of-week baseline and a sample; the user sees a chart.",
            _params(
                {
                    "sensor_id": {
                        "type": "string",
                        "description": "e.g. n54, p227, PUMP_1, T1 or pressure__n54",
                    },
                    "sensor_type": {"type": "string", "enum": list(SENSOR_TYPES)},
                    "start": _TS,
                    "end": _TS,
                    "around_alert_id": {
                        **_ALERT_ID,
                        "description": "show the series around this alert",
                    },
                    "source": {"type": "string", "enum": ["battledim", "sim"]},
                },
                ["sensor_id"],
            ),
            get_sensor_series,
        ),
        Tool(
            "get_gnn_candidates",
            "The localizer's top-5 candidate leak nodes for an alert, with the pipes at each node "
            "(or the fallback pipes nearest the top sensors when there are none).",
            _params({"alert_id": _ALERT_ID}, ["alert_id"]),
            get_gnn_candidates,
        ),
        Tool(
            "estimate_loss",
            "Estimated water lost for an alert: its excess-inflow rate (m3/h) times its duration. "
            "Always an estimate.",
            _params({"alert_id": _ALERT_ID}, ["alert_id"]),
            estimate_loss,
        ),
        Tool(
            "search_manual",
            "Search the indexed manuals (CPHEEO, LeakWatch SOP and methodology) and return "
            "passages with document, section, page and a citation string to copy.",
            _params(
                {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                ["query"],
            ),
            search_manual,
        ),
        Tool(
            "log_complaint",
            "Log a resident complaint: extract area, issue type and time, match it to alerts in "
            "that area and return the status reply.",
            _params(
                {
                    "text": {
                        "type": "string",
                        "description": "the complaint as the resident wrote it",
                    },
                    "reported_at": {**_TS, "description": "when the complaint was received (UTC)"},
                },
                ["text"],
            ),
            log_complaint,
        ),
    )
}


@dataclass
class ToolResult:
    """One executed tool call."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    chart: dict[str, Any] | None
    error: str | None
    elapsed_ms: float

    def content(self, max_chars: int = 12000) -> str:
        """The text sent back to the model (JSON, truncated beyond ``max_chars``)."""
        text = json.dumps(self.output, ensure_ascii=False, default=str)
        if len(text) > max_chars:
            text = text[:max_chars] + f" … [truncated: {len(text) - max_chars} more characters]"
        return text


class ToolRegistry:
    """Executes tool calls against one :class:`ToolContext`."""

    def __init__(self, ctx: ToolContext, names: Sequence[str] | None = None) -> None:
        """Expose ``names`` (default: all tools)."""
        self.ctx = ctx
        self.tools = {n: TOOLS[n] for n in (names or TOOLS)}

    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas in the OpenAI format."""
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: Mapping[str, Any] | None) -> ToolResult:
        """Run one tool; argument problems and exceptions become ``{"error": …}`` results."""
        args = dict(arguments or {})
        t0 = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            output: dict[str, Any] = {
                "error": f"unknown tool {name!r}; available: {', '.join(self.tools)}"
            }
        elif "__invalid_json__" in args:
            output = {"error": "the tool arguments were not valid JSON"}
        else:
            props = tool.parameters.get("properties", {})
            missing = [r for r in tool.parameters.get("required", []) if args.get(r) in (None, "")]
            unknown = [k for k in args if k not in props]
            if missing or unknown:
                output = {
                    "error": f"bad arguments (missing: {missing or 'none'}, "
                    f"unknown: {unknown or 'none'}); "
                    f"allowed: {', '.join(props) or 'none'}"
                }
            else:
                try:
                    output = tool.func(self.ctx, args)
                except Exception as exc:
                    log.exception("tool %s failed", name)
                    output = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        chart = output.pop("_chart", None) if isinstance(output, dict) else None
        error = output.get("error") if isinstance(output, dict) else None
        return ToolResult(
            name=name,
            arguments=args,
            output=output,
            chart=chart,
            error=str(error) if error else None,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
