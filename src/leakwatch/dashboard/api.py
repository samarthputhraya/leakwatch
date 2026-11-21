"""Control-room screen, server side (D-052): ``python -m leakwatch.dashboard.api``.

A FastAPI app that serves the page (``static/``, no build step) and its JSON:

- ``GET /api/live``: everything the live page shows, shaped by ``view.live_view``. One background
  thread rebuilds it every ``dashboard.snapshot_seconds`` from ``state.json`` and the demo lake
  (DuckDB); requests return the last snapshot, so any number of browser tabs cost the same.
- ``GET /api/network``: the map geometry (``geometry.py``), built once.
- ``POST /api/demo/inject | speed | restart``: the three ``demo.commands`` of the runner.
- ``GET /api/health``: for scripts that wait for the server.

Commands are accepted from the dashboard's own page only (``from_the_page``), and the app
answers only to the host names in ``dashboard.allowed_hosts``: another website open in the
same browser cannot restart the replay or start a leak.

DuckDB is used from the snapshot thread only. A lake that cannot be read is reported in the
snapshot (``error``), never as an empty result. Failures answer with a sentence for the screen,
never with a stack trace.
"""

from __future__ import annotations

import argparse
import math
import re
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.dashboard import view

log = get_logger("leakwatch.dashboard.api")
STATIC = Path(__file__).with_name("static")
LAYERS = ("alert_events", "work_orders", "features_stream", "scores")
#: Lake reads, each re-run only when the Parquet files of its layers change.
GROUPS = {
    "alerts": ("alert_events", "work_orders"),
    "residuals": ("features_stream",),
    "scores": ("scores",),
}
ALERT_COLUMNS = (
    "alert_id, origin, status, start_ts, end_ts, area, kmeans_score_max, threshold, "
    "est_loss_m3h, windows_above, last_window_end, top_sensors, gnn_top5, work_order_json, "
    "work_order_generator, created_at, updated_at"
)
VERSION_SQL = (
    "SELECT alert_id, version, updated_at, est_loss_m3h, last_window_end, "
    "coalesce(len(gnn_top5) > 0, false) AS located FROM alert_events ORDER BY updated_at"
)
ORDER_SQL = (
    "SELECT alert_id, generator, generated_at, work_order_json FROM work_orders "
    "ORDER BY generated_at"
)
RESIDUAL_COLUMN = re.compile(r"^pressure__(\w+?)__residual$")
NOT_FROM_PAGE = "This request did not come from the dashboard page."
KAFKA_DOWN = "The command could not be sent: Kafka did not accept it. Start Kafka with `make up`."


class CommandNotSentError(RuntimeError):
    """The ``demo.commands`` message did not reach Kafka."""


def plain(value: Any) -> Any:
    """A DuckDB/pandas value as plain Python: arrays -> lists, NaN/NaT -> None."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if getattr(value, "ndim", 0) and hasattr(value, "tolist"):  # numpy array (DuckDB list)
        return [plain(v) for v in value.tolist()]
    if hasattr(value, "to_pydatetime"):
        return view.to_utc(value)
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def files_signature(*folders: Path) -> tuple[int, int]:
    """(number of Parquet files, newest modification time in ns) under the folders.

    Cheap to compute; the lake is read again only when this changes.
    """
    count, newest = 0, 0
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.parquet"):
            try:
                newest = max(newest, path.stat().st_mtime_ns)
            except OSError:  # removed while listing (the runner rewrites its lake)
                continue
            count += 1
    return count, newest


def json_response(payload: Any, status_code: int = 200) -> Response:
    """JSON with UTC datetimes and no NaN (orjson writes NaN as null)."""
    body = orjson.dumps(payload, option=orjson.OPT_UTC_Z | orjson.OPT_NON_STR_KEYS)
    return Response(content=body, status_code=status_code, media_type="application/json")


def refused(status_code: int, message: str) -> Response:
    """A request the server did not carry out, with the sentence the page shows."""
    return json_response({"ok": False, "message": message}, status_code)


def from_the_page(method: str, headers: Mapping[str, str]) -> bool:
    """Whether a state-changing request was made by the dashboard's own page.

    The page sends ``X-Requested-With: leakwatch``. A browser lets another site add that header
    only after a preflight, which this server never approves. Where the browser also reports
    where the request came from (``Sec-Fetch-Site``, ``Origin``), it must be this server.
    """
    if method in ("GET", "HEAD", "OPTIONS"):
        return True
    if headers.get("x-requested-with") != "leakwatch":
        return False
    if headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        return False
    origin = headers.get("origin")
    return origin is None or origin.split("://", 1)[-1] == headers.get("host")


def send_demo_command(command: str, **params: Any) -> None:
    """Publish one ``demo.commands`` message; raise if Kafka did not take it."""
    from leakwatch.demo.runner import send_command

    if send_command(command, **params):
        raise CommandNotSentError(command)


class FailedLeaks:
    """Leaks that were requested and vanished from the runner's state without starting.

    The runner drops a leak whose EPANET signature failed; its state file then shows neither a
    pending nor an active leak. Remembering the pending ones makes that visible for a while.
    """

    def __init__(self, keep_s: float) -> None:
        self.keep = timedelta(seconds=keep_s)
        self._pending: dict[str, dict[str, Any]] = {}
        self._failed: list[tuple[datetime, dict[str, Any]]] = []
        self._last_sim: datetime | None = None

    def update(self, state: Mapping[str, Any], now: datetime) -> list[dict[str, Any]]:
        """Feed the newest state; returns the leaks to show as failed."""
        sim = view.to_utc(state.get("sim_ts"))
        rewound = sim is not None and self._last_sim is not None and sim < self._last_sim
        if not state or rewound or state.get("status") == "starting":
            self._pending.clear()  # a restart clears pending leaks without a failure
        self._last_sim = sim
        pending = {str(p.get("leak_id")): dict(p) for p in state.get("pending_leaks") or []}
        active = {str(lk.get("leak_id")) for lk in state.get("leaks") or []}
        for leak_id, leak in list(self._pending.items()):
            if leak_id not in pending:
                del self._pending[leak_id]
                if leak_id not in active:
                    self._failed.append((now, leak))
        for leak_id, leak in pending.items():
            if "error" in leak and leak_id not in self._pending:
                self._failed.append((now, leak))
            self._pending[leak_id] = leak
        self._failed = [(t, lk) for t, lk in self._failed if now - t <= self.keep]
        seen: set[str] = set()
        out = []
        for _, leak in self._failed:
            if str(leak.get("leak_id")) not in seen:
                seen.add(str(leak.get("leak_id")))
                out.append(leak)
        return out


class LiveSource:
    """Reads the demo runner's state file and the demo lake; returns the live view model."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        self._con: Any = None
        self._layers: frozenset[str] = frozenset()
        self._columns: list[tuple[str, str]] | None = None
        self._cache: dict[str, tuple[tuple[int, int], Any]] = {}
        self._n_pressure: int | None = None
        self._failed = FailedLeaks(float(self.cfg["dashboard"]["failed_leak_seconds"]))

    def _connection(self) -> Any:
        """One DuckDB connection over the demo lake, rebuilt when a layer appears or vanishes."""
        from leakwatch.common import lake
        from leakwatch.demo.runner import demo_lake

        dirs = lake.layer_dirs(demo_lake())
        present = frozenset(name for name in LAYERS if lake.has_parquet(dirs[name]))
        if self._con is not None and present == self._layers:
            return self._con
        self._con, self._columns, self._layers = None, None, present
        if not present:
            return None
        only = [*sorted(present), *(["alerts"] if "alert_events" in present else [])]
        con = lake.connect(lake_root=demo_lake(), only=only)
        # One thread and a small memory cap: Spark needs the laptop more than this does.
        con.execute("SET threads = 1")
        con.execute(f"SET memory_limit = '{self.cfg['dashboard']['duckdb_memory']}'")
        self._con = con
        return con

    def _cached(self, group: str, read: Callable[[Any], Any], empty: Any) -> Any:
        """``read(connection)`` for a group of layers, re-run only when their files changed."""
        from leakwatch.common import lake
        from leakwatch.demo.runner import demo_lake

        dirs = lake.layer_dirs(demo_lake())
        files = files_signature(*(dirs[name] for name in GROUPS[group]))
        hit = self._cache.get(group)
        if hit is not None and hit[0] == files:
            return hit[1]
        if files[0] == 0:  # not written yet, or wiped by a restart
            self._cache[group] = (files, empty)
            return empty
        con = self._connection()
        if con is None:
            return empty
        try:
            value = read(con)
        except Exception:
            self._con = None  # files moved under the view (demo reset): reconnect next time
            raise
        if value:  # an empty answer over existing files is asked again next time
            self._cache[group] = (files, value)
        return value

    def _read_alerts(self, con: Any) -> dict[str, list[dict[str, Any]]]:
        from leakwatch.common import lake

        limit = int(self.cfg["dashboard"]["alerts_limit"])
        rows = {
            "alerts": f"SELECT {ALERT_COLUMNS} FROM alerts ORDER BY start_ts DESC LIMIT {limit}",
            "versions": VERSION_SQL,
            "orders": ORDER_SQL,
        }
        return {
            name: [plain(r) for r in lake.query_df(con, sql).to_dict("records")]
            for name, sql in rows.items()
        }

    def _residual_columns(self, con: Any) -> list[tuple[str, str]]:
        """``(column, sensor id)`` of the per-sensor pressure residuals in ``features_stream``."""
        if self._columns is None:
            names = [r[0] for r in con.execute("DESCRIBE features_stream").fetchall()]
            found = ((name, RESIDUAL_COLUMN.match(name)) for name in names)
            columns = [(name, m.group(1)) for name, m in found if m]
            # None again if the schema was read while a file was being replaced
            self._columns = columns or None
            return columns
        return self._columns

    def _recent(self, table: str, select: str) -> str:
        hours = int(self.cfg["dashboard"]["chart_hours"])
        return (
            f"SELECT {select} FROM {table} WHERE window_end > "
            f"(SELECT max(window_end) FROM {table}) - INTERVAL {hours} HOUR ORDER BY window_end"
        )

    def _read_residuals(self, con: Any) -> list[dict[str, Any]]:
        from leakwatch.common import lake

        columns = self._residual_columns(con)
        if not columns:
            return []
        select = ", ".join(["window_end", *(f'"{c}" AS "{s}"' for c, s in columns)])
        df = lake.query_df(con, self._recent("features_stream", select))
        return [plain(r) for r in df.to_dict("records")]

    def _read_scores(self, con: Any) -> list[dict[str, Any]]:
        from leakwatch.common import lake

        df = lake.query_df(con, self._recent("scores", "window_end, score, threshold, above"))
        return [plain(r) for r in df.to_dict("records")]

    def _read_lake(self) -> dict[str, list[dict[str, Any]]]:
        """Newest alerts with their versions and orders, and the recent windows of the stream."""
        out = dict(
            self._cached("alerts", self._read_alerts, {"alerts": [], "versions": [], "orders": []})
        )
        out["residuals"] = self._cached("residuals", self._read_residuals, [])
        out["scores"] = self._cached("scores", self._read_scores, [])
        return out

    def _pressure_sensors(self) -> int | None:
        if self._n_pressure is None:
            from leakwatch.data import network

            try:
                self._n_pressure = len(network.parse_sensor_config()["pressure"])
            except OSError as exc:
                log.warning("sensor configuration not readable: %s", exc)
        return self._n_pressure

    def __call__(self) -> dict[str, Any]:
        """Build the live view model now."""
        from leakwatch.demo.runner import read_state

        started = time.perf_counter()
        now = datetime.now(UTC)
        state = read_state()
        rows: dict[str, list[dict[str, Any]]] = {}
        error: str | None = None
        try:
            rows = self._read_lake()
        except Exception as exc:
            log.warning("demo lake not readable: %s", exc)
            error = "The demo lake could not be read. It is retried every second."
        out = view.live_view(
            state=state,
            alerts=rows.get("alerts", []),
            now=now,
            cfg=self.cfg,
            n_pressure=self._pressure_sensors(),
            lake_error=error,
            failed_leaks=self._failed.update(state, now),
            versions=rows.get("versions", []),
            residuals=rows.get("residuals", []),
        )
        out["generated_at"] = datetime.now(UTC)
        out["build_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return out


class Snapshots:
    """Keeps the newest result of ``source()``; a daemon thread refreshes it at a fixed interval."""

    def __init__(self, source: Callable[[], dict[str, Any]], interval_s: float) -> None:
        self.source = source
        self.interval_s = interval_s
        self._latest: dict[str, Any] | None = None
        self._lock = threading.Lock()  # one build at a time: the source holds one DuckDB connection
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh(self) -> dict[str, Any]:
        """Build one snapshot now (a failure becomes an ``error`` snapshot, not an exception).

        Builds never overlap: the first page request can arrive while the refresh thread is
        still building the first snapshot, and both would use the same DuckDB connection.
        """
        with self._lock:
            try:
                latest = self.source()
            except Exception as exc:
                log.error("snapshot failed: %s", exc)
                latest = view.unavailable_view(datetime.now(UTC))
            self._latest = latest
            return latest

    def latest(self) -> dict[str, Any]:
        """The newest snapshot (built now if there is none yet)."""
        return self._latest if self._latest is not None else self.refresh()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        """Start the refresh thread."""
        self._thread = threading.Thread(target=self._run, name="dashboard-snapshots", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the refresh thread to end."""
        self._stop.set()


class InjectBody(BaseModel):
    """``POST /api/demo/inject``."""

    node_id: str
    diameter_mm: float


class SpeedBody(BaseModel):
    """``POST /api/demo/speed``."""

    speed: int


def create_app(
    source: Callable[[], dict[str, Any]] | None = None,
    *,
    background: bool = True,
    cfg: Mapping[str, Any] | None = None,
    sender: Callable[..., None] | None = None,
    geometry_loader: Callable[[], Any] | None = None,
) -> FastAPI:
    """The dashboard app.

    Args:
        source: builds the live view model (default: ``LiveSource``, the demo lake + state file).
        background: refresh snapshots in a thread; ``False`` builds one per request (tests).
        cfg: configuration (default: ``load_config()``).
        sender: publishes a demo command (default: ``send_demo_command``, Kafka).
        geometry_loader: returns the map ``Geometry`` (default: ``geometry.load``).

    """
    cfg = cfg or load_config()
    dash = cfg["dashboard"]
    snapshots = Snapshots(source or LiveSource(cfg), float(dash["snapshot_seconds"]))
    send = sender or send_demo_command
    geometry_lock = threading.Lock()

    def geometry() -> Any:
        from leakwatch.dashboard import geometry as geo

        with geometry_lock:  # the first build takes a few seconds; build it once
            return (geometry_loader or geo.load)()

    def current() -> dict[str, Any]:
        return snapshots.latest() if background else snapshots.refresh()

    def command(name: str, **params: Any) -> Response:
        try:
            send(name, **params)
        except CommandNotSentError:
            return refused(503, KAFKA_DOWN)
        except Exception as exc:
            log.error("demo command %s failed: %s", name, exc)
            return refused(503, KAFKA_DOWN)
        return json_response({"ok": True})

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        if background:
            snapshots.start()
        yield
        snapshots.stop()

    app = FastAPI(title="LeakWatch control room", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def guard(request, call_next):  # type: ignore[no-untyped-def]
        """Refuse commands from other sites; make the browser revalidate page files."""
        if not from_the_page(request.method, request.headers):
            return refused(403, NOT_FROM_PAGE)
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/health")
    def health() -> Response:
        return json_response({"ok": True})

    @app.get("/api/live")
    def live() -> Response:
        return json_response(current())

    @app.get("/api/settings")
    def settings() -> Response:
        return json_response(
            {
                "poll_ms": int(dash["poll_ms"]),
                "leak_mm": {
                    "min": int(dash["leak_mm"]["min"]),
                    "max": int(dash["leak_mm"]["max"]),
                    "default": round(float(cfg["demo"]["leak_diameter_m"]) * 1000),
                },
                "speeds": [int(s) for s in dash["speeds"]],
                "ring_floor_m": float(dash["ring_floor_m"]),
            }
        )

    @app.get("/api/network")
    def network_json() -> Response:
        try:
            return json_response(geometry().payload)
        except Exception as exc:
            log.error("network geometry not available: %s", exc)
            return refused(503, "The network model could not be read. Run `make data`.")

    @app.post("/api/demo/inject")
    def inject(body: InjectBody) -> Response:
        low, high = float(dash["leak_mm"]["min"]), float(dash["leak_mm"]["max"])
        if not low <= body.diameter_mm <= high:
            return refused(422, f"Leak size must be between {low:.0f} and {high:.0f} mm.")
        try:
            junctions = set(geometry().payload["junctions"])
        except Exception as exc:
            log.error("network geometry not available: %s", exc)
            return refused(503, "The network model could not be read. Run `make data`.")
        if body.node_id not in junctions:
            return refused(422, f"{body.node_id} is not a junction of L-Town.")
        if current()["runner"]["code"] != "running":
            return refused(
                409, "The replay is not running yet. A leak can start once it says Running."
            )
        return command("inject_leak", node_id=body.node_id, diameter_m=body.diameter_mm / 1000)

    @app.post("/api/demo/speed")
    def speed(body: SpeedBody) -> Response:
        if body.speed not in [int(s) for s in dash["speeds"]]:
            return refused(422, "That replay speed is not offered.")
        return command("set_speed", speed=body.speed)

    @app.post("/api/demo/restart")
    def restart() -> Response:
        if current()["runner"]["code"] in ("offline", "stale", "error"):
            return refused(409, "The demo runner is not running. Start it with `make demo`.")
        return command("reset")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[str(h) for h in dash["allowed_hosts"]])
    return app


def main(argv: list[str] | None = None) -> int:
    """Serve the dashboard with uvicorn."""
    import uvicorn

    cfg = load_config()
    parser = argparse.ArgumentParser(description="LeakWatch control-room screen (D-052)")
    parser.add_argument("--host", default=str(cfg["dashboard"]["host"]))
    parser.add_argument("--port", type=int, default=int(cfg["dashboard"]["port"]))
    args = parser.parse_args(argv)
    log.info("dashboard on http://localhost:%d", args.port)
    uvicorn.run(create_app(cfg=cfg), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
