"""Control-room screen, server side (D-052): ``python -m leakwatch.dashboard.api``.

A FastAPI app that serves the page (``static/``, no build step) and its JSON:

- ``GET /api/live``: everything the live page shows, shaped by ``view.live_view``. One background
  thread rebuilds it every ``dashboard.snapshot_seconds`` from ``state.json`` and the demo lake
  (DuckDB); requests return the last snapshot, so any number of browser tabs cost the same.
- ``GET /api/health``: for scripts that wait for the server.

DuckDB is used from the snapshot thread only. A lake that cannot be read is reported in the
snapshot (``error``), never as an empty result.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.dashboard import view

log = get_logger("leakwatch.dashboard.api")
STATIC = Path(__file__).with_name("static")
LIVE_VIEWS = ("alerts",)
ALERT_COLUMNS = (
    "alert_id, origin, status, start_ts, end_ts, area, kmeans_score_max, threshold, "
    "est_loss_m3h, windows_above, last_window_end, top_sensors, gnn_top5, work_order_json, "
    "work_order_generator, created_at, updated_at"
)


def plain(value: Any) -> Any:
    """A DuckDB/pandas value as plain Python: arrays -> lists, NaN/NaT -> None."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if getattr(value, "ndim", 0) and hasattr(value, "tolist"):  # numpy array (DuckDB list column)
        return [plain(v) for v in value.tolist()]
    if hasattr(value, "to_pydatetime"):
        return view.to_utc(value)
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def json_response(payload: Any, status_code: int = 200) -> Response:
    """JSON with UTC datetimes and no NaN (orjson writes NaN as null)."""
    body = orjson.dumps(payload, option=orjson.OPT_UTC_Z | orjson.OPT_NON_STR_KEYS)
    return Response(content=body, status_code=status_code, media_type="application/json")


class LiveSource:
    """Reads the demo runner's state file and the demo lake; returns the live view model."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        self._con: Any = None
        self._n_pressure: int | None = None

    def _connection(self) -> Any:
        """One DuckDB connection over the demo lake, rebuilt until its views exist."""
        from leakwatch.common import lake
        from leakwatch.demo.runner import demo_lake

        if self._con is None:
            con = lake.connect(lake_root=demo_lake(), only=LIVE_VIEWS)
            names = {r[0] for r in con.execute("SELECT view_name FROM duckdb_views()").fetchall()}
            if "alerts" not in names:  # a fresh demo lake has no alert yet: nothing to read
                con.close()
                return None
            self._con = con
        return self._con

    def _alerts(self) -> list[dict[str, Any]]:
        from leakwatch.common import lake

        con = self._connection()
        if con is None:
            return []
        limit = int(self.cfg["dashboard"]["alerts_limit"])
        try:
            df = lake.query_df(
                con, f"SELECT {ALERT_COLUMNS} FROM alerts ORDER BY start_ts DESC LIMIT {limit}"
            )
        except Exception:
            self._con = None  # files moved under the view (demo reset): reconnect next time
            raise
        return [plain(row) for row in df.to_dict("records")]

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
        state = read_state()
        alerts: list[dict[str, Any]] = []
        error: str | None = None
        try:
            alerts = self._alerts()
        except Exception as exc:
            log.warning("demo lake not readable: %s", exc)
            error = "The demo lake could not be read. It is retried every second."
        out = view.live_view(
            state=state,
            alerts=alerts,
            now=datetime.now(UTC),
            cfg=self.cfg,
            n_pressure=self._pressure_sensors(),
            lake_error=error,
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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh(self) -> dict[str, Any]:
        """Build one snapshot now (a failure becomes an ``error`` snapshot, not an exception)."""
        try:
            self._latest = self.source()
        except Exception as exc:
            log.error("snapshot failed: %s", exc)
            self._latest = {
                "runner": {
                    "code": "error",
                    "label": "Screen data unavailable",
                    "detail": "",
                    "live": False,
                },
                "clock": None,
                "banner": {
                    "kind": "unknown",
                    "title": "Screen data unavailable",
                    "sub": "The server could not build the live view. See its log.",
                },
                "open_alerts": 0,
                "error": "snapshot failed",
                "generated_at": datetime.now(UTC),
            }
        return self._latest

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


def create_app(
    source: Callable[[], dict[str, Any]] | None = None,
    *,
    background: bool = True,
    cfg: Mapping[str, Any] | None = None,
) -> FastAPI:
    """The dashboard app.

    Args:
        source: builds the live view model (default: ``LiveSource``, the demo lake + state file).
        background: refresh snapshots in a thread; ``False`` builds one per request (tests).
        cfg: configuration (default: ``load_config()``).

    """
    cfg = cfg or load_config()
    snapshots = Snapshots(source or LiveSource(cfg), float(cfg["dashboard"]["snapshot_seconds"]))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if background:
            snapshots.start()
        yield
        snapshots.stop()

    app = FastAPI(title="LeakWatch control room", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def revalidate(request, call_next):  # type: ignore[no-untyped-def]
        """Page files are revalidated on every load, so an updated file is never served stale."""
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/health")
    def health() -> Response:
        return json_response({"ok": True})

    @app.get("/api/live")
    def live() -> Response:
        return json_response(snapshots.latest() if background else snapshots.refresh())

    @app.get("/api/settings")
    def settings() -> Response:
        dash = cfg["dashboard"]
        return json_response({"poll_ms": int(dash["poll_ms"])})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    """Serve the dashboard with uvicorn."""
    import uvicorn

    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=str(cfg["dashboard"]["host"]))
    parser.add_argument("--port", type=int, default=int(cfg["dashboard"]["port"]))
    args = parser.parse_args(argv)
    log.info("dashboard on http://localhost:%d", args.port)
    uvicorn.run(create_app(cfg=cfg), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
