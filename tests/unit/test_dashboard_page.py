"""The control-room page in a real browser (D-052): idle, alert, work order and error states.

The app is served with fixed snapshots built by ``view.live_view``; headless Chrome loads the page
and dumps the DOM once the page has fetched and rendered. WSL has no Chrome, so the Windows Chrome
is used through interop. The WebGL map is not asserted (headless Chrome may not provide WebGL; the
page then shows its own note), the panels around it are. Run with ``make ui-check``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from html import unescape

import networkx as nx
import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from leakwatch.common.config import load_config  # noqa: E402
from leakwatch.dashboard import api, geometry, view  # noqa: E402

pytestmark = pytest.mark.slow

CHROMES = (
    os.environ.get("LEAKWATCH_CHROME") or "",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
)
CHROME = next((c for c in CHROMES if c and os.path.exists(c)), None)
CFG = load_config(environ={})
NOW = datetime.now(UTC)

LEAK = {
    "leak_id": "demo-21d0e4",
    "node_id": "n470",
    "diameter_m": 0.03,
    "requested_at": (NOW - timedelta(seconds=60)).isoformat(),
    "est_flow_m3h": 57.03,
    "start_ts": "2018-11-19T19:35:00+00:00",
}
ALERT = {
    "alert_id": "5bab5c3a",
    "status": "open",
    "area": "A",
    "start_ts": "2018-11-19T20:15:00+00:00",
    "kmeans_score_max": 108.14,
    "threshold": 1.3664,
    "est_loss_m3h": 53.966,
    "top_sensors": [
        {"sensor_id": "pressure__n469", "contribution": 0.79},
        {"sensor_id": "pressure__n114", "contribution": 0.08},
    ],
    "gnn_top5": [{"node_id": n, "prob": 0.0} for n in ("n469", "n470", "n471", "n94", "n468")],
    "created_at": (NOW - timedelta(seconds=46)).isoformat(),
    "updated_at": (NOW - timedelta(seconds=5)).isoformat(),
}
ORDER = {
    "alert_id": "5bab5c3a",
    "generator": "llm",
    "generated_at": (NOW - timedelta(seconds=30)).isoformat(),
    "work_order_json": json.dumps(
        {
            "priority": "P1",
            "estimated_loss_m3_per_hour": 47.874,
            "pipes_to_inspect": ["p437", "p438", "p439"],
            "recommended_actions": ["Check the top sensors.", "Walk the pipes in order."],
            "manual_citations": [{"doc": "LeakWatch SOP", "chapter": "6. Confirm", "page": "n/a"}],
            "summary": "A possible leak in area A.",
        }
    ),
}


def state(**extra) -> dict:
    return {
        "status": "running",
        "sim_ts": "2018-11-20T05:30:00+00:00",
        "speed": 2880.0,
        "background_start": "2018-11-19 00:00",
        "updated_at": datetime.now(UTC).isoformat(),
        **extra,
    }


def snapshot(kind: str) -> dict:
    residuals = [{"window_end": "2018-11-20T05:15:00Z", "n469": -2.0, "n114": -0.8}]
    if kind == "error":
        return view.unavailable_view(datetime.now(UTC))
    if kind == "idle":
        return view.live_view(
            state=state(), alerts=[], now=datetime.now(UTC), cfg=CFG, n_pressure=33
        )
    return view.live_view(
        state=state(leaks=[LEAK]),
        alerts=[ALERT],
        now=datetime.now(UTC),
        cfg=CFG,
        n_pressure=33,
        versions=[{"alert_id": "5bab5c3a", "updated_at": ALERT["created_at"], "located": True}],
        residuals=residuals,
        orders=[ORDER] if kind == "order" else [],
        distance=lambda a, b: 49.0,
    )


def tiny_geometry() -> geometry.Geometry:
    nodes = {
        "n469": [0.0, 0.0],
        "n470": [50.0, 0.0],
        "n471": [100.0, 0.0],
        "n94": [100.0, 60.0],
        "n468": [0.0, 60.0],
        "n114": [50.0, 60.0],
    }
    pipes = [
        {"id": f"p43{k}", "a": a, "b": b, "kind": "Pipe", "d": 0.1, "area": "A"}
        for k, (a, b) in enumerate(
            [
                ("n469", "n470"),
                ("n470", "n471"),
                ("n471", "n94"),
                ("n94", "n114"),
                ("n114", "n468"),
                ("n468", "n469"),
            ]
        )
    ]
    payload = {
        "bounds": [0.0, 0.0, 100.0, 60.0],
        "nodes": nodes,
        "junctions": sorted(nodes),
        "areas": dict.fromkeys(nodes, "A"),
        "pipes": pipes,
        "sensors": ["n469", "n114"],
        "area_labels": {"A": [50.0, 30.0]},
        "outlines": {"A": [[[-40, -40], [140, -40], [140, 100], [-40, 100], [-40, -40]]]},
        "tint_full_m": 2.0,
        "tint": {"n469": [[0, 1.0]], "n114": [[1, 1.0]]},
        "counts": {"junctions": 6, "pipes": 6, "pressure_sensors": 2},
    }
    return geometry.Geometry(payload=payload, graph=nx.Graph())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def page():
    if CHROME is None:
        pytest.skip("no Chrome or Edge found (set LEAKWATCH_CHROME)")
    exe = str(CHROME)
    shown = {"kind": "idle"}
    app = api.create_app(
        lambda: snapshot(shown["kind"]), background=False, cfg=CFG, geometry_loader=tiny_geometry
    )
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.1)

    def dump(kind: str) -> str:
        shown["kind"] = kind
        res = subprocess.run(
            [
                exe,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--mute-audio",
                "--virtual-time-budget=8000",
                "--window-size=1480,860",
                "--dump-dom",
                f"http://localhost:{port}/",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return res.stdout

    yield dump
    server.should_exit = True
    thread.join(10)


def text_of(dom: str) -> str:
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", dom, flags=re.S)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", body)))


def no_placeholders(text: str) -> None:
    for bad in ("undefined", "NaN", "null", "[object Object]", "Traceback"):
        assert bad not in text, bad


def test_idle_page(page):
    text = text_of(page("idle"))
    assert "LeakWatch" in text and "Running" in text
    assert "No open alerts" in text and "Watching 33 pressure sensors" in text
    assert "Start a leak" in text and "Ask the copilot" in text
    no_placeholders(text)


def test_alert_page_waits_for_the_work_order(page):
    text = text_of(page("alert"))
    assert "Alert in Area A" in text and "score 79 times the threshold" in text
    assert "Five candidate junctions, best first" in text and "n471" in text
    assert "Writing the work order for this alert" in text
    assert "Context: the alert on screen is 5bab5c3a" in text
    no_placeholders(text)


def test_work_order_page(page):
    text = text_of(page("order"))
    assert "Priority P1" in text and "Pipes to inspect, in order" in text
    assert "p437" in text and "47.9" in text and "Model version" in text
    assert "candidate 1: n469" in text  # finished steps folded to one line
    assert "Reveal the true leak" in text
    no_placeholders(text)


def test_error_page_says_what_happened(page):
    text = text_of(page("error"))
    assert "Screen data unavailable" in text
    assert "The server could not build the live view" in text
    no_placeholders(text)
