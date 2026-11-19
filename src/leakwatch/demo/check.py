"""``make demo-check``: run the 90-second demo three times without a person (PRD Phase 7 DoD).

Starts the demo runner (``demo.runner``) as a subprocess, waits until the stream is warm, then
for each run: inject a leak at a demo junction through ``demo.commands`` and measure, in wall
time from the click, when (1) the alert is in the demo lake, (2) it carries the localizer's top 5
and (3) a work order exists. PRD §16 wants steps 2–5 within 60 s. Between runs it resets the
demo. Also records the pipe distance from the localizer's candidates to the injected junction.

Writes ``reports/metrics/demo_runs.json``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any

import duckdb
import pandas as pd

from leakwatch.common import lake
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.demo.runner import demo_lake, read_state, send_command, state_path

log = get_logger("leakwatch.demo.check")


def wait_for(predicate: Any, timeout_s: float, poll_s: float = 1.0) -> float | None:
    """Seconds until ``predicate()`` is true, or None after ``timeout_s``."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            if predicate():
                return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(poll_s)
    return None


ALERT_COLUMNS = ["alert_id", "start_ts", "last_window_end", "gnn_top5", "work_order_json"]


def current_alerts(after: pd.Timestamp) -> pd.DataFrame:
    """Alerts in the demo lake that fired after a simulated time (empty before the first one)."""
    con = lake.connect(lake_root=demo_lake())
    try:
        df = lake.query_df(con, f"SELECT {', '.join(ALERT_COLUMNS)} FROM alerts")
    except duckdb.CatalogException:  # a fresh demo lake has no alerts table yet
        return pd.DataFrame(columns=ALERT_COLUMNS)
    return df[df["start_ts"] >= after]


def distance_to(node: str, candidates: list[str]) -> dict[str, float | None]:
    """Pipe distance (m) from the true leak junction to the top-1 and nearest of the top 5."""
    import networkx as nx

    from leakwatch.data import network

    graph = network.pipe_graph(network.load_network())
    d = nx.single_source_dijkstra_path_length(graph, node, weight="length")
    dist = [float(d[c]) for c in candidates if c in d]
    return {"top1_m": dist[0] if dist else None, "best_of_top5_m": min(dist) if dist else None}


def main(argv: list[str] | None = None) -> int:
    """Run the demo ``--runs`` times and write the timings."""
    parser = argparse.ArgumentParser(description="Headless demo rehearsal.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    from leakwatch.demo import preflight

    if preflight.main() != 0:
        return 1
    cfg = load_config()
    nodes = list(cfg["demo"]["check_nodes"])
    diameter = float(cfg["demo"]["leak_diameter_m"])
    # A state file from an earlier demo says "running" before this runner has subscribed to
    # the command topic; an inject sent then is lost (the first rehearsal failed that way).
    state_path().unlink(missing_ok=True)
    runner = subprocess.Popen([sys.executable, "-m", "leakwatch.demo.runner"])
    runs: list[dict[str, Any]] = []
    try:
        for k in range(args.runs):
            if k:
                send_command("reset")
                wait_for(lambda: read_state().get("status") != "running", 120)
            warm = wait_for(lambda: read_state().get("status") == "running", 600, 2)
            if warm is None:
                runs.append({"run": k + 1, "ok": False, "error": "stream did not warm up"})
                continue
            time.sleep(float(cfg["demo"]["quiet_before_inject_s"]))
            node = nodes[k % len(nodes)]
            sim_before = pd.Timestamp(read_state()["sim_ts"])
            quiet = current_alerts(pd.Timestamp("1970-01-01", tz="UTC")).empty
            t_click = time.perf_counter()
            send_command("inject_leak", node_id=node, diameter_m=diameter)

            def since_click(predicate: Any, t0: float = t_click) -> float | None:
                """Wall seconds from the click until ``predicate`` holds (None on timeout)."""
                if wait_for(predicate, args.budget_s * 3) is None:
                    return None
                return round(time.perf_counter() - t0, 1)

            alert_s = since_click(lambda t=sim_before: not current_alerts(t).empty)
            gnn_s = since_click(
                lambda t=sim_before: (
                    current_alerts(t)["gnn_top5"].map(lambda v: v is not None and len(v) > 0).any()
                )
            )
            wo_s = since_click(
                lambda t=sim_before: current_alerts(t)["work_order_json"].notna().any()
            )
            row: dict[str, Any] = {
                "run": k + 1,
                "node": node,
                "diameter_m": diameter,
                "quiet_before_inject": bool(quiet),
                "seconds_to_alert": alert_s,
                "seconds_to_localizer": gnn_s,
                "seconds_to_work_order": wo_s,
                "wall_s": round(time.perf_counter() - t_click, 1),
                "speed": read_state().get("speed"),
            }
            df = current_alerts(sim_before)
            if not df.empty and df["gnn_top5"].iloc[0] is not None:
                cands = [c["node_id"] for c in df["gnn_top5"].iloc[0]]
                row["localizer_top5_at_alert"] = cands
                row["distance_at_alert"] = distance_to(node, cands)
                row["alert_start_sim"] = df["start_ts"].iloc[0].isoformat()
                row["sim_hours_inject_to_alert"] = round(
                    (df["start_ts"].iloc[0] - sim_before).total_seconds() / 3600, 2
                )
            # The live localizer refines an open alert during its first hours (D-033): record the
            # top 5 the operator sees once that is over.
            refine = pd.Timedelta(hours=float(cfg["dl"]["gnn"].get("live_refine_hours", 0)))
            row["seconds_to_refined_localizer"] = since_click(
                lambda t=sim_before, r=refine: (
                    (current_alerts(t)["last_window_end"] - current_alerts(t)["start_ts"]) > r
                ).any()
            )
            df = current_alerts(sim_before)
            if not df.empty and df["gnn_top5"].iloc[0] is not None:
                cands = [c["node_id"] for c in df["gnn_top5"].iloc[0]]
                row["localizer_top5"] = cands
                row["distance"] = distance_to(node, cands)
            row["ok"] = bool(
                row["seconds_to_work_order"] is not None
                and row["seconds_to_work_order"] <= args.budget_s
            )
            runs.append(row)
            log.info("run %d: %s", k + 1, row)
    finally:
        runner.terminate()
        try:
            runner.wait(30)
        except subprocess.TimeoutExpired:
            runner.kill()
    payload = {
        "runs": runs,
        "all_ok": all(r.get("ok") for r in runs) and len(runs) == args.runs,
        "budget_s": args.budget_s,
        "note": "wall seconds from the inject command to each artefact in the demo lake; "
        "the leak signature is simulated with EPANET and added to replayed BattLeDIM 2018 data",
    }
    path = write_metrics("demo_runs", payload)
    log.info("demo check: all_ok=%s -> %s", payload["all_ok"], path)
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
