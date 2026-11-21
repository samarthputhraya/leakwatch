"""``make demo-runner``: the live demo backend (PRD §16, Phase 7; D-025).

One process that

1. replays a quiet stretch of BattLeDIM 2018 (normal under D-015) through Kafka topic
   ``demo.readings`` at a paced speed (``demo.speed``; 2880× = one hour of data every 1.25 s);
2. runs two Spark Structured Streaming queries on it, into the demo lake ``data/demo/lake``:
   the raw ingest (``stream.raw_ingest``) and the feature stream with live K-Means scoring,
   alerts, the GNN localizer (``gnn_top5`` on new alerts) and Kafka ``alerts``;
3. turns every new alert into a work order (``copilot.dispatch``; template fallback when no
   LLM is configured);
4. listens on ``demo.commands`` (``schemas.DemoCommand``):
   ``inject_leak`` {node_id, diameter_m} adds a simulated leak signature (``demo.signature``)
   to the readings from the next step on; ``set_speed`` {speed}; ``reset`` starts over.

``data/demo/state.json`` tells the dashboard what is happening (simulated time, speed, injected
leaks with their true location, status).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from confluent_kafka import Consumer

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.schemas import DemoCommand
from leakwatch.data import battledim
from leakwatch.demo.signature import apply_signature, leak_signature
from leakwatch.ingest import kafka_io
from leakwatch.ingest.replayer import Pacer, encode_step, sensor_meta
from leakwatch.stream import lake_writer

log = get_logger("leakwatch.demo.runner")


def demo_root() -> Path:
    """Data root of the demo (its own lake, checkpoints and state file)."""
    return paths.data_dir() / "demo"


def demo_lake() -> Path:
    """The demo lake (DuckDB views over it: ``lake.connect(lake_root=demo_lake())``)."""
    return demo_root() / "lake"


def state_path() -> Path:
    """JSON status file read by the dashboard."""
    return demo_root() / "state.json"


def read_state() -> dict[str, Any]:
    """Current demo state ({} when the runner has not started)."""
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class DemoRunner:
    """Background replay + streaming queries + leak injection (see module docstring)."""

    def __init__(self, cfg: dict[str, Any], speed: float) -> None:
        self.cfg = cfg
        self.dcfg = cfg["demo"]
        self.speed = speed
        self.topic = str(self.dcfg["topic"])
        self.leaks: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.queries: list[Any] = []
        self.dispatch_threads: list[threading.Thread] = []
        self.started = datetime.now(UTC)
        self.llm: Any = None  # one client for every work order, warmed at start (D-046)
        self.sim_ts: pd.Timestamp | None = None
        self.status = "starting"
        self._reset_flag = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------- lifecycle

    def stream_warm(self) -> bool:
        """The scored feature stream has finished ``demo.warm_batches`` micro-batches with data.

        Its first batches seed the re-baselining history from the lake and compile the Spark
        plans; until then an injected leak waits behind them (D-043). Without a query: warm.
        """
        if not self.queries:
            return True
        need = int(self.dcfg.get("warm_batches", 0))
        try:
            done = sum(
                1 for p in self.queries[0].recentProgress if int(p.get("numInputRows") or 0) > 0
            )
        except Exception:  # a query that just failed reports no progress
            return False
        return done >= need

    def write_state(self) -> None:
        """Atomically update ``state.json``."""
        doc = {
            "status": self.status,
            "sim_ts": None if self.sim_ts is None else self.sim_ts.isoformat(),
            "speed": self.speed,
            "topic": self.topic,
            "lake": str(demo_lake()),
            "background_start": str(self.dcfg["background_start"]),
            "leaks": [{k: v for k, v in lk.items() if k != "delta"} for lk in self.leaks],
            "pending_leaks": [{k: v for k, v in p.items() if k != "delta"} for p in self.pending],
            "started_at": self.started.isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        tmp = state_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
        tmp.replace(state_path())

    def start_streams(self) -> None:
        """Fresh demo lake + topic; start the raw ingest and the scored feature stream."""
        from leakwatch.dl.gnn import make_localizer
        from leakwatch.features.baselines import read_baselines
        from leakwatch.features.spec import load_feature_spec
        from leakwatch.stream.features_stream import (
            FeatureBatchWriter,
            FeatureStreamPaths,
            make_live_scoring,
            start_query,
        )
        from leakwatch.stream.raw_ingest import RawIngestPaths, kafka_readings
        from leakwatch.stream.raw_ingest import start_query as start_raw
        from leakwatch.stream.spark import get_spark

        shutil.rmtree(demo_lake(), ignore_errors=True)
        demo_lake().mkdir(parents=True, exist_ok=True)
        kafka = self.cfg["kafka"]
        kafka_io.delete_topic(kafka, self.topic)
        kafka_io.create_topic(kafka, self.topic, partitions=int(self.dcfg["partitions"]))
        spark = get_spark(
            {**self.cfg["spark"], "shuffle_partitions": int(self.dcfg["shuffle_partitions"])},
            "leakwatch-demo",
            with_kafka=True,
        )
        spec = load_feature_spec(self.cfg)
        baselines = read_baselines(spark, paths.lake_baselines_dir()).cache()
        live = make_live_scoring(self.cfg, spec)
        live.localize = make_localizer()
        inner = live.publish

        def publish(rows: list[dict[str, Any]]) -> None:
            if inner is not None:
                inner(rows)
            new = [r["alert_id"] for r in rows if r.get("version") == 1]
            if new:
                t = threading.Thread(target=self.dispatch, args=(new,), daemon=True)
                t.start()
                self.dispatch_threads.append(t)

        live.publish = publish
        fpaths = FeatureStreamPaths.under(demo_lake())
        writer = FeatureBatchWriter(fpaths, spec, baselines, live)
        trigger = float(self.dcfg["trigger_seconds"])
        stream_cfg = {**self.cfg["features"]["stream"], "starting_offsets": "earliest"}
        readings = kafka_readings(spark, self.cfg, self.topic, stream_cfg)
        self.queries = [start_query(readings, writer, trigger)]
        raw_paths = RawIngestPaths(
            layout=lake_writer.LakeLayout(
                demo_lake() / "raw",
                demo_lake() / "_staging" / "raw",
                demo_lake() / "_commits" / "raw",
            ),
            rejects_dir=demo_lake() / "_rejects" / "raw",
            checkpoint_dir=demo_lake() / "checkpoints" / "raw_ingest",
        )
        raw_cfg = {
            **self.cfg,
            "stream": {
                **self.cfg["stream"],
                "raw": {
                    **self.cfg["stream"]["raw"],
                    "starting_offsets": "earliest",
                    "trigger_seconds": trigger,
                },
            },
        }
        self.queries.append(start_raw(spark, raw_cfg, self.topic, raw_paths))
        log.info("demo streams running on %s -> %s", self.topic, demo_lake())

    def stop_streams(self) -> None:
        """Stop both queries."""
        for q in self.queries:
            try:
                q.stop()
            except Exception as exc:
                log.warning("stopping query: %s", exc)
        self.queries = []

    def warm_llm(self) -> None:
        """Build the work-order model client and send one tiny request (D-046).

        The first request to a hosted endpoint pays for the TLS handshake and, on free tiers,
        often a queue; doing it at start keeps it out of the first alert's work order.
        """
        try:
            from leakwatch.copilot.llm import build_llm
            from leakwatch.copilot.settings import llm_settings

            self.llm = build_llm(llm_settings(), self.cfg["copilot"]["llm"])
            if self.llm is None:
                log.info("no LLM configured: work orders are templates")
                return
            t0 = time.perf_counter()
            self.llm.chat([{"role": "user", "content": "Reply with: ok"}], fast=True)
            log.info("LLM warmed up in %.1f s", time.perf_counter() - t0)
        except Exception as exc:
            log.warning("LLM warm-up failed (work orders fall back to the template): %s", exc)

    def dispatch(self, alert_ids: list[str]) -> None:
        """Work orders for new alerts: the template at once, then the model's version (D-046).

        The template is written first, so the operator has a work order as soon as the alert
        exists; the model's order (same evidence, numbers checked) replaces it in the alerts view
        when it arrives (newest ``generated_at`` wins). A model failure leaves the template.
        """
        try:
            from leakwatch.copilot.dispatch import (
                generate_work_order,
                run_dispatch,
                write_work_order,
            )
            from leakwatch.copilot.tools import ToolContext

            ctx = ToolContext(lake_root=demo_lake())
            counts = run_dispatch(ctx, None, alert_ids=alert_ids)
            log.info("template work orders for %s: %s", alert_ids, counts)
            if self.llm is None:
                return
            for alert_id in alert_ids:
                gen = generate_work_order(ctx, alert_id, self.llm)
                if gen.generator == "llm":
                    write_work_order(gen, ctx.lake_root)
                    log.info("model work order for %s", alert_id)
        except Exception as exc:
            log.error("dispatch failed for %s: %s", alert_ids, exc)

    # -------------------------------------------------------------------------- commands

    def handle(self, cmd: DemoCommand) -> None:
        """Apply one ``demo.commands`` message."""
        if cmd.command == "set_speed":
            self.speed = float(cmd.params.get("speed", self.speed))
            self.pacer.set_speed(self.speed, self.sim_ts)
        elif cmd.command == "reset":
            self._reset_flag = True
        elif cmd.command == "inject_leak":
            node = str(cmd.params["node_id"])
            diameter = float(cmd.params.get("diameter_m", self.dcfg["leak_diameter_m"]))
            leak = {
                "leak_id": f"demo-{uuid.uuid4().hex[:6]}",
                "node_id": node,
                "diameter_m": diameter,
                "requested_sim_ts": None if self.sim_ts is None else self.sim_ts.isoformat(),
                "requested_at": datetime.now(UTC).isoformat(),
            }
            self.pending.append(leak)
            threading.Thread(target=self._prepare, args=(leak,), daemon=True).start()

    def _prepare(self, leak: dict[str, Any]) -> None:
        """Simulate the leak signature from the current simulated time (runs in a thread)."""
        try:
            start = (self.sim_ts or pd.Timestamp.now(tz="UTC")).floor("5min")
            delta, flow = leak_signature(
                leak["node_id"], leak["diameter_m"], start, float(self.dcfg["signature_hours"])
            )
            with self._lock:
                leak["delta"] = delta
                leak["est_flow_m3h"] = round(flow, 2)
        except Exception as exc:
            log.error("leak signature failed: %s", exc)
            with self._lock:
                leak["error"] = str(exc)

    # ------------------------------------------------------------------------------ loop

    def run(self) -> None:
        """Replay until stopped (Ctrl+C); reset on command."""
        demo_root().mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self.warm_llm, daemon=True).start()
        start = pd.Timestamp(self.dcfg["background_start"], tz="UTC")
        hours = float(self.dcfg["background_hours"])
        wide = battledim.load_scada(start.year).loc[start : start + pd.Timedelta(hours=hours)]
        columns = list(wide.columns)
        metas = sensor_meta(columns)
        values = wide.to_numpy(dtype="float64")
        kafka = self.cfg["kafka"]
        consumer = Consumer(
            {
                "bootstrap.servers": kafka["bootstrap_servers"],
                "group.id": f"demo-runner-{uuid.uuid4().hex[:6]}",
                "auto.offset.reset": "latest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([kafka["topics"]["demo_commands"]["name"]])
        producer = kafka_io.make_producer(kafka)
        while True:
            # Overwrite a state file left by an earlier runner (or the pre-reset "running") at
            # once: readers wait for "running", which must mean *this* replay is warm.
            self.status = "starting"
            self.write_state()
            self.start_streams()
            self.leaks, self.pending, self._reset_flag = [], [], False
            self.status = "warming up"
            self.pacer = Pacer(self.speed)
            for i, ts in enumerate(wide.index):
                self.sim_ts = ts
                self.pacer.wait_for(ts)
                msg = consumer.poll(0)
                while msg is not None:
                    if msg.error() is None:
                        try:
                            self.handle(DemoCommand.model_validate_json(msg.value()))
                        except Exception as exc:
                            log.warning("ignored command %r: %s", msg.value(), exc)
                    msg = consumer.poll(0)
                if self._reset_flag:
                    break
                with self._lock:
                    for leak in [p for p in self.pending if "delta" in p or "error" in p]:
                        self.pending.remove(leak)
                        if "delta" in leak:
                            leak["start_ts"] = ts.isoformat()
                            leak["start_step"] = i
                            self.leaks.append(leak)
                row = values[i]
                for leak in self.leaks:
                    row = apply_signature(row, columns, leak["delta"], i - int(leak["start_step"]))
                for key, payload in encode_step(
                    ts.strftime("%Y-%m-%dT%H:%M:%SZ"), int(ts.year), row, metas, "battledim", None
                ):
                    producer.produce(self.topic, value=payload, key=key)
                producer.poll(0)
                if i % 3 == 0:
                    if self.status != "running" and i * 300 > 5400 and self.stream_warm():
                        self.status = "running"
                    self.write_state()
            producer.flush(10)
            self.stop_streams()
            if not self._reset_flag:
                self.status = "finished"
                self.write_state()
                log.info("background replay finished; waiting for reset")
                while not self._reset_flag:
                    msg = consumer.poll(1.0)
                    if msg is not None and msg.error() is None:
                        try:
                            self.handle(DemoCommand.model_validate_json(msg.value()))
                        except Exception as exc:
                            log.warning("ignored command %r: %s", msg.value(), exc)
            log.info("reset")


def send_command(command: str, **params: Any) -> None:
    """Publish one ``demo.commands`` message (used by the dashboard and the demo check)."""
    cfg = load_config()
    producer = kafka_io.make_producer(cfg["kafka"])
    cmd = DemoCommand(command=command, params=params)  # type: ignore[arg-type]
    producer.produce(
        cfg["kafka"]["topics"]["demo_commands"]["name"], value=orjson.dumps(cmd.model_dump())
    )
    producer.flush(10)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LeakWatch live demo backend.")
    parser.add_argument("--speed", type=float, help="simulated seconds per wall second")
    args = parser.parse_args(argv)
    cfg = load_config()
    kafka_io.wait_for_kafka(cfg["kafka"])
    runner = DemoRunner(cfg, float(args.speed or cfg["demo"]["speed"]))
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.stop_streams()
    return 0


if __name__ == "__main__":
    sys.exit(main())
