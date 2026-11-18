"""``make stream-latency``: steady-state micro-batch time and publish→lake latency at demo speed.

A full-year replay at max speed measures throughput, but its latencies only show how far Spark is
behind the replayer. This benchmark replays a few simulated hours at a paced speed (default 360×,
one 5-minute step every 0.83 s, as in the live demo) into a throwaway topic and a scratch lake
(``data/bench/latency``), so the real lake and its watermark are untouched.

Writes ``reports/metrics/stream_latency_paced.json``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid

import pandas as pd

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.data import battledim
from leakwatch.ingest import kafka_io
from leakwatch.ingest.replayer import Pacer, Replayer, ReplayStats, delivery_error_counter
from leakwatch.stream import lake_writer
from leakwatch.stream.raw_ingest import RawIngestPaths, await_query, start_query, summarise_run
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.stream.latency_bench")


def bench_paths() -> RawIngestPaths:
    """Scratch lake + checkpoint under ``data/bench/latency`` (deleted before and after)."""
    root = paths.data_dir() / "bench" / "latency"
    return RawIngestPaths(
        layout=lake_writer.LakeLayout(root / "raw", root / "_staging", root / "_commits"),
        rejects_dir=root / "_rejects",
        checkpoint_dir=root / "checkpoint",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the paced benchmark and write its metrics."""
    parser = argparse.ArgumentParser(description="Paced replay -> Spark latency benchmark.")
    parser.add_argument("--start", default="2019-01-15 18:00", help="first simulated timestamp")
    parser.add_argument("--hours", type=float, default=6.0, help="simulated hours to replay")
    parser.add_argument("--speed", type=float, default=360.0, help="simulated s per wall s")
    args = parser.parse_args(argv)

    cfg = load_config()
    kafka_cfg = cfg["kafka"]
    kafka_io.wait_for_kafka(kafka_cfg)
    start = pd.Timestamp(args.start, tz="UTC")
    end = start + battledim.TIMESTEP * (round(args.hours * 12) - 1)
    wide = battledim.load_scada(start.year).loc[start:end]

    topic = f"bench.latency.{uuid.uuid4().hex[:8]}"
    raw_paths = bench_paths()
    root = raw_paths.layout.layer_dir.parent
    shutil.rmtree(root, ignore_errors=True)
    kafka_io.create_topic(
        kafka_cfg, topic, partitions=int(kafka_cfg["topics"]["readings"]["partitions"])
    )
    try:
        spark = get_spark(cfg["spark"], "leakwatch-latency-bench", with_kafka=True)
        query = start_query(spark, cfg, topic, raw_paths)
        stats = ReplayStats()
        producer = kafka_io.make_producer(kafka_cfg, on_error=delivery_error_counter(stats))
        log.info("replaying %d steps at %gx into %s", len(wide), args.speed, topic)
        Replayer(producer, topic, Pacer(args.speed), progress_every_steps=0, stats=stats).run(wide)
        await_query(query, stop_when_idle_s=20)
        progress = [dict(p) for p in query.recentProgress]
        summary = summarise_run(progress, lake_writer.read_markers(raw_paths.layout))
    finally:
        kafka_io.delete_topic(kafka_cfg, topic)
        shutil.rmtree(root, ignore_errors=True)

    path = write_metrics(
        "stream_latency_paced",
        {
            "scenario": {
                "start": str(start),
                "simulated_hours": args.hours,
                "speed": args.speed,
                "steps": stats.steps,
                "readings": stats.readings,
                "readings_per_s_offered": round(stats.readings / stats.wall_s, 1),
            },
            "trigger_seconds": cfg["stream"]["raw"]["trigger_seconds"],
            **summary,
            "note": (
                "publish_to_lake_s.newest_in_batch = commit time minus Kafka timestamp of the "
                "newest message in each micro-batch; includes waiting for the next trigger."
            ),
        },
    )
    log.info(
        "batches %d, batch p95 %.2f s, publish->lake p50 %.2f s / p95 %.2f s -> %s",
        summary["batches_with_data"],
        summary["batch_duration_s"]["p95"] or 0,
        summary["publish_to_lake_s"]["oldest_in_batch"]["p50"] or 0,
        summary["publish_to_lake_s"]["oldest_in_batch"]["p95"] or 0,
        path,
    )
    return 0 if summary["rows_written"] == stats.readings else 1


if __name__ == "__main__":
    sys.exit(main())
