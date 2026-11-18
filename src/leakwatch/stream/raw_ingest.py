"""Spark Structured Streaming: Kafka ``scada.readings`` → raw lake layer (PRD §7.2, Phase 1).

Per micro-batch:
1. parse the JSON payload against the ``scada.readings`` contract and cast ``ts`` to a UTC
   timestamp; messages that break the contract are kept aside in ``lake/_rejects``;
2. drop duplicate readings (same source, scenario, sensor and timestamp) that arrive within the
   event-time watermark (``stream.raw.watermark``) — e.g. a replayer restarted mid-run;
3. flag values outside the configured physical range (``stream.ranges``) without dropping them;
4. write ``lake/raw/source=…/year=…/date=…/sensor_type=…/`` exactly once (``lake_writer``).

Stateful cleaning (forward-filling short gaps, stuck-sensor runs) creates or relabels rows, so it
belongs to the cleaned layer built in Phase 2; the raw layer keeps readings as received
(docs/DECISIONS.md D-011).

Usage:
    python -m leakwatch.stream.raw_ingest                       # run until Ctrl+C
    python -m leakwatch.stream.raw_ingest --stop-when-idle 60   # stop once caught up
    python -m leakwatch.stream.raw_ingest --reset               # delete raw lake + checkpoint
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.common.schemas import (
    RAW_LAKE_COLUMNS,
    SENSOR_TYPES,
    SOURCES,
    UNITS,
    readings_payload_spark_schema,
)
from leakwatch.stream import lake_writer
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.stream.raw_ingest")

LAYER = "raw"
PARTITION_COLS = ("source", "year", "date", "sensor_type")
#: A reading is identified by these fields; repeats within the watermark are duplicates.
DEDUP_KEY = (
    "source",
    "scenario_id",
    "sensor_type",
    "sensor_id",
    "ts",
    "reject_partition",
    "reject_offset",
)
TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ssXXX"


# --------------------------------------------------------------------------------------------
# Transformations (identical for streaming and static DataFrames)
# --------------------------------------------------------------------------------------------


def _map_lookup(mapping: Mapping[str, Any], key: Column) -> Column:
    pairs = []
    for k, v in mapping.items():
        pairs += [F.lit(k), F.lit(v)]
    return F.create_map(*pairs)[key]


def out_of_range_flag(ranges: Mapping[str, list[float]]) -> Column:
    """True when ``value`` lies outside ``ranges[sensor_type]`` (inclusive bounds)."""
    lo = _map_lookup({t: float(r[0]) for t, r in ranges.items()}, F.col("sensor_type"))
    hi = _map_lookup({t: float(r[1]) for t, r in ranges.items()}, F.col("sensor_type"))
    return F.coalesce((F.col("value") < lo) | (F.col("value") > hi), F.lit(False))


def parse_readings(kafka_df: DataFrame, ranges: Mapping[str, list[float]]) -> DataFrame:
    """Kafka records (``value``, ``partition``, ``offset``, ``timestamp``) → typed readings.

    Adds ``is_valid`` (the message satisfies the ``scada.readings`` contract), ``out_of_range``
    and ``date`` (UTC, the partition column). Invalid rows keep their raw text in ``raw_json`` and
    get a unique ``reject_partition``/``reject_offset`` so deduplication never merges them.
    """
    p = F.from_json(F.col("value").cast("string"), readings_payload_spark_schema())
    parsed = kafka_df.select(
        p.alias("p"),
        F.col("value").cast("string").alias("raw_json"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_ts"),
    ).select(
        F.to_timestamp(F.col("p.ts"), TS_FORMAT).alias("ts"),
        F.col("p.sensor_id").alias("sensor_id"),
        F.col("p.sensor_type").alias("sensor_type"),
        F.col("p.value").alias("value"),
        F.col("p.unit").alias("unit"),
        F.col("p.source").alias("source"),
        F.col("p.scenario_id").alias("scenario_id"),
        F.col("p.year").alias("year"),
        "raw_json",
        "kafka_partition",
        "kafka_offset",
        "kafka_ts",
    )
    expected_unit = _map_lookup(UNITS, F.col("sensor_type"))
    valid = (
        F.col("ts").isNotNull()
        & F.col("sensor_id").isNotNull()
        & (F.length("sensor_id") > 0)
        & F.col("sensor_type").isin(*SENSOR_TYPES)
        & F.col("value").isNotNull()
        & ~F.isnan("value")
        & (F.abs("value") != F.lit(float("inf")))
        & (F.col("unit") == expected_unit)
        & F.col("source").isin(*SOURCES)
        & ~((F.col("source") == "sim") & F.col("scenario_id").isNull())
        & (F.col("year") == F.year("ts"))
    )
    return (
        parsed.withColumn("is_valid", F.coalesce(valid, F.lit(False)))
        .withColumn("out_of_range", out_of_range_flag(ranges))
        .withColumn("date", F.date_format("ts", "yyyy-MM-dd"))
        .withColumn(
            "reject_partition",
            F.when(F.col("is_valid"), F.lit(None)).otherwise(F.col("kafka_partition")),
        )
        .withColumn(
            "reject_offset", F.when(F.col("is_valid"), F.lit(None)).otherwise(F.col("kafka_offset"))
        )
    )


def deduplicate(readings: DataFrame, watermark: str) -> DataFrame:
    """Drop repeated readings that arrive within ``watermark`` of event time (streaming only)."""
    return readings.withWatermark("ts", watermark).dropDuplicatesWithinWatermark(list(DEDUP_KEY))


# --------------------------------------------------------------------------------------------
# Sink
# --------------------------------------------------------------------------------------------


@dataclass
class RawBatchWriter:
    """``foreachBatch`` sink: valid readings → raw layer (exactly once), invalid → rejects."""

    layout: lake_writer.LakeLayout
    rejects_dir: Path

    def __call__(self, batch_df: DataFrame, batch_id: int) -> None:  # noqa: D102 (see class doc)
        if lake_writer.is_committed(self.layout, batch_id):
            log.info("batch %d already committed; skipping replay of it", batch_id)
            return
        batch_df.persist()
        try:
            rows = (
                batch_df.where("is_valid")
                .withColumn("ingest_ts", F.current_timestamp())
                .select(*RAW_LAKE_COLUMNS)
            )
            (
                rows.repartition("date", "sensor_type")
                .sortWithinPartitions("sensor_id", "ts")
                .write.mode("overwrite")
                .partitionBy(*PARTITION_COLS)
                .parquet(str(self.layout.staging_for(batch_id)))
            )
            # Timestamps leave Spark as epoch seconds / UTC text: PySpark would otherwise convert
            # them to naive datetimes in the Python process's local time zone.
            valid_ts = F.when(F.col("is_valid"), F.col("ts"))
            stats = batch_df.agg(
                F.coalesce(F.sum(F.col("is_valid").cast("long")), F.lit(0)).alias("rows"),
                F.coalesce(F.sum((~F.col("is_valid")).cast("long")), F.lit(0)).alias("rejects"),
                F.coalesce(
                    F.sum((F.col("is_valid") & F.col("out_of_range")).cast("long")), F.lit(0)
                ).alias("out_of_range"),
                F.min(F.col("kafka_ts").cast("double")).alias("oldest_kafka_epoch_s"),
                F.max(F.col("kafka_ts").cast("double")).alias("newest_kafka_epoch_s"),
                F.date_format(F.min(valid_ts), "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("min_ts"),
                F.date_format(F.max(valid_ts), "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("max_ts"),
            ).first()
            info = stats.asDict() if stats is not None else {}
            info["committed_epoch_s"] = time.time()
            if info.get("rejects"):
                (
                    batch_df.where(~F.col("is_valid"))
                    .select("raw_json", "kafka_partition", "kafka_offset", "kafka_ts")
                    .write.mode("overwrite")
                    .json(str(self.rejects_dir / f"batch={batch_id}"))
                )
                log.warning("batch %d: %d messages rejected", batch_id, info["rejects"])
            lake_writer.commit_batch(self.layout, batch_id, info)
        finally:
            batch_df.unpersist()


# --------------------------------------------------------------------------------------------
# Query lifecycle
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawIngestPaths:
    """Filesystem locations of the raw ingest job (all under the data dir)."""

    layout: lake_writer.LakeLayout
    rejects_dir: Path
    checkpoint_dir: Path

    @classmethod
    def default(cls) -> RawIngestPaths:
        """Locations under the configured data dir."""
        return cls(
            layout=lake_writer.LakeLayout(
                layer_dir=paths.lake_raw_dir(),
                staging_dir=paths.lake_staging_dir(LAYER),
                commits_dir=paths.lake_commits_dir(LAYER),
            ),
            rejects_dir=paths.lake_rejects_dir() / LAYER,
            checkpoint_dir=paths.checkpoints_dir() / "raw_ingest",
        )

    def reset(self) -> None:
        """Delete the raw layer, its bookkeeping and the streaming checkpoint."""
        for d in (
            self.layout.layer_dir,
            self.layout.staging_dir,
            self.layout.commits_dir,
            self.rejects_dir,
            self.checkpoint_dir,
        ):
            if d.exists():
                shutil.rmtree(d)
                log.info("deleted %s", d)

    def check_consistent(self) -> None:
        """Refuse to start a fresh checkpoint on top of existing lake data (would duplicate it)."""
        has_lake = (
            any(self.layout.commits_dir.glob("*.json"))
            if self.layout.commits_dir.exists()
            else False
        )
        if has_lake and not self.checkpoint_dir.exists():
            raise RuntimeError(
                "raw lake has data but the streaming checkpoint is missing; re-reading the topic "
                "would duplicate readings. Run with --reset to rebuild the raw layer."
            )


def kafka_readings(
    spark: SparkSession,
    cfg: Mapping[str, Any],
    topic: str,
    source_cfg: Mapping[str, Any],
) -> DataFrame:
    """Streaming DataFrame of parsed, deduplicated readings from a Kafka topic.

    Shared by the raw ingest and the feature stream so both see exactly the same readings.

    Args:
        spark: session with the Kafka connector.
        cfg: merged config (``kafka``, ``stream.ranges``, ``stream.raw.watermark``).
        topic: topic to subscribe to.
        source_cfg: ``starting_offsets`` and ``max_offsets_per_trigger`` of this query.

    """
    source = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", cfg["kafka"]["bootstrap_servers"])
        .option("subscribe", topic)
        .option("startingOffsets", str(source_cfg["starting_offsets"]))
        .option("maxOffsetsPerTrigger", int(source_cfg["max_offsets_per_trigger"]))
        .option("failOnDataLoss", "false")
        .load()
    )
    return deduplicate(
        parse_readings(source, cfg["stream"]["ranges"]), str(cfg["stream"]["raw"]["watermark"])
    )


def start_query(
    spark: SparkSession,
    cfg: Mapping[str, Any],
    topic: str,
    raw_paths: RawIngestPaths,
    available_now: bool = False,
):
    """Build and start the streaming query (returns ``StreamingQuery``)."""
    raw_cfg = cfg["stream"]["raw"]
    readings = kafka_readings(spark, cfg, topic, raw_cfg)
    raw_paths.layout.ensure()
    writer = (
        readings.writeStream.queryName("raw_ingest")
        .foreachBatch(RawBatchWriter(raw_paths.layout, raw_paths.rejects_dir))
        .option("checkpointLocation", str(raw_paths.checkpoint_dir))
    )
    if available_now:
        writer = writer.trigger(availableNow=True)
    else:
        writer = writer.trigger(processingTime=f"{int(raw_cfg['trigger_seconds'])} seconds")
    return writer.start()


def await_query(query, stop_when_idle_s: float | None, poll_s: float = 2.0) -> None:
    """Wait for the query; stop it after ``stop_when_idle_s`` without input once data has flowed."""
    last_batch_id = -1
    last_data_at: float | None = None
    try:
        while query.isActive:
            query.awaitTermination(poll_s)
            for prog in query.recentProgress:
                if prog["batchId"] > last_batch_id:
                    last_batch_id = prog["batchId"]
                    if prog["numInputRows"] > 0:
                        last_data_at = time.monotonic()
            if (
                stop_when_idle_s is not None
                and last_data_at is not None
                and time.monotonic() - last_data_at > stop_when_idle_s
                and not query.status["isDataAvailable"]
            ):
                log.info("no new data for %.0f s; stopping", stop_when_idle_s)
                query.stop()
    except KeyboardInterrupt:
        log.info("interrupted; stopping query")
        query.stop()
    exc = query.exception()
    if exc is not None:
        raise RuntimeError(f"streaming query failed: {exc}")


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarise_run(progress: list[dict[str, Any]], markers: list[dict[str, Any]]) -> dict[str, Any]:
    """Throughput and latency numbers for the metrics file."""
    busy = [p for p in progress if p["numInputRows"] > 0]
    durations = [p["durationMs"]["triggerExecution"] / 1000 for p in busy]
    ops = [op for p in progress for op in p.get("stateOperators", [])]
    dropped = sum(op.get("numRowsDroppedByWatermark", 0) for op in ops)
    duplicates = sum(
        (op.get("customMetrics") or {}).get("numDroppedDuplicateRows", 0) for op in ops
    )
    rows_in = sum(p["numInputRows"] for p in busy)
    # Publish -> lake latency of the newest message in each batch (best case) and of the oldest
    # (includes time spent queued in Kafka behind earlier batches).
    lag_newest = [
        m["committed_epoch_s"] - m["newest_kafka_epoch_s"]
        for m in markers
        if m.get("newest_kafka_epoch_s") is not None
    ]
    lag_oldest = [
        m["committed_epoch_s"] - m["oldest_kafka_epoch_s"]
        for m in markers
        if m.get("oldest_kafka_epoch_s") is not None
    ]
    first = busy[0]["timestamp"] if busy else None
    last = busy[-1]["timestamp"] if busy else None
    span_s = None
    if first and last:
        t0 = datetime.fromisoformat(first.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(last.replace("Z", "+00:00"))
        span_s = (t1 - t0).total_seconds() + (durations[-1] if durations else 0)
    # The first batch pays one-off costs (JVM warm-up, code generation, Kafka connector start), so
    # steady-state numbers exclude it; it is reported on its own.
    first_id = busy[0]["batchId"] if busy else None
    steady_durations = durations[1:]
    steady = [m for m in markers if m.get("batch_id") != first_id]
    steady_newest = [
        m["committed_epoch_s"] - m["newest_kafka_epoch_s"]
        for m in steady
        if m.get("newest_kafka_epoch_s") is not None
    ]
    steady_oldest = [
        m["committed_epoch_s"] - m["oldest_kafka_epoch_s"]
        for m in steady
        if m.get("oldest_kafka_epoch_s") is not None
    ]
    return {
        "batches_with_data": len(busy),
        "kafka_records_read": rows_in,
        "rows_written": sum(int(m.get("rows") or 0) for m in markers),
        "rejects": sum(int(m.get("rejects") or 0) for m in markers),
        "out_of_range_flagged": sum(int(m.get("out_of_range") or 0) for m in markers),
        "duplicates_dropped": int(duplicates),
        "late_rows_dropped_by_watermark": int(dropped),
        "batch_duration_s": {
            "p50": _pct(durations, 0.5),
            "p95": _pct(durations, 0.95),
            "max": max(durations) if durations else None,
            "mean": statistics.fmean(durations) if durations else None,
        },
        "target_batch_duration_s": 10,
        "batches_over_target": sum(d > 10 for d in durations),
        "processing_span_s": span_s,
        "rows_per_s_over_span": (rows_in / span_s) if span_s else None,
        "publish_to_lake_s": {
            "newest_in_batch": {"p50": _pct(lag_newest, 0.5), "p95": _pct(lag_newest, 0.95)},
            "oldest_in_batch": {"p50": _pct(lag_oldest, 0.5), "p95": _pct(lag_oldest, 0.95)},
        },
        "first_batch_duration_s": durations[0] if durations else None,
        "steady_state_excluding_first_batch": {
            "batches": len(steady_durations),
            "batch_duration_s": {
                "p50": _pct(steady_durations, 0.5),
                "p95": _pct(steady_durations, 0.95),
                "max": max(steady_durations) if steady_durations else None,
            },
            "batches_over_target": sum(d > 10 for d in steady_durations),
            "publish_to_lake_s": {
                "newest_in_batch": {
                    "p50": _pct(steady_newest, 0.5),
                    "p95": _pct(steady_newest, 0.95),
                },
                "oldest_in_batch": {
                    "p50": _pct(steady_oldest, 0.5),
                    "p95": _pct(steady_oldest, 0.95),
                },
            },
        },
    }


def run(
    cfg: Mapping[str, Any],
    topic: str | None = None,
    raw_paths: RawIngestPaths | None = None,
    stop_when_idle_s: float | None = None,
    available_now: bool = False,
    spark: SparkSession | None = None,
) -> dict[str, Any]:
    """Run the raw ingest query to completion and return its summary."""
    topic_name: str = topic or cfg["kafka"]["topics"]["readings"]["name"]
    raw_paths = raw_paths or RawIngestPaths.default()
    raw_paths.check_consistent()
    spark = spark or get_spark(cfg["spark"], "leakwatch-raw-ingest", with_kafka=True)
    markers_before = {m["batch_id"] for m in lake_writer.read_markers(raw_paths.layout)}
    query = start_query(spark, cfg, topic_name, raw_paths, available_now=available_now)
    log.info("raw ingest running: topic=%s -> %s", topic_name, raw_paths.layout.layer_dir)
    t0 = time.perf_counter()
    if available_now:
        query.awaitTermination()
        exc = query.exception()
        if exc is not None:
            raise RuntimeError(f"streaming query failed: {exc}")
    else:
        await_query(query, stop_when_idle_s)
    wall = time.perf_counter() - t0
    progress = [dict(p) for p in query.recentProgress]
    markers = [
        m for m in lake_writer.read_markers(raw_paths.layout) if m["batch_id"] not in markers_before
    ]
    summary = {"topic": topic_name, "wall_s": round(wall, 1), **summarise_run(progress, markers)}
    log.info(
        "raw ingest done: %d rows written in %d batches (p95 batch %.1f s)",
        summary["rows_written"],
        summary["batches_with_data"],
        summary["batch_duration_s"]["p95"] or 0.0,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Kafka scada.readings -> raw Parquet lake.")
    parser.add_argument("--topic", help="override the readings topic")
    parser.add_argument("--stop-when-idle", type=float, help="stop after N seconds without data")
    parser.add_argument("--available-now", action="store_true", help="drain the topic, then stop")
    parser.add_argument(
        "--reset", action="store_true", help="delete raw lake + checkpoint and exit"
    )
    parser.add_argument("--metrics-name", help="write reports/metrics/<name>.json when done")
    args = parser.parse_args(argv)

    cfg = load_config()
    raw_paths = RawIngestPaths.default()
    if args.reset:
        raw_paths.reset()
        return 0
    summary = run(
        cfg,
        topic=args.topic,
        raw_paths=raw_paths,
        stop_when_idle_s=args.stop_when_idle,
        available_now=args.available_now,
    )
    if args.metrics_name:
        path = write_metrics(
            args.metrics_name,
            {
                **summary,
                "spark": {
                    k: cfg["spark"][k] for k in ("master", "driver_memory", "shuffle_partitions")
                },
                "stream_raw": dict(cfg["stream"]["raw"]),
            },
        )
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
