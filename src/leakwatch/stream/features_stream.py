"""Spark Structured Streaming: Kafka ``scada.readings`` → live feature rows (PRD §7.2, Phase 2).

The query parses and deduplicates readings with the raw ingest's functions, then hands each
micro-batch to :class:`FeatureBatchWriter`, which calls the same
:func:`~leakwatch.features.windows.window_features` as the batch job (batch/stream parity,
PRD §11). Cleaning and 1 h windows need readings from earlier micro-batches, so the writer carries
a small *tail* of recent readings from one batch to the next (docs/DECISIONS.md D-018):

- state after each batch: latest event time seen, start of the next window to emit, and the tail
  (readings from ``next window start - history`` on, where history covers the stuck and
  forward-fill rules);
- a window is emitted once data ``lateness`` past its last step has arrived, each window once,
  in order; readings for windows already emitted are counted as late and ignored;
- feature files, tail and state are committed together with the lake_writer protocol (D-013),
  so a retried micro-batch starts from the state the previous committed batch left.

Event time is global across sources and scenarios, like the raw ingest's watermark (D-013).

Output: ``lake/features_stream/source=…/window=1h/date=…/`` (same columns as ``lake/features``).

Usage:
    python -m leakwatch.stream.features_stream --stop-when-idle 60 --metrics-name features_stream
    python -m leakwatch.stream.features_stream --reset
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.features.baselines import read_baselines
from leakwatch.features.clean import READING_COLUMNS
from leakwatch.features.spec import FeatureSpec, load_feature_spec
from leakwatch.features.windows import (
    ceil_to,
    complete_window_range,
    select_windows,
    window_features,
    with_partitions,
)
from leakwatch.stream import lake_writer
from leakwatch.stream.raw_ingest import await_query, kafka_readings, summarise_run
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.stream.features_stream")

JOB = "features_stream"
PARTITION_COLS = ("source", "window", "date")


@dataclass(frozen=True)
class FeatureStreamPaths:
    """Filesystem locations of the feature stream (all under one lake + checkpoint root)."""

    layout: lake_writer.LakeLayout
    state_dir: Path
    checkpoint_dir: Path

    @classmethod
    def default(cls) -> FeatureStreamPaths:
        """Locations under the configured data dir."""
        return cls(
            layout=lake_writer.LakeLayout(
                layer_dir=paths.lake_features_stream_dir(),
                staging_dir=paths.lake_staging_dir(JOB),
                commits_dir=paths.lake_commits_dir(JOB),
            ),
            state_dir=paths.lake_state_dir(JOB),
            checkpoint_dir=paths.checkpoints_dir() / JOB,
        )

    @classmethod
    def under(cls, root: Path) -> FeatureStreamPaths:
        """Same layout below ``root`` (tests and benchmarks)."""
        return cls(
            layout=lake_writer.LakeLayout(
                layer_dir=root / "features_stream",
                staging_dir=root / "_staging" / JOB,
                commits_dir=root / "_commits" / JOB,
            ),
            state_dir=root / "_state" / JOB,
            checkpoint_dir=root / "checkpoints" / JOB,
        )

    def ensure(self) -> None:
        """Create the layer, staging, commit and state directories."""
        self.layout.ensure()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Delete the stream's features, bookkeeping, state and checkpoint."""
        for d in (
            self.layout.layer_dir,
            self.layout.staging_dir,
            self.layout.commits_dir,
            self.state_dir,
            self.checkpoint_dir,
        ):
            if d.exists():
                shutil.rmtree(d)
                log.info("deleted %s", d)

    def check_consistent(self) -> None:
        """Refuse to start a fresh checkpoint on top of committed batches (would duplicate)."""
        if lake_writer.latest_marker(self.layout) is not None and not self.checkpoint_dir.exists():
            raise RuntimeError(
                "feature stream has committed batches but no checkpoint; run with --reset"
            )


@dataclass(frozen=True)
class StreamState:
    """What the feature stream carries from one micro-batch to the next."""

    max_ts_s: int | None = None
    next_window_start_s: int | None = None
    tail: str | None = None  # directory name under state_dir

    @classmethod
    def load(cls, layout: lake_writer.LakeLayout) -> StreamState:
        """State left by the latest committed batch (empty before the first one)."""
        marker = lake_writer.latest_marker(layout)
        return cls(**marker["state"]) if marker and "state" in marker else cls()


def _staged_rows(stage: Path) -> int:
    return sum(pq.ParquetFile(f).metadata.num_rows for f in stage.rglob("*.parquet"))


@dataclass
class FeatureBatchWriter:
    """``foreachBatch`` sink: readings → feature rows (exactly once), with carried history."""

    stream_paths: FeatureStreamPaths
    spec: FeatureSpec
    baselines: DataFrame

    def __call__(self, batch_df: DataFrame, batch_id: int) -> None:  # noqa: D102 (class doc)
        layout = self.stream_paths.layout
        if lake_writer.is_committed(layout, batch_id):
            log.info("batch %d already committed; skipping replay of it", batch_id)
            return
        state = StreamState.load(layout)
        batch_df.persist()
        try:
            valid = batch_df.where(F.col("is_valid"))
            ts_s = F.col("ts").cast("long")
            late = (
                ts_s < F.lit(state.next_window_start_s)
                if state.next_window_start_s is not None
                else F.lit(False)
            )
            aggs = [
                F.count(F.lit(1)).alias("rows"),
                F.min(ts_s).alias("min_ts_s"),
                F.max(ts_s).alias("max_ts_s"),
                F.coalesce(F.sum(late.cast("long")), F.lit(0)).alias("late_readings"),
            ]
            if "kafka_ts" in batch_df.columns:
                aggs += [
                    F.min(F.col("kafka_ts").cast("double")).alias("oldest_kafka_epoch_s"),
                    F.max(F.col("kafka_ts").cast("double")).alias("newest_kafka_epoch_s"),
                ]
            first = valid.agg(*aggs).first()
            info: dict[str, Any] = first.asDict() if first is not None else {"rows": 0}
            if not info["rows"]:
                lake_writer.commit_batch(
                    layout, batch_id, {**info, "windows": 0, "state": asdict(state)}
                )
                return
            self._emit(batch_df.sparkSession, valid, batch_id, state, info)
        finally:
            batch_df.unpersist()

    def _emit(
        self,
        spark: SparkSession,
        valid: DataFrame,
        batch_id: int,
        state: StreamState,
        info: dict[str, Any],
    ) -> None:
        spec, layout = self.spec, self.stream_paths.layout
        state_dir = self.stream_paths.state_dir
        max_ts = max(int(info["max_ts_s"]), state.max_ts_s or 0)
        next_start = (
            state.next_window_start_s
            if state.next_window_start_s is not None
            else ceil_to(int(info["min_ts_s"]), spec.slide_s)
        )
        _, last_start = complete_window_range(next_start, max_ts, spec, spec.lateness_s)

        combined = valid.select(*READING_COLUMNS)
        if state.tail:
            combined = combined.unionByName(spark.read.parquet(str(state_dir / state.tail)))

        windows = 0
        next_after = next_start
        if last_start >= next_start:
            stage = layout.staging_for(batch_id)
            features = window_features(combined, self.baselines, spec)
            (
                with_partitions(select_windows(features, next_start, last_start), spec)
                .repartition("date")
                .sortWithinPartitions("window_start")
                .write.mode("overwrite")
                .partitionBy(*PARTITION_COLS)
                .parquet(str(stage))
            )
            windows = _staged_rows(stage)
            next_after = last_start + spec.slide_s

        tail = f"tail-b{batch_id:010d}"
        keep_from = F.timestamp_seconds(F.lit(next_after - spec.history_s))
        combined.where(F.col("ts") >= keep_from).write.mode("overwrite").parquet(
            str(state_dir / tail)
        )
        new_state = StreamState(max_ts_s=max_ts, next_window_start_s=next_after, tail=tail)
        lake_writer.commit_batch(
            layout,
            batch_id,
            {
                **info,
                "committed_epoch_s": time.time(),
                "windows": windows,
                "first_window_start_s": next_start if windows else None,
                "last_window_start_s": last_start if windows else None,
                "state": asdict(new_state),
            },
        )
        for old in state_dir.glob("tail-b*"):
            if old.name != tail:
                shutil.rmtree(old, ignore_errors=True)


def start_query(
    readings: DataFrame,
    writer: FeatureBatchWriter,
    trigger_seconds: float | None = None,
    available_now: bool = False,
):
    """Start the feature query on a streaming DataFrame of readings (returns ``StreamingQuery``).

    ``readings`` needs ``clean.READING_COLUMNS`` plus ``is_valid``: parsed Kafka messages in
    production (:func:`~leakwatch.stream.raw_ingest.kafka_readings`), a file stream in tests.
    """
    writer.stream_paths.ensure()
    stream = (
        readings.writeStream.queryName(JOB)
        .foreachBatch(writer)
        .option("checkpointLocation", str(writer.stream_paths.checkpoint_dir))
    )
    if available_now:
        stream = stream.trigger(availableNow=True)
    else:
        stream = stream.trigger(processingTime=f"{int(trigger_seconds or 5)} seconds")
    return stream.start()


def summarise(progress: list[dict[str, Any]], markers: list[dict[str, Any]]) -> dict[str, Any]:
    """Run summary: the raw ingest's timing numbers plus feature-specific totals."""
    base = summarise_run(progress, markers)
    base["readings_processed"] = base.pop("rows_written")
    for key in ("rejects", "out_of_range_flagged"):
        base.pop(key, None)
    emitted = [m for m in markers if m.get("windows")]
    return {
        **base,
        "windows_emitted": sum(int(m["windows"]) for m in markers),
        "late_readings_ignored": sum(int(m.get("late_readings") or 0) for m in markers),
        "first_window_start_s": emitted[0]["first_window_start_s"] if emitted else None,
        "last_window_start_s": emitted[-1]["last_window_start_s"] if emitted else None,
    }


def run(
    cfg: Mapping[str, Any],
    topic: str | None = None,
    stream_paths: FeatureStreamPaths | None = None,
    stop_when_idle_s: float | None = None,
    available_now: bool = False,
    spark: SparkSession | None = None,
    spec: FeatureSpec | None = None,
) -> dict[str, Any]:
    """Run the feature stream from Kafka until idle (or drained) and return its summary."""
    feat_cfg = cfg["features"]["stream"]
    topic_name: str = topic or cfg["kafka"]["topics"]["readings"]["name"]
    stream_paths = stream_paths or FeatureStreamPaths.default()
    stream_paths.check_consistent()
    spec = spec or load_feature_spec(cfg)
    spark = spark or get_spark(cfg["spark"], "leakwatch-features-stream", with_kafka=True)
    baselines = read_baselines(spark, paths.lake_baselines_dir()).cache()
    before = {m["batch_id"] for m in lake_writer.read_markers(stream_paths.layout)}
    readings = kafka_readings(spark, cfg, topic_name, feat_cfg)
    writer = FeatureBatchWriter(stream_paths, spec, baselines)
    query = start_query(readings, writer, feat_cfg["trigger_seconds"], available_now)
    log.info("feature stream running: topic=%s -> %s", topic_name, stream_paths.layout.layer_dir)
    t0 = time.perf_counter()
    if available_now:
        query.awaitTermination()
        if (exc := query.exception()) is not None:
            raise RuntimeError(f"streaming query failed: {exc}")
    else:
        await_query(query, stop_when_idle_s)
    wall = time.perf_counter() - t0
    markers = [
        m for m in lake_writer.read_markers(stream_paths.layout) if m["batch_id"] not in before
    ]
    summary = {
        "topic": topic_name,
        "wall_s": round(wall, 1),
        **summarise([dict(p) for p in query.recentProgress], markers),
    }
    log.info(
        "feature stream done: %d readings -> %d windows in %d batches (p95 batch %.1f s)",
        summary["readings_processed"],
        summary["windows_emitted"],
        summary["batches_with_data"],
        summary["batch_duration_s"]["p95"] or 0.0,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Kafka scada.readings -> live feature rows.")
    parser.add_argument("--topic", help="override the readings topic")
    parser.add_argument("--stop-when-idle", type=float, help="stop after N seconds without data")
    parser.add_argument("--available-now", action="store_true", help="drain the topic, then stop")
    parser.add_argument("--reset", action="store_true", help="delete stream features + state")
    parser.add_argument("--metrics-name", help="write reports/metrics/<name>.json when done")
    args = parser.parse_args(argv)

    cfg = load_config()
    stream_paths = FeatureStreamPaths.default()
    if args.reset:
        stream_paths.reset()
        return 0
    summary = run(
        cfg,
        topic=args.topic,
        stream_paths=stream_paths,
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
                "features_stream": dict(cfg["features"]["stream"]),
            },
        )
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
