"""Batch/stream parity (PRD §7.2, §11): the streaming job must produce the batch job's features.

The toy readings (gaps, a stuck run, an out-of-range AMR value) are cut into chunks whose
boundaries fall inside windows, inside a filled gap and inside the stuck run. Spark Structured
Streaming reads one chunk file per micro-batch; the query is stopped half-way and restarted from
its checkpoint. The result must equal the batch features of the same readings, window by window,
except the last window, which the stream cannot emit without data 15 minutes past its end.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from leakwatch.common import lake
from leakwatch.features.parity import compare_tables
from leakwatch.features.windows import (
    complete_window_range,
    select_windows,
    window_features,
    with_partitions,
)
from leakwatch.stream import lake_writer
from leakwatch.stream.features_stream import FeatureBatchWriter, FeatureStreamPaths, start_query

pytestmark = pytest.mark.spark

#: Chunk boundaries in 5-minute steps (toy data has 144 steps).
CUTS = [0, 10, 17, 31, 40, 41, 85, 90, 144]
RESTART_AFTER = 4  # chunks in the first run of the query

STREAM_SCHEMA = StructType(
    [
        StructField("source", StringType()),
        StructField("scenario_id", StringType()),
        StructField("sensor_type", StringType()),
        StructField("sensor_id", StringType()),
        StructField("ts", TimestampType()),
        StructField("value", DoubleType()),
        StructField("out_of_range", BooleanType()),
    ]
)


def write_chunks(toy: pd.DataFrame, input_dir: Path, first: int, last: int) -> None:
    """Write chunks ``first..last-1`` as Parquet files with increasing modification times."""
    input_dir.mkdir(parents=True, exist_ok=True)
    step = ((toy["ts"] - toy["ts"].min()) / pd.Timedelta(minutes=5)).astype(int)
    for k in range(first, last):
        part = toy[(step >= CUTS[k]) & (step < CUTS[k + 1])].copy()
        part["source"] = "battledim"
        part["scenario_id"] = None
        part = part[[f.name for f in STREAM_SCHEMA.fields]]
        table = pa.Table.from_pandas(part, preserve_index=False).cast(
            pa.schema(
                [
                    ("source", pa.string()),
                    ("scenario_id", pa.string()),
                    ("sensor_type", pa.string()),
                    ("sensor_id", pa.string()),
                    ("ts", pa.timestamp("us", tz="UTC")),
                    ("value", pa.float64()),
                    ("out_of_range", pa.bool_()),
                ]
            )
        )
        path = input_dir / f"chunk-{k:02d}.parquet"
        pq.write_table(table, path)
        os.utime(path, (1_700_000_000 + k * 10, 1_700_000_000 + k * 10))


def run_stream(spark, input_dir: Path, writer: FeatureBatchWriter) -> None:
    readings = (
        spark.readStream.schema(STREAM_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .parquet(str(input_dir))
        .withColumn("is_valid", F.lit(True))
    )
    query = start_query(readings, writer, available_now=True)
    query.awaitTermination()
    assert query.exception() is None


@pytest.fixture(scope="module")
def parity(spark, small_spec, toy_pdf, to_spark, toy_base, tmp_path_factory):
    root = tmp_path_factory.mktemp("parity")
    lake_root, input_dir = root / "lake", root / "input"
    stream_paths = FeatureStreamPaths.under(lake_root)
    writer = FeatureBatchWriter(stream_paths, small_spec, toy_base)

    write_chunks(toy_pdf, input_dir, 0, RESTART_AFTER)
    run_stream(spark, input_dir, writer)
    after_first_run = len(lake_writer.read_markers(stream_paths.layout))
    write_chunks(toy_pdf, input_dir, RESTART_AFTER, len(CUTS) - 1)
    run_stream(spark, input_dir, writer)  # restarts from the checkpoint and carried state
    markers = lake_writer.read_markers(stream_paths.layout)
    run_stream(spark, input_dir, writer)  # nothing new: must not add batches

    readings = to_spark(toy_pdf)
    t = toy_pdf["ts"]
    first, last = complete_window_range(
        int(t.min().timestamp()), int(t.max().timestamp()), small_spec
    )
    batch = with_partitions(
        select_windows(window_features(readings, toy_base, small_spec), first, last), small_spec
    )
    batch.write.partitionBy("source", "window", "date").parquet(str(lake_root / "features"))
    result = compare_tables(lake.connect(lake_root=lake_root), small_spec, "battledim")
    return {
        "result": result,
        "markers": markers,
        "after_first_run": after_first_run,
        "markers_after_rerun": len(lake_writer.read_markers(stream_paths.layout)),
        "stream_paths": stream_paths,
    }


def test_stream_features_equal_batch_features(parity):
    r = parity["result"]
    assert r["mismatched_cells"] == 0, r["mismatches_by_column"]
    assert r["parity"] is True
    assert r["windows"]["stream_duplicates"] == 0
    assert r["windows"]["stream"] == r["windows"]["matched"] == 44
    assert r["cells_compared"] == 44 * r["columns_compared"]


def test_only_the_last_window_is_missing_from_the_stream(parity):
    r = parity["result"]
    assert r["windows"]["batch_only"] == 1
    assert r["batch_only_windows"] == ["2018-01-01T11:00:00Z"]


def test_one_micro_batch_per_chunk_and_restart_resumes(parity):
    markers = parity["markers"]
    assert parity["after_first_run"] == RESTART_AFTER
    assert [m["batch_id"] for m in markers] == list(range(len(CUTS) - 1))
    assert markers[0]["windows"] == 0  # first 50 minutes: no complete window yet
    assert sum(m["windows"] for m in markers) == 44
    assert parity["markers_after_rerun"] == len(markers)
    # windows are emitted in order, each once
    emitted = [
        (m["first_window_start_s"], m["last_window_start_s"]) for m in markers if m["windows"]
    ]
    for (_, prev_last), (nxt_first, _) in itertools.pairwise(emitted):
        assert nxt_first == prev_last + 900


def test_only_the_latest_tail_is_kept(parity):
    tails = sorted(p.name for p in parity["stream_paths"].state_dir.glob("tail-b*"))
    assert tails == [f"tail-b{len(CUTS) - 2:010d}"]
