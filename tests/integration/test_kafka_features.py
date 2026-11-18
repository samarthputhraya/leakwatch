"""End to end parity: replayer -> Kafka -> Spark feature stream == Spark batch features.

Replays two real BattLeDIM days into a throwaway topic and runs the feature stream (same Kafka
parsing and dedup as the raw ingest) in many small micro-batches into a temporary lake. The batch
function applied to the same two days must give the same feature rows (needs `make up`).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import functions as F

from leakwatch.common import lake, paths
from leakwatch.common.config import deep_merge, load_config
from leakwatch.data import battledim
from leakwatch.features.baselines import compute_baselines, read_baselines
from leakwatch.features.clean import clean_readings
from leakwatch.features.parity import compare_tables
from leakwatch.features.spec import load_feature_spec
from leakwatch.features.windows import (
    complete_window_range,
    select_windows,
    window_features,
    with_partitions,
)
from leakwatch.ingest import kafka_io
from leakwatch.ingest.replayer import Pacer, Replayer, ReplayStats, delivery_error_counter
from leakwatch.stream import lake_writer
from leakwatch.stream.features_stream import FeatureBatchWriter, FeatureStreamPaths, start_query
from leakwatch.stream.raw_ingest import kafka_readings
from leakwatch.stream.spark import get_spark

pytestmark = [pytest.mark.integration, pytest.mark.needs_data, pytest.mark.slow]

START, END = "2019-03-01 00:00", "2019-03-02 23:55"


@pytest.fixture(scope="module")
def cfg():
    config = load_config()
    try:
        kafka_io.wait_for_kafka(config["kafka"], timeout_s=5)
    except TimeoutError:
        pytest.skip("Kafka is not running (`make up`)")
    # Small micro-batches: the two days span ~7 batches, so state is carried several times.
    small = {"max_offsets_per_trigger": 10_000, "starting_offsets": "earliest"}
    return deep_merge(config, {"features": {"stream": small}})


@pytest.fixture
def topic(cfg):
    name = f"it.features.{uuid.uuid4().hex[:8]}"
    kafka_io.create_topic(cfg["kafka"], name, partitions=3)
    yield name
    kafka_io.delete_topic(cfg["kafka"], name)


def static_readings(spark, wide: pd.DataFrame):
    long = battledim.to_long(wide)
    rows = [
        (r.source, r.scenario_id, r.sensor_type, r.sensor_id, int(r.ts.timestamp()), r.value)
        for r in long.itertuples(index=False)
    ]
    ddl = "source string, scenario_id string, sensor_type string, sensor_id string, " + (
        "t long, value double"
    )
    return (
        spark.createDataFrame(rows, ddl)
        .withColumn("ts", F.timestamp_seconds("t"))
        .withColumn("out_of_range", F.lit(False))
        .drop("t")
    )


def test_kafka_feature_stream_matches_batch(cfg, topic, tmp_path: Path):
    wide = battledim.load_scada(2019).loc[START:END]
    stats = ReplayStats()
    producer = kafka_io.make_producer(cfg["kafka"], on_error=delivery_error_counter(stats))
    Replayer(producer, topic, Pacer(None), progress_every_steps=0, stats=stats).run(wide)
    assert stats.readings == 2 * 288 * 119 and stats.delivery_errors == 0

    spark = get_spark(cfg["spark"], "leakwatch-it-features", with_kafka=True, master="local[2]")
    spec = load_feature_spec(cfg)
    readings = static_readings(spark, wide)
    if any(paths.lake_baselines_dir().glob("*.parquet")):
        baselines = read_baselines(spark, paths.lake_baselines_dir()).cache()
    else:  # any baseline works for parity; use the two days themselves
        baselines = compute_baselines(clean_readings(readings, spec), spec, None).cache()

    stream_paths = FeatureStreamPaths.under(tmp_path)
    writer = FeatureBatchWriter(stream_paths, spec, baselines)
    stream = kafka_readings(spark, cfg, topic, cfg["features"]["stream"])
    query = start_query(stream, writer, available_now=True)
    query.awaitTermination()
    assert query.exception() is None
    markers = lake_writer.read_markers(stream_paths.layout)
    assert len([m for m in markers if m["rows"]]) >= 5
    assert sum(int(m["rows"]) for m in markers) == 2 * 288 * 119
    assert sum(int(m["late_readings"] or 0) for m in markers) == 0

    t0 = int(wide.index[0].timestamp())
    first, last = complete_window_range(t0, int(wide.index[-1].timestamp()), spec)
    features = window_features(readings, baselines, spec)
    batch = with_partitions(select_windows(features, first, last), spec)
    batch.write.partitionBy("source", "window", "date").parquet(str(tmp_path / "features"))

    result = compare_tables(lake.connect(lake_root=tmp_path), spec, "battledim")
    assert result["mismatched_cells"] == 0, result["mismatches_by_column"]
    assert result["parity"] is True
    # 2 days = 192 windows, minus the last 4 (not complete in batch, 3) / not yet emitted (1)
    assert result["windows"]["batch"] == 2 * 96 - 3
    assert result["windows"]["stream"] == result["windows"]["matched"] == 2 * 96 - 4
    assert result["batch_only_windows"] == ["2019-03-02T23:00:00Z"]
