"""End to end: replayer -> Kafka -> Spark Structured Streaming -> raw lake (needs `make up`).

Replays one real BattLeDIM day into a throwaway topic, adds one malformed message and one exact
duplicate, runs the raw-ingest query in several micro-batches into a temporary lake, and checks
that the lake holds exactly the source readings: nothing lost, nothing duplicated, reject kept.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

from leakwatch.common import lake
from leakwatch.common.config import deep_merge, load_config
from leakwatch.common.schemas import RAW_LAKE_SCHEMA, sensor_column
from leakwatch.data import battledim
from leakwatch.ingest import kafka_io
from leakwatch.ingest.replayer import Pacer, Replayer, ReplayStats, delivery_error_counter
from leakwatch.stream.raw_ingest import RawIngestPaths, run
from leakwatch.stream.spark import get_spark

pytestmark = [pytest.mark.integration, pytest.mark.needs_data, pytest.mark.slow]

DAY = "2019-03-01"


@pytest.fixture(scope="module")
def cfg():
    config = load_config()
    try:
        kafka_io.wait_for_kafka(config["kafka"], timeout_s=5)
    except TimeoutError:
        pytest.skip("Kafka is not running (`make up`)")
    # Small micro-batches so the day spans several batches (exercises cross-batch dedup/commits).
    return deep_merge(config, {"stream": {"raw": {"max_offsets_per_trigger": 10_000}}})


@pytest.fixture
def topic(cfg):
    name = f"it.readings.{uuid.uuid4().hex[:8]}"
    kafka_io.create_topic(cfg["kafka"], name, partitions=3)
    yield name
    kafka_io.delete_topic(cfg["kafka"], name)


def test_one_day_lands_in_the_lake_exactly_once(cfg, topic, tmp_path: Path, monkeypatch):
    day = battledim.load_scada(2019).loc[DAY]  # load before pointing the data dir at tmp_path

    stats = ReplayStats()
    producer = kafka_io.make_producer(cfg["kafka"], on_error=delivery_error_counter(stats))
    Replayer(producer, topic, Pacer(None), progress_every_steps=0, stats=stats).run(day)
    assert stats.readings == 288 * 119 and stats.delivery_errors == 0

    def copy_of(i: int) -> bytes:
        ts = day.index[i].strftime("%Y-%m-%dT%H:%M:%SZ")
        value = day["pressure__n54"].iloc[i]
        return (
            f'{{"ts":"{ts}","sensor_id":"n54","sensor_type":"pressure","value":{value},'
            f'"unit":"m","source":"battledim","scenario_id":null,"year":2019}}'
        ).encode()

    # A repeat of the day's last reading arrives within the 1 h watermark -> removed by dedup.
    producer.produce(topic, value=copy_of(-1), key=b"pressure__n54")
    # A repeat of the first reading arrives ~24 h of event time late -> removed as late data.
    producer.produce(topic, value=copy_of(0), key=b"pressure__n54")
    producer.produce(topic, value=b'{"ts": "garbage"', key=b"broken")
    assert producer.flush(30) == 0

    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    spark = get_spark(cfg["spark"], "leakwatch-it", with_kafka=True, master="local[2]")
    summary = run(
        cfg, topic=topic, raw_paths=RawIngestPaths.default(), available_now=True, spark=spark
    )

    assert summary["batches_with_data"] >= 3
    assert summary["kafka_records_read"] == 288 * 119 + 3
    assert summary["rows_written"] == 288 * 119
    assert summary["rejects"] == 1
    assert summary["duplicates_dropped"] == 1
    assert summary["late_rows_dropped_by_watermark"] == 1
    assert summary["out_of_range_flagged"] == 0

    con = lake.connect()
    df = lake.query_df(con, "SELECT * FROM raw_readings")
    RAW_LAKE_SCHEMA.validate(df[list(RAW_LAKE_SCHEMA.columns)])
    assert len(df) == 288 * 119
    assert not df.duplicated(["sensor_type", "sensor_id", "ts"]).any()

    # Every value equals the source table.
    df["column"] = [
        sensor_column(t, s) for t, s in zip(df["sensor_type"], df["sensor_id"], strict=True)
    ]
    back = df.pivot(index="ts", columns="column", values="value").sort_index()
    back.index = back.index.rename("ts")
    pd.testing.assert_frame_equal(
        back[day.columns], day, check_names=False, check_freq=False, check_index_type=False
    )

    # Partition layout follows PRD §6.2.
    raw = tmp_path / "lake" / "raw"
    assert (
        raw / "source=battledim" / "year=2019" / f"date={DAY}" / "sensor_type=pressure"
    ).is_dir()
    assert any((tmp_path / "lake" / "_rejects" / "raw").rglob("*.json"))
