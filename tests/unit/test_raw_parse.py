"""Spark parsing of scada.readings messages, on static DataFrames shaped like Kafka records."""

from __future__ import annotations

from datetime import UTC, datetime

import orjson
import pytest
from pyspark.sql.types import (
    BinaryType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from leakwatch.common.config import load_config
from leakwatch.stream.raw_ingest import parse_readings

pytestmark = pytest.mark.spark

KAFKA_SCHEMA = StructType(
    [
        StructField("key", BinaryType()),
        StructField("value", BinaryType()),
        StructField("topic", StringType()),
        StructField("partition", IntegerType()),
        StructField("offset", LongType()),
        StructField("timestamp", TimestampType()),
        StructField("timestampType", IntegerType()),
    ]
)
RANGES = load_config(environ={})["stream"]["ranges"]
KAFKA_TS = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


def msg(**overrides) -> bytes:
    base = {
        "ts": "2019-03-01T23:55:00Z",
        "sensor_id": "n54",
        "sensor_type": "pressure",
        "value": 41.25,
        "unit": "m",
        "source": "battledim",
        "scenario_id": None,
        "year": 2019,
    }
    return orjson.dumps({**base, **overrides})


def kafka_df(spark, payloads: list[bytes]):
    rows = [
        (b"k", p, "scada.readings", 0, i, KAFKA_TS.replace(tzinfo=None), 0)
        for i, p in enumerate(payloads)
    ]
    return spark.createDataFrame(rows, KAFKA_SCHEMA)


def test_valid_message_is_parsed_in_utc(spark):
    out = parse_readings(kafka_df(spark, [msg()]), RANGES).collect()[0]
    assert out["is_valid"] is True
    assert out["sensor_id"] == "n54"
    assert out["value"] == pytest.approx(41.25)
    assert out["date"] == "2019-03-01"  # UTC date even at 23:55
    assert out["out_of_range"] is False
    assert out["reject_offset"] is None
    # ts is 2019-03-01 23:55 UTC: compare as epoch seconds to avoid local-time conversion
    epoch = (
        parse_readings(kafka_df(spark, [msg()]), RANGES)
        .selectExpr("cast(ts as long) as e")
        .first()["e"]
    )
    assert epoch == int(datetime(2019, 3, 1, 23, 55, tzinfo=UTC).timestamp())


@pytest.mark.parametrize(
    "bad",
    [
        b"not json",
        msg(sensor_type="temperature"),
        msg(unit="m3/h"),  # wrong unit for pressure
        msg(ts="2019-03-01 23:55"),  # not ISO 8601 with zone
        msg(year=2018),  # year disagrees with ts
        msg(source="sim", scenario_id=None),
        msg(sensor_id=""),
        orjson.dumps({"ts": "2019-03-01T00:00:00Z", "sensor_id": "n54"}),  # fields missing
    ],
)
def test_contract_violations_are_rejected_not_dropped(spark, bad):
    out = parse_readings(kafka_df(spark, [bad]), RANGES).collect()
    assert len(out) == 1
    assert out[0]["is_valid"] is False
    assert out[0]["reject_offset"] == 0  # unique key: never merged by deduplication
    assert out[0]["raw_json"] is not None


def test_nan_value_is_rejected(spark):
    payload = msg().replace(b"41.25", b"NaN")
    out = parse_readings(kafka_df(spark, [payload]), RANGES).collect()[0]
    assert out["is_valid"] is False


@pytest.mark.parametrize(
    ("sensor_type", "unit", "value", "flag"),
    [
        ("pressure", "m", -0.5, True),
        ("pressure", "m", 100.0, False),  # bounds are inclusive
        ("pressure", "m", 100.01, True),
        ("flow", "m3/h", 205.4, False),
        ("amr", "L/h", 37795.88, False),
        ("level", "m", 12.0, True),
    ],
)
def test_out_of_range_flag(spark, sensor_type, unit, value, flag):
    payload = msg(sensor_type=sensor_type, unit=unit, value=value)
    out = parse_readings(kafka_df(spark, [payload]), RANGES).collect()[0]
    assert out["is_valid"] is True  # flagged, still a valid reading
    assert out["out_of_range"] is flag


def test_sim_reading_with_scenario_is_valid(spark):
    out = parse_readings(kafka_df(spark, [msg(source="sim", scenario_id="s-1")]), RANGES).first()
    assert out["is_valid"] is True and out["scenario_id"] == "s-1"
