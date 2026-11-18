"""Spark cleaning (features.clean) against the pandas reference in data.quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from leakwatch.data import battledim, quality
from leakwatch.features.clean import CLEAN_COLUMNS, clean_readings

pytestmark = pytest.mark.spark


def to_pandas(df) -> pd.DataFrame:
    out = df.withColumn("t", F.col("ts").cast("long")).drop("ts").toPandas()
    out["ts"] = pd.to_datetime(out["t"], unit="s", utc=True)
    out["column"] = out["sensor_type"] + "__" + out["sensor_id"]
    return out.drop(columns="t")


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2018-03-01", periods=200, freq="5min", tz="UTC", name="ts")
    df = pd.DataFrame(
        {
            "pressure__n1": np.round(rng.normal(40, 0.3, 200), 1),
            "pressure__n2": np.repeat(rng.normal(35, 1, 20), 10),
            "flow__p1": np.repeat([0.0, 44.07], 100),
        },
        index=idx,
    )
    df.iloc[150:163, 0] = 41.7  # 13 equal steps: stuck from the 12th
    df.iloc[50:65, 1] = 33.0  # 15 equal steps
    df.iloc[[0, 10, 20, 21, 30, 31, 32], 0] = np.nan  # leading, 1-, 2-, 3-step gaps
    df.iloc[[100, 101], 1] = np.nan
    df.iloc[[150], 2] = np.nan
    return df


@pytest.fixture(scope="module")
def cleaned(wide, to_spark, small_spec) -> pd.DataFrame:
    long = battledim.to_long(wide)
    return to_pandas(clean_readings(to_spark(long), small_spec))


def test_columns(cleaned):
    assert list(cleaned.columns[: len(CLEAN_COLUMNS) - 1]) == [
        c for c in CLEAN_COLUMNS if c != "ts"
    ]


def test_values_match_reference_forward_fill(wide, cleaned):
    got = cleaned.pivot(index="ts", columns="column", values="value").reindex(
        index=wide.index, columns=wide.columns
    )
    expected = quality.forward_fill_short_gaps(wide, max_gap=2)
    pd.testing.assert_frame_equal(got, expected, check_names=False, check_freq=False)


def test_filled_flag_marks_exactly_the_filled_steps(wide, cleaned):
    expected = quality.forward_fill_short_gaps(wide, max_gap=2)
    should_fill = wide.isna() & expected.notna()
    got = cleaned.pivot(index="ts", columns="column", values="filled").reindex(
        index=wide.index, columns=wide.columns
    )
    assert int(should_fill.to_numpy().sum()) == 1 + 2 + 2 + 1  # n1: 1+2, n2: 2, flow: 1
    for col in wide.columns:
        filled = got[col].eq(True)
        assert filled.equals(should_fill[col].rename(filled.name)), col


def test_stuck_flags_match_causal_reference(wide, cleaned):
    got = cleaned.pivot(index="ts", columns="column", values="stuck").reindex(
        index=wide.index, columns=wide.columns
    )
    for col in ("pressure__n1", "pressure__n2"):
        expected = quality.stuck_flags(wide[col], min_steps=12)
        observed = wide[col].notna()
        assert got.loc[observed, col].astype(bool).tolist() == expected[observed].tolist(), col
    assert got["pressure__n1"].eq(True).sum() == 2
    assert got["pressure__n2"].eq(True).sum() == 4
    # flow is exempt even though it sits at 0 / 44.07 for 100 steps
    assert not got["flow__p1"].eq(True).any()


def test_trailing_gap_is_not_filled(to_spark, small_spec):
    df = pd.DataFrame(
        {
            "sensor_type": "pressure",
            "sensor_id": "n1",
            "ts": pd.date_range("2018-01-01", periods=3, freq="5min", tz="UTC"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    out = to_pandas(clean_readings(to_spark(df), small_spec))
    assert len(out) == 3 and not out["filled"].any()


def test_out_of_range_flag_is_carried_and_copied_to_fills(to_spark, small_spec):
    df = pd.DataFrame(
        {
            "sensor_type": "pressure",
            "sensor_id": "n1",
            "ts": pd.to_datetime(["2018-01-01 00:00", "2018-01-01 00:10"], utc=True),
            "value": [500.0, 40.0],
            "out_of_range": [True, False],
        }
    )
    out = to_pandas(clean_readings(to_spark(df), small_spec)).sort_values("ts")
    assert out["out_of_range"].tolist() == [True, True, False]
    assert out["filled"].tolist() == [False, True, False]
