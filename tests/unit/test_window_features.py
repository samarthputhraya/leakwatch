"""Windowed features and baselines (features.windows, features.baselines) on hand-checkable data."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from leakwatch.features.baselines import BASELINE_COLUMNS, compute_baselines
from leakwatch.features.clean import clean_readings
from leakwatch.features.spec import QUALITY_COLUMNS
from leakwatch.features.windows import (
    WINDOW_KEY,
    complete_window_range,
    hour_of_week,
    select_windows,
    window_features,
)

pytestmark = pytest.mark.spark

T0 = pd.Timestamp("2018-01-01 00:00", tz="UTC")  # a Monday
T0_S = int(T0.timestamp())


def rows_by_start(df) -> dict[int, dict]:
    out = df.withColumn("_s", F.col("window_start").cast("long")).collect()
    return {r["_s"]: r.asDict() for r in out}


@pytest.fixture(scope="module")
def ramp(spark, to_spark, small_spec):
    """2 hours: pressure n1 = 40 + 0.5 i; AMRs a1 = 100 and a2 = 200 + i (area C)."""
    spec = dataclasses.replace(
        small_spec, sensor_series=("pressure__n1",), amr_areas={"a1": "C", "a2": "C"}
    )
    idx = pd.date_range(T0, periods=24, freq="5min")
    i = np.arange(24)
    frames = [
        pd.DataFrame({"sensor_type": t, "sensor_id": s, "ts": idx, "value": v})
        for t, s, v in [
            ("pressure", "n1", 40 + 0.5 * i),
            ("amr", "a1", np.full(24, 100.0)),
            ("amr", "a2", 200.0 + i),
        ]
    ]
    readings = to_spark(pd.concat(frames, ignore_index=True))
    base = spark.createDataFrame(
        [("pressure__n1", 0, 40.0, 1), ("pressure__n1", 1, 41.0, 1), ("amr_sum__C", 0, 300.0, 1)],
        "series_id string, hour_of_week int, baseline double, n_readings long",
    )
    first, last = complete_window_range(T0_S, T0_S + 23 * 300, spec)
    out = select_windows(window_features(readings, base, spec), first, last)
    return spec, out, rows_by_start(out)


def test_columns_and_window_count(ramp):
    spec, out, rows = ramp
    assert out.columns == [*WINDOW_KEY, *QUALITY_COLUMNS, *spec.feature_columns()]
    assert sorted(rows) == [T0_S + k * 900 for k in range(5)]  # 00:00 ... 01:00


def test_first_window_statistics(ramp):
    _, _, rows = ramp
    r = rows[T0_S]
    assert r["window_end"] is not None
    assert r["pressure__n1__mean"] == pytest.approx(42.75)
    assert r["pressure__n1__min"] == pytest.approx(40.0)
    assert r["pressure__n1__max"] == pytest.approx(45.5)
    assert r["pressure__n1__std"] == pytest.approx(0.5 * np.arange(12).std())
    assert r["pressure__n1__slope"] == pytest.approx(6.0)  # 0.5 per 5 min = 6 per hour
    assert r["pressure__n1__residual"] == pytest.approx(2.75)  # baseline 40 in hour 0
    # AMR area sum: 300 + i
    assert r["amr_sum__C__mean"] == pytest.approx(305.5)
    assert r["amr_sum__C__slope"] == pytest.approx(12.0)
    assert r["amr_sum__C__min"] == pytest.approx(300.0)
    assert r["amr_sum__C__residual"] == pytest.approx(5.5)
    assert (r["n_readings"], r["n_filled"], r["n_stuck"], r["n_series_missing"]) == (24, 0, 0, 0)


def test_window_crossing_an_hour_uses_both_baselines(ramp):
    _, _, rows = ramp
    r = rows[T0_S + 900]  # 00:15-01:15: steps 3..14
    assert r["pressure__n1__mean"] == pytest.approx(44.25)
    # steps 3..11 against 40, steps 12..14 against 41
    assert r["pressure__n1__residual"] == pytest.approx(4.0)
    # no baseline for amr_sum__C in hour 1 -> residual averages only hour-0 steps 3..11
    assert r["amr_sum__C__residual"] == pytest.approx(np.mean(np.arange(3, 12)))


def test_toy_data_quality_counts(spark, small_spec, toy_pdf, to_spark, toy_base):
    out = window_features(to_spark(toy_pdf), toy_base, small_spec)
    rows = rows_by_start(out)
    full = small_spec.steps_per_window * len(small_spec.series)  # 12 x 6
    # 00:00-01:00: everything present
    assert rows[T0_S]["n_readings"] == full
    # 02:15-03:15 holds steps 27..38: n1 misses 30, 31 -> both filled
    w = rows[T0_S + 9 * 900]
    assert (w["n_readings"], w["n_filled"]) == (full, 2)
    # 05:00-06:00 holds steps 60..71: n1 misses 60..63 (not filled); a1 is out of range at 70
    w = rows[T0_S + 20 * 900]
    assert w["n_readings"] == full - 4
    assert w["n_out_of_range"] == 1
    a2_at_70 = (
        toy_pdf.query("sensor_id == 'a2'")
        .set_index("ts")["value"]
        .loc[T0 + pd.Timedelta(minutes=5 * 70)]
    )
    assert w["amr_sum__C__min"] == pytest.approx(a2_at_70)  # sum without the bad a1 value
    # 07:00-08:00 holds steps 84..95: n1 constant since step 80 -> stuck from step 91
    assert rows[T0_S + 28 * 900]["n_stuck"] == 5
    # n9 is not a feature series; flow p1 is exempt from the stuck rule
    assert not any(c.startswith("pressure__n9") for c in out.columns)


def test_slope_is_exact_for_large_values(small_spec, to_spark, toy_base):
    """Large values with big swings around a small trend (like the AMR area sum, ~30,000 L/h):
    the slope must match numpy to ~1e-9 absolute, so batch and stream agree whatever the order
    in which Spark aggregates the rows (first full-year parity run: 1e-6 differences)."""
    spec = dataclasses.replace(small_spec, sensor_series=("pressure__n1",), amr_areas={})
    t0 = pd.Timestamp("2029-06-01 00:00", tz="UTC")  # large epoch values
    i = np.arange(12)
    values = 30000.0 + 450.0 * (-1.0) ** i + 3.0 * i
    df = pd.DataFrame(
        {
            "sensor_type": "pressure",
            "sensor_id": "n1",
            "ts": pd.date_range(t0, periods=12, freq="5min"),
            "value": values,
        }
    )
    r = rows_by_start(window_features(to_spark(df), toy_base, spec))[int(t0.timestamp())]
    expected = np.polyfit(i / 12.0, values, 1)[0]
    assert abs(r["pressure__n1__slope"] - expected) < 1e-9


def test_missing_series_counted(spark, small_spec, to_spark, toy_base):
    df = pd.DataFrame(
        {
            "sensor_type": "pressure",
            "sensor_id": "n1",
            "ts": pd.date_range(T0, periods=12, freq="5min"),
            "value": 40.0,
        }
    )
    r = rows_by_start(window_features(to_spark(df), toy_base, small_spec))[T0_S]
    assert r["n_series_missing"] == len(small_spec.series) - 1
    assert r["pressure__n2__mean"] is None
    assert r["pressure__n1__std"] == 0.0
    assert r["pressure__n1__slope"] == pytest.approx(0.0)


def test_hour_of_week_matches_pandas(spark):
    idx = pd.date_range("2018-01-01", "2018-01-15", freq="37min", tz="UTC")
    df = spark.createDataFrame([(int(t.timestamp()),) for t in idx], "t long")
    got = [r["h"] for r in df.select(hour_of_week(F.timestamp_seconds("t")).alias("h")).collect()]
    assert got == list(idx.dayofweek * 24 + idx.hour)
    assert got[0] == 0 and max(got) == 167


def test_baselines_are_hour_of_week_medians(spark, small_spec, toy_pdf, to_spark):
    clean = clean_readings(to_spark(toy_pdf), small_spec)
    normal = F.col("ts") < F.timestamp_seconds(F.lit(T0_S + 6 * 3600))  # first 6 hours only
    base = compute_baselines(clean, small_spec, normal).toPandas()
    assert list(base.columns) == list(BASELINE_COLUMNS)
    assert set(base["series_id"]) == set(small_spec.series)
    assert set(base["hour_of_week"]) == set(range(6))
    p2 = toy_pdf.query("sensor_id == 'n2'").set_index("ts")["value"]
    p2 = p2[p2.index < T0 + pd.Timedelta(hours=6)]
    expected = p2.groupby(p2.index.hour).median()
    got = base.query("series_id == 'pressure__n2'").set_index("hour_of_week")["baseline"]
    assert got.sort_index().to_numpy() == pytest.approx(expected.to_numpy())
    assert base.query("series_id == 'pressure__n2'")["n_readings"].eq(12).all()
