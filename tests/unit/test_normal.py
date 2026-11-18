"""D-015: the exact rule for "normal" timestamps (baselines, K-Means training)."""

from __future__ import annotations

import pandas as pd
import pytest

from leakwatch.common.config import load_config
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.features import normal


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def gt_frame(rows: list[dict]) -> pd.DataFrame:
    base = {"pipe_id": "p1", "peak_ts": ts("2018-01-01")}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True)
    return df


GT = gt_frame(
    [
        # big incipient leak, repaired: excluded start..repair
        {
            "leak_id": "big",
            "type": "incipient",
            "peak_outflow_m3h": 20.0,
            "start_ts": ts("2018-02-01"),
            "end_ts": ts("2018-03-01"),
        },
        # small incipient, never repaired: background, only the first 7 days excluded
        {
            "leak_id": "small",
            "type": "incipient",
            "peak_outflow_m3h": 6.0,
            "start_ts": ts("2018-05-01"),
            "end_ts": None,
        },
        # small abrupt: abrupt leaks are always significant
        {
            "leak_id": "abrupt",
            "type": "abrupt",
            "peak_outflow_m3h": 5.0,
            "start_ts": ts("2018-07-01"),
            "end_ts": ts("2018-07-03"),
        },
        # unknown size: treated as significant
        {
            "leak_id": "unknown",
            "type": "incipient",
            "peak_outflow_m3h": float("nan"),
            "start_ts": ts("2018-09-01"),
            "end_ts": ts("2018-09-02"),
        },
    ]
)


def test_main_rule_intervals():
    out = normal.excluded_intervals(GT, background_max_peak_m3h=10.0, onset_exclusion_days=7)
    by_id = out.set_index("leak_id")
    assert by_id.loc["big", "reason"] == "active"
    assert by_id.loc["big", "end_ts"] == ts("2018-03-01")
    assert by_id.loc["small", "reason"] == "onset"
    assert by_id.loc["small", "end_ts"] == ts("2018-05-08")
    assert by_id.loc["abrupt", "reason"] == "active"
    assert by_id.loc["unknown", "reason"] == "active"
    assert list(out["start_ts"]) == sorted(out["start_ts"])


def test_onset_is_capped_at_repair():
    gt = gt_frame(
        [
            {
                "leak_id": "short",
                "type": "incipient",
                "peak_outflow_m3h": 3.0,
                "start_ts": ts("2018-05-01"),
                "end_ts": ts("2018-05-03"),
            }
        ]
    )
    out = normal.excluded_intervals(gt, 10.0, 7)
    assert out.loc[0, "end_ts"] == ts("2018-05-03")


def test_strict_rule_excludes_every_leak_while_active():
    out = normal.excluded_intervals(GT, 10.0, 7, rule="strict")
    small = out.set_index("leak_id").loc["small"]
    assert small["reason"] == "active"
    assert small["end_ts"] == pd.Timestamp("2019-12-31 23:55", tz="UTC")


def test_is_normal_bounds_are_inclusive():
    intervals = normal.excluded_intervals(GT, 10.0, 7)
    idx = pd.DatetimeIndex(
        [
            ts("2018-01-31 23:55"),
            ts("2018-02-01 00:00"),  # big leak starts
            ts("2018-03-01 00:00"),  # big leak repaired (still excluded)
            ts("2018-03-01 00:05"),
            ts("2018-05-07 12:00"),  # small leak onset
            ts("2018-05-08 00:05"),  # small leak after onset = background
        ]
    )
    assert list(normal.is_normal(idx, intervals)) == [True, False, False, True, False, True]


def test_normal_summary_counts_stretches():
    intervals = normal.excluded_intervals(GT, 10.0, 7)
    idx = pd.date_range("2018-01-31 23:00", "2018-02-01 01:00", freq="5min", tz="UTC")
    out = normal.normal_summary(idx, intervals, step_s=300)
    assert out["normal_steps"] == 12  # 23:00 .. 23:55
    assert out["normal_stretches"] == [
        {
            "start": "2018-01-31T23:00:00+00:00",
            "end": "2018-01-31T23:55:00+00:00",
            "days": round(12 / 288, 2),
        }
    ]


@pytest.mark.spark
def test_spark_condition_matches_pandas(spark):
    from pyspark.sql import functions as F

    intervals = normal.excluded_intervals(GT, 10.0, 7)
    idx = pd.date_range("2018-01-01", "2018-12-31", freq="7h", tz="UTC")
    df = spark.createDataFrame([(int(t.timestamp()),) for t in idx], "t long").withColumn(
        "ts", F.timestamp_seconds("t")
    )
    got = [r["n"] for r in df.select(normal.normal_condition("ts", intervals).alias("n")).collect()]
    assert got == list(normal.is_normal(idx, intervals))


@pytest.mark.needs_data
def test_battledim_2018_rule():
    """The configured rule on the real ground truth: the 4 never-repaired small leaks are the
    background; every other 2018 leak is excluded while active."""
    cfg = load_config(environ={})["normal"]
    gt = load_ground_truth()
    out = normal.intervals_from_config(gt, cfg)
    y2018 = out[out["start_ts"].dt.year == 2018]
    background = set(y2018.loc[y2018["reason"] == "onset", "pipe_id"])
    assert background == {"p257", "p427", "p654", "p810"}
    assert (y2018.loc[y2018["reason"] == "active", "type"].value_counts().to_dict()) == {
        "incipient": 4,
        "abrupt": 6,
    }
    idx = pd.date_range("2018-01-01", "2018-12-31 23:55", freq="5min", tz="UTC")
    summary = normal.normal_summary(idx, out, step_s=300)
    strict = normal.normal_summary(
        idx, normal.intervals_from_config(gt, cfg, rule="strict"), step_s=300
    )
    # Far more normal time than the strict leak-free rule (~8 days, D-010), but not everything.
    assert strict["normal_days"] < 10
    assert 60 < summary["normal_days"] < 250
