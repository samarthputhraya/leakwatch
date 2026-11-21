"""Exact values of the onset contrast and the chance level (kills mutation survivors, D-049)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.eval.detection import chance_detection, onset_contrast

T0 = pd.Timestamp("2019-03-01 00:00", tz="UTC")
START, END = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-12-31 23:55", tz="UTC")


def gt(*leaks: tuple[str, str, pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"leak_id": i, "type": k, "start_ts": t} for i, k, t in leaks],
        columns=["leak_id", "type", "start_ts"],
    )


def scores(values: dict[pd.Timestamp, float]) -> pd.DataFrame:
    return pd.DataFrame({"window_end": list(values), "score": list(values.values())})


def test_onset_windows_bounds_and_values():
    h = pd.Timedelta(hours=1)
    s = scores(
        {
            T0 - 2 * h: 1.0,  # outside the 2 h before-window: (start - 2h, start]
            T0 - h: 2.0,
            T0: 4.0,  # the start itself counts as before
            T0 + h: 100.0,  # the first hour after the start is skipped
            T0 + 1.5 * h: 9.0,
            T0 + 2 * h: 12.0,  # the end of the after-window is included
            T0 + 3 * h: 50.0,  # beyond it
        }
    )
    out = onset_contrast(s, gt(("L1", "abrupt", T0)), START, END, hours=2.0)
    row = out["per_leak"][0]
    assert row["before_mean"] == pytest.approx(3.0) and row["after_mean"] == pytest.approx(10.5)
    assert row["ratio"] == pytest.approx(3.5)
    assert row["after_above_before_p95"] is True  # 10.5 > p95 of [2, 4] = 3.9
    assert out["hours"] == 2.0 and out["leaks"] == 1
    assert out["share_rising_overall"] == 1.0 and out["share_rising_abrupt"] == 1.0
    assert out["share_rising_incipient"] is None
    assert out["median_ratio_abrupt"] == pytest.approx(3.5)


def test_onset_zero_before_mean_and_types_and_period():
    h = pd.Timedelta(hours=1)
    t1 = T0 + pd.Timedelta(days=3)
    s = scores({T0: 0.0, T0 + 2 * h: 1.0, t1: 5.0, t1 + 2 * h: 1.0})
    leaks = gt(
        ("L1", "incipient", T0),
        ("L2", "abrupt", t1),
        ("L3", "abrupt", pd.Timestamp("2018-06-01", tz="UTC")),  # before the period
    )
    out = onset_contrast(s, leaks, START, END, hours=2.0)
    by = {r["leak_id"]: r for r in out["per_leak"]}
    assert set(by) == {"L1", "L2"}
    assert by["L1"]["ratio"] is None  # before mean 0: no ratio
    assert (
        by["L1"]["after_above_before_p95"] is True and by["L2"]["after_above_before_p95"] is False
    )
    assert out["share_rising_incipient"] == 1.0 and out["share_rising_abrupt"] == 0.0
    assert out["share_rising_overall"] == 0.5
    assert out["median_ratio_abrupt"] == pytest.approx(0.2)


def test_onset_skips_leaks_without_scores_on_both_sides():
    s = scores({T0: 1.0})
    out = onset_contrast(s, gt(("L1", "abrupt", T0)), START, END, hours=2.0)
    assert out["leaks"] == 0 and out["share_rising_overall"] is None
    assert out["median_ratio_abrupt"] is None and out["per_leak"] == []


def test_chance_detection_inclusive_horizon_and_seed():
    one = gt(("L1", "abrupt", T0))
    # an alert exactly at the leak start (lower bound) and exactly at start + horizon count
    period_start, period_end = T0, T0 + pd.Timedelta(days=7)
    r = chance_detection(50, one, period_start, period_end, horizon_days=7.0, n_iter=50, seed=3)
    assert r["mean_rate"] == 1.0 and r["p95_rate"] == 1.0 and r["n_alerts"] == 50
    assert r["iterations"] == 50
    # one alert in a long period: rate is the chance it falls in the week after the start
    long_end = T0 + pd.Timedelta(days=70)
    got = chance_detection(1, one, T0, long_end, horizon_days=7.0, n_iter=4000, seed=1)
    assert got["mean_rate"] == pytest.approx(0.1, abs=0.02)
    assert got["p95_rate"] == 1.0
    rates = [
        chance_detection(3, one, T0, long_end, n_iter=100, seed=s)["mean_rate"] for s in (1, 1)
    ]
    assert rates[0] == rates[1]
    assert np.isclose(
        chance_detection(3, one, T0, long_end, n_iter=2000, seed=2)["mean_rate"],
        1 - (1 - 0.1) ** 3,
        atol=0.03,
    )
