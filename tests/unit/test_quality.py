from __future__ import annotations

import numpy as np
import pandas as pd

from leakwatch.data import quality


def frame(values: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(values.values())))
    idx = pd.date_range("2019-01-01", periods=n, freq="5min", tz="UTC", name="ts")
    return pd.DataFrame(values, index=idx)


def test_constant_runs_basic():
    s = frame({"a": [1, 1, 1, 2, 2, np.nan, np.nan, 2]})["a"]
    runs = quality.constant_runs(s)
    assert list(runs["length"]) == [3, 2, 1]  # NaN breaks runs and never forms one
    assert list(runs["value"]) == [1.0, 2.0, 2.0]
    assert runs["start"].iloc[0] == s.index[0]
    assert runs["end"].iloc[0] == s.index[2]


def test_constant_runs_empty():
    assert quality.constant_runs(pd.Series([], dtype=float)).empty


def test_stuck_summary_flags_runs_at_threshold():
    df = frame({"stuck": [5.0] * 12 + [6.0], "live": list(np.arange(13.0))})
    out = quality.stuck_summary(df, min_steps=12)
    assert out.loc["stuck", "stuck_runs"] == 1
    assert out.loc["stuck", "stuck_steps"] == 12
    assert out.loc["stuck", "longest_run"] == 12
    assert out.loc["live", "stuck_runs"] == 0
    assert quality.stuck_summary(df, min_steps=13).loc["stuck", "stuck_runs"] == 0


def test_forward_fill_only_short_gaps():
    df = frame({"a": [1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, 8]})
    out = quality.forward_fill_short_gaps(df, max_gap=2)
    assert list(out["a"].iloc[:4]) == [1, 1, 1, 4]  # gap of 2 filled
    assert out["a"].iloc[4:7].isna().all()  # gap of 3 left alone
    assert out["a"].iloc[7] == 8


def test_forward_fill_leading_gap_stays_nan():
    df = frame({"a": [np.nan, 2.0, 3.0]})
    assert np.isnan(quality.forward_fill_short_gaps(df)["a"].iloc[0])
