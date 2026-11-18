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


def test_stuck_flags_are_causal():
    values = [1.0] * 14 + [2.0] + [3.0] * 11 + [np.nan] + [3.0] * 12
    s = frame({"a": values})["a"]
    flags = quality.stuck_flags(s, min_steps=12)
    # first run (14 long): flagged from its 12th step on
    assert not flags.iloc[:11].any()
    assert flags.iloc[11:14].all()
    # 11 equal values, then NaN breaks the run; the 12 after it reach the threshold at the end
    assert not flags.iloc[14:38].any()
    assert flags.iloc[38]
    assert flags.sum() == 3 + 1
    assert flags.index.equals(s.index)


def test_stuck_flags_agree_with_stuck_summary():
    rng = np.random.default_rng(0)
    values = np.repeat(rng.integers(0, 3, 40).astype(float), rng.integers(1, 20, 40))
    s = pd.Series(values, index=pd.date_range("2019-01-01", periods=len(values), freq="5min"))
    flags = quality.stuck_flags(s, min_steps=12)
    runs = quality.constant_runs(s)
    long_runs = runs[runs["length"] >= 12]
    assert flags.sum() == int((long_runs["length"] - 11).sum())


def test_forward_fill_only_short_gaps():
    df = frame({"a": [1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, 8]})
    out = quality.forward_fill_short_gaps(df, max_gap=2)
    assert list(out["a"].iloc[:4]) == [1, 1, 1, 4]  # gap of 2 filled
    assert out["a"].iloc[4:7].isna().all()  # gap of 3 left alone
    assert out["a"].iloc[7] == 8


def test_forward_fill_leading_gap_stays_nan():
    df = frame({"a": [np.nan, 2.0, 3.0]})
    assert np.isnan(quality.forward_fill_short_gaps(df)["a"].iloc[0])
