"""Re-baselined residuals (D-038)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.features.rebaseline import Rebaseline, rebaselined

RB = Rebaseline(window_days=2, gap_days=1)


def frame(values: np.ndarray, start: str = "2018-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(values), freq="15min", tz="UTC")
    return pd.DataFrame({"window_start": ts, "a": values, "b": values * 2})


def test_step_change_shows_then_is_absorbed():
    n_day = 96
    v = np.zeros(8 * n_day)
    v[4 * n_day :] = -3.0  # a leak on day 4
    out = rebaselined(frame(v), ["a", "b"], RB)
    a = out["a"].to_numpy()
    assert np.isnan(a[: n_day + n_day // 2]).all()  # reference needs half its window first
    assert np.allclose(a[2 * n_day : 4 * n_day], 0.0)
    assert np.allclose(a[4 * n_day : 5 * n_day], -3.0)  # visible for the gap (1 day) and more
    assert np.allclose(a[7 * n_day :], 0.0)  # absorbed once most of the reference has it
    assert np.allclose(out["b"].to_numpy()[4 * n_day], -6.0)


def test_row_order_and_index_are_kept():
    rng = np.random.default_rng(0)
    f = frame(rng.normal(size=5 * 96)).set_axis(range(1000, 1000 + 5 * 96))
    shuffled = f.sample(frac=1.0, random_state=1)
    out = rebaselined(shuffled, ["a"], RB)
    assert list(out.index) == list(shuffled.index)
    expect = rebaselined(f, ["a"], RB).loc[shuffled.index, "a"]
    assert np.allclose(out["a"], expect, equal_nan=True)


def test_gaps_are_handled_by_time():
    v = np.ones(6 * 96)
    f = frame(v)
    f = f.drop(index=range(3 * 96, 3 * 96 + 40))  # 10 h of missing windows
    out = rebaselined(f, ["a"], RB)
    assert np.allclose(out["a"].dropna(), 0.0)


def test_from_config():
    rb = Rebaseline.from_config({"features": {"rebaseline": {"window_days": 7, "gap_days": 1}}})
    assert rb.history == pd.Timedelta(days=8)
    assert rebaselined(frame(np.array([])), ["a"], rb).empty


@pytest.mark.parametrize("gap", [0.5, 1.0])
def test_reference_never_includes_the_window_itself(gap):
    v = np.zeros(4 * 96)
    v[-1] = 10.0
    out = rebaselined(frame(v), ["a"], Rebaseline(window_days=1, gap_days=gap))
    assert out["a"].iloc[-1] == pytest.approx(10.0)
