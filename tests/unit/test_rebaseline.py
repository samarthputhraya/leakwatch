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


# Gaps found by mutation testing (D-040).


def test_reference_needs_exactly_half_its_window():
    n_day = 96
    out = rebaselined(frame(np.zeros(5 * n_day)), ["a"], RB)  # window 2 days, gap 1 day
    a = out["a"].to_numpy()
    # first reference: rolling median at s = 2 days (96 rows = half of 2 days), shifted by 1 day
    assert np.isnan(a[2 * n_day - 1])
    assert a[2 * n_day] == 0.0


@pytest.mark.parametrize("gap", [0.0, 1.0 / 96])
def test_reference_window_is_closed_on_the_left(gap):
    # 30-minute window: the reference of row i is the median of the two rows before i - gap
    v = np.zeros(12)
    v[6] = 10.0
    rb = Rebaseline(window_days=1.0 / 48, gap_days=gap)
    a = rebaselined(frame(v), ["a"], rb)["a"].to_numpy()
    lag = round(gap * 96)
    assert a[6] == pytest.approx(10.0)  # its own value is never in its reference
    assert a[6 + lag + 1] == pytest.approx(-5.0)  # median(0, 10) once the spike is in
    assert a[6 + lag + 3] == pytest.approx(0.0)  # and gone once it left the window


def test_min_rows_floor_and_minute_steps():
    one = rebaselined(frame(np.array([1.0, 4.0])), ["a"], Rebaseline(1.0 / 48, 0.0))
    assert np.isnan(one["a"].iloc[0]) and one["a"].iloc[1] == pytest.approx(3.0)
    ts = pd.date_range("2018-01-01", periods=12, freq="1min", tz="UTC")
    f = pd.DataFrame({"window_start": ts, "a": np.arange(12.0)})
    a = rebaselined(f, ["a"], Rebaseline(10.0 / 1440, 0.0))["a"].to_numpy()
    assert np.isnan(a[:5]).all()  # 10-minute window at 1-minute steps needs 5 rows
    assert a[5] == pytest.approx(5.0 - 2.0)


def test_short_frames_use_their_own_step():
    single = rebaselined(frame(np.array([3.0])), ["a"], RB)
    assert len(single) == 1 and np.isnan(single["a"].iloc[0])
    ts = pd.DatetimeIndex(["2018-01-01 00:00", "2018-01-01 01:00"], tz="UTC")
    f = pd.DataFrame({"window_start": ts, "a": [1.0, 4.0]})
    out = rebaselined(f, ["a"], Rebaseline(2.0 / 24, 0.0))  # 2 h window at 1 h steps
    assert out["a"].iloc[1] == pytest.approx(3.0)


def test_reference_ending_in_a_data_hole_is_missing():
    n_day = 96
    f = frame(np.ones(5 * n_day)).drop(index=range(2 * n_day, 3 * n_day))  # day 3 missing
    out = rebaselined(f, ["a"], Rebaseline(window_days=1, gap_days=1))
    day4 = out.loc[3 * n_day : 4 * n_day - 1, "a"]
    assert day4.iloc[0] == 0.0  # the last reference before the hole holds for one step
    assert day4.iloc[1:].isna().all()  # then t - gap is in the hole: nothing is carried forward
    assert np.allclose(out.loc[4 * n_day + n_day // 2 :, "a"], 0.0)


def test_duplicate_timestamps_are_tolerated():
    f = frame(np.zeros(4 * 96))
    f = pd.concat([f, f.iloc[[300]]]).reset_index(drop=True)
    out = rebaselined(f, ["a"], RB)
    assert len(out) == len(f)
    assert out["a"].iloc[-1] == out["a"].iloc[300]


def test_from_config_reads_both_fields():
    rb = Rebaseline.from_config({"features": {"rebaseline": {"window_days": 3, "gap_days": 2}}})
    assert (rb.window_days, rb.gap_days) == (3.0, 2.0)
