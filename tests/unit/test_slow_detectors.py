"""Slow detectors for small leaks: robust scale, CUSUM, night statistics (D-048)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.bdt.slow_detectors import cusum, mad_scale, night_residual, night_values


def test_mad_scale_is_the_standard_deviation_for_normal_data():
    x = np.random.default_rng(0).normal(0, 3.0, 20_000)
    assert mad_scale(x) == pytest.approx(3.0, rel=0.03)
    cols = np.stack([x, 2 * x], axis=1)
    np.testing.assert_allclose(mad_scale(cols, axis=0), [3.0, 6.0], rtol=0.03)
    assert mad_scale(np.array([1.0, np.nan, 1.0, 1.0])) == 0.0


def test_cusum_accumulates_a_step_and_resets_at_zero():
    z = np.array([0.0, 0.0, 2.0, 2.0, 2.0, -10.0, 0.4])
    np.testing.assert_allclose(cusum(z, 0.5), [0, 0, 1.5, 3.0, 4.5, 0, 0])
    two = cusum(np.array([[1.0, -1.0], [1.0, np.nan]]), 0.5)
    np.testing.assert_allclose(two, [[0.5, 0.0], [1.0, 0.0]])  # per column; NaN counts as 0


def test_night_values_and_residual():
    ws = pd.Series(pd.date_range("2018-11-01", periods=4 * 24 * 4, freq="15min", tz="UTC"))
    v = np.where(ws.dt.hour.between(1, 3), 10.0, 99.0) + np.repeat([0.0, 0.0, 0.0, 6.0], 96)
    nights = night_values(ws, v, [1, 4])
    np.testing.assert_allclose(nights.to_numpy(), [10, 10, 10, 16])
    r = night_residual(nights, 2)
    assert np.isnan(r.iloc[0]) and r.iloc[3] == pytest.approx(6.0)  # minus the previous nights
