"""Choosing the posterior localizer's prior length and its bootstrap gains (D-042)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from leakwatch.dl.gnn_prior import (
    choose_length,
    gains,
    length_key,
    paired_bootstrap,
)


def test_length_keys():
    assert length_key(None) == "none"
    assert length_key(300.0) == "300"
    assert length_key(37.5) == "37.5"


def test_best_2018_length_wins_and_ties_go_to_the_weaker_prior():
    assert choose_length({None: 10.0, 600.0: 30.0, 300.0: 20.0}) == 600.0
    assert choose_length({None: 30.0, 600.0: 30.0, 300.0: 30.0}) is None
    assert choose_length({600.0: 30.0, 300.0: 30.0, 150.0: 5.0}) == 600.0


def test_paired_bootstrap_of_identical_columns_is_zero():
    a = [100.0, 250.0, 900.0, 40.0]
    out = paired_bootstrap(a, a, lambda v: float(np.median(v)), draws=200, seed=0)
    assert out == {"estimate": 0.0, "ci95": [0.0, 0.0], "draws": 200}


def test_paired_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1000, 30)
    b = a + 100.0  # b is always 100 m worse
    one = paired_bootstrap(b, a, lambda v: float(np.median(v)), draws=500, seed=3)
    two = paired_bootstrap(b, a, lambda v: float(np.median(v)), draws=500, seed=3)
    assert one == two
    lo, hi = one["ci95"]
    assert lo <= one["estimate"] <= hi and one["estimate"] == pytest.approx(100.0)


def test_gains_are_positive_when_the_posterior_is_closer():
    per_leak = [
        {"posterior_distance_top1_m": 100.0, "nearest_sensor_distance_top1_m": 400.0},
        {"posterior_distance_top1_m": 200.0, "nearest_sensor_distance_top1_m": 250.0},
        {"posterior_distance_top1_m": 900.0, "nearest_sensor_distance_top1_m": 1000.0},
    ]
    g = gains(per_leak, "nearest_sensor", within_m=300.0, draws=100, seed=0)
    # within 300 m: posterior 2 of 3, sensor 1 of 3
    assert g["within_m_top1_gain"]["estimate"] == pytest.approx(1 / 3, abs=1e-4)
    assert g["median_distance_gain_m"]["estimate"] == pytest.approx(200.0)
    assert gains([], "nearest_sensor", 300.0, 10, 0) == {}
