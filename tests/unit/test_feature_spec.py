from __future__ import annotations

import dataclasses

import pytest

from leakwatch.common.config import load_config
from leakwatch.features.spec import (
    FeatureSpec,
    amr_sum_series,
    feature_column,
    window_label,
)
from leakwatch.features.windows import ceil_to, complete_window_range, floor_to

SENSORS = {
    "pressure": ["n10", "n2", "n1"],
    "flow": ["p227", "PUMP_1"],
    "level": ["T1"],
    "amr": ["n1", "n5", "n9"],
}
AREAS = {"n1": "C", "n5": "C", "n9": "A", "n2": "A", "n10": "B"}


def spec(**overrides) -> FeatureSpec:
    base = FeatureSpec.from_config(load_config(environ={}), SENSORS, AREAS)
    return dataclasses.replace(base, **overrides)


def test_from_config_orders_series_naturally_and_adds_amr_sums():
    s = spec()
    assert s.sensor_series == (
        "pressure__n1",
        "pressure__n2",
        "pressure__n10",
        "flow__PUMP_1",
        "flow__p227",
        "level__T1",
    )
    assert s.area_series == ("amr_sum__A", "amr_sum__C")
    assert s.series[-2:] == ("amr_sum__A", "amr_sum__C")
    assert s.sensor_types == ("pressure", "flow", "level")


def test_defaults_match_prd():
    s = spec()
    assert (s.step_s, s.window_s, s.slide_s) == (300, 3600, 900)
    assert s.stats == ("mean", "std", "min", "max", "slope", "residual")
    assert s.steps_per_window == 12
    assert s.window_label == "1h"
    # 11 steps of history for the stuck rule (12 equal readings) dominates the 3 for filling.
    assert s.history_s == 11 * 300


def test_feature_columns_are_series_major():
    s = spec(stats=("mean", "slope"))
    cols = s.feature_columns()
    assert cols[:4] == [
        "pressure__n1__mean",
        "pressure__n1__slope",
        "pressure__n2__mean",
        "pressure__n2__slope",
    ]
    assert len(cols) == len(s.series) * 2
    assert feature_column(amr_sum_series("C"), "mean") == "amr_sum__C__mean"


@pytest.mark.parametrize(
    "overrides",
    [
        {"slide_s": 700},  # window not a multiple of slide
        {"stats": ("mean", "median")},
        {"stuck_exempt_types": ("valve",)},
        {"lateness_s": 600},  # does not cover a 2-step gap + the closing reading
        {"sensor_series": ("pressure__n1", "pressure__n1")},
    ],
)
def test_invalid_specs_are_rejected(overrides):
    with pytest.raises(ValueError):
        spec(**overrides)


def test_window_label():
    assert window_label(3600) == "1h"
    assert window_label(1800) == "30min"
    assert window_label(90) == "90s"


def test_alignment_helpers():
    assert ceil_to(901, 900) == 1800
    assert ceil_to(900, 900) == 900
    assert floor_to(1799, 900) == 900


def test_complete_window_range():
    s = spec()
    t0 = 1514764800  # 2018-01-01 00:00 UTC
    last = t0 + 2 * 3600 - 300  # two full hours of 5-minute data
    first, last_start = complete_window_range(t0, last, s)
    assert first == t0
    assert last_start == t0 + 3600  # [01:00, 02:00) is the last complete window
    # With 15 min of lateness the stream must wait for 01:55 + 15 min before emitting it.
    assert complete_window_range(t0, last, s, lateness_s=900)[1] == t0 + 2700
