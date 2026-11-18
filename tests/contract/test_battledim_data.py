"""Data contracts on the real BattLeDIM files built by `make data` (skipped when absent)."""

from __future__ import annotations

import pandas as pd
import pytest

from leakwatch.common import paths
from leakwatch.common.schemas import (
    GROUND_TRUTH_SCHEMA,
    READINGS_SCHEMA,
    split_sensor_column,
    wide_scada_schema,
)
from leakwatch.data import battledim, ground_truth, network

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module", params=battledim.YEARS)
def year_wide(request) -> tuple[int, pd.DataFrame]:
    return request.param, battledim.load_scada(request.param)


def test_wide_scada_contract(year_wide):
    year, wide = year_wide
    wide_scada_schema(expected_rows=battledim.STEPS_PER_YEAR).validate(wide)
    assert wide.index[0] == pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
    assert wide.index[-1] == pd.Timestamp(f"{year}-12-31 23:55", tz="UTC")
    assert wide.index.to_series().diff().dropna().eq(battledim.TIMESTEP).all()


def test_sensor_columns_match_official_sensor_list(year_wide):
    _, wide = year_wide
    official = network.parse_sensor_config()
    by_type: dict[str, set[str]] = {}
    for col in wide.columns:
        sensor_type, sensor_id = split_sensor_column(col)
        by_type.setdefault(sensor_type, set()).add(sensor_id)
    assert {t: set(ids) for t, ids in official.items()} == by_type
    assert {t: len(ids) for t, ids in by_type.items()} == {
        "pressure": 33,
        "flow": 3,
        "level": 1,
        "amr": 82,
    }


def test_about_12_5_million_readings_per_year(year_wide):
    _, wide = year_wide
    assert wide.notna().sum().sum() == 105_120 * 119  # 12,509,280


def test_long_format_day_matches_readings_contract():
    day = battledim.load_scada(2019).loc["2019-03-01"]
    long = battledim.to_long(day)
    READINGS_SCHEMA.validate(long)
    assert len(long) == 288 * 119


def test_ground_truth_parquet_contract():
    gt = ground_truth.load_ground_truth()
    GROUND_TRUTH_SCHEMA.validate(gt)
    assert gt["peak_outflow_m3h"].notna().all()
    assert len(ground_truth.active_in_year(gt, 2019)) == 23


def test_leak_flows_cover_exactly_the_active_leaks():
    gt = ground_truth.load_ground_truth()
    for year in battledim.YEARS:
        flows = battledim.load_leak_flows(year)
        assert set(flows.columns) == set(ground_truth.active_in_year(gt, year)["pipe_id"])
        assert len(flows) == battledim.STEPS_PER_YEAR
        assert (flows >= 0).all().all()


def test_raw_files_match_published_md5():
    raw = paths.battledim_raw_dir()
    for f in battledim.FILES:
        if f.required:
            assert battledim.md5sum(raw / f.name) == f.md5, f.name
