"""Contracts on the real raw lake built by `make ingest-year YEAR=2018` (skipped when absent).

Phase 1 Definition of Done: the full 2018 year is in the lake, exactly once, with every value
equal to the source data.
"""

from __future__ import annotations

import pytest

from leakwatch.common import lake
from leakwatch.common.schemas import RAW_LAKE_SCHEMA, sensor_column
from leakwatch.data import battledim

pytestmark = [pytest.mark.needs_lake, pytest.mark.needs_data]

YEAR = 2018
WHERE = f"source = 'battledim' AND year = {YEAR}"


@pytest.fixture(scope="module")
def con():
    return lake.connect()


def test_full_year_present_exactly_once(con):
    rows, distinct = con.execute(
        "SELECT count(*), count(DISTINCT (sensor_type, sensor_id, ts)) "
        f"FROM raw_readings WHERE {WHERE}"
    ).fetchone()
    assert rows == battledim.STEPS_PER_YEAR * 119 == 12_509_280
    assert distinct == rows


def test_every_sensor_matches_source(con):
    lake_stats = con.execute(
        f"""SELECT sensor_type, sensor_id, count(*) AS n, sum(value) AS s,
                   min(ts) AS first_ts, max(ts) AS last_ts
            FROM raw_readings WHERE {WHERE} GROUP BY ALL"""
    ).df()
    lake_stats["column"] = [
        sensor_column(t, s)
        for t, s in zip(lake_stats["sensor_type"], lake_stats["sensor_id"], strict=True)
    ]
    lake_stats = lake_stats.set_index("column")
    wide = battledim.load_scada(YEAR)
    assert set(lake_stats.index) == set(wide.columns)
    for col in wide.columns:
        assert lake_stats.at[col, "n"] == wide[col].notna().sum(), col
        assert lake_stats.at[col, "s"] == pytest.approx(wide[col].sum(), rel=1e-9), col


def test_partitions_match_timestamps(con):
    bad = con.execute(
        f"""SELECT count(*) FROM raw_readings WHERE {WHERE}
            AND (strftime(ts, '%Y-%m-%d') <> date OR year(ts) <> year)"""
    ).fetchone()[0]
    assert bad == 0
    dates = con.execute(f"SELECT count(DISTINCT date) FROM raw_readings WHERE {WHERE}").fetchone()[
        0
    ]
    assert dates == 365


def test_one_day_satisfies_row_contract(con):
    df = lake.query_df(con, f"SELECT * FROM raw_readings WHERE {WHERE} AND date = '{YEAR}-06-15'")
    RAW_LAKE_SCHEMA.validate(df[list(RAW_LAKE_SCHEMA.columns)])
    assert len(df) == 288 * 119


def test_no_benchmark_reading_is_out_of_range(con):
    flagged = con.execute(
        f"SELECT count(*) FROM raw_readings WHERE {WHERE} AND out_of_range"
    ).fetchone()[0]
    assert flagged == 0
