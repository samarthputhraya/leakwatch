"""Contracts on the real cleaned layer, baselines and feature tables (`make features`).

Phase 2 Definition of Done: feature tables for 2018 and 2019. Besides counts and schemas, one
window per year is recomputed independently with numpy/pandas from the source SCADA files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.common import lake
from leakwatch.common.schemas import (
    BASELINES_SCHEMA,
    CLEAN_LAKE_SCHEMA,
    feature_table_schema,
    sensor_column,
)
from leakwatch.data import battledim, quality
from leakwatch.features.spec import load_feature_spec

pytestmark = [pytest.mark.needs_features, pytest.mark.needs_lake, pytest.mark.needs_data]

#: Windows that start in each year and are fully covered by data (D-017): 2018 ends with the
#: window starting 23:45 (it uses 2019 readings); 2019 with 23:00 (no 2020 data).
EXPECTED_WINDOWS = {2018: 365 * 96, 2019: 365 * 96 - 3}
LAST_WINDOW = {2018: "2018-12-31 23:45", 2019: "2019-12-31 23:00"}


@pytest.fixture(scope="module")
def con():
    return lake.connect()


@pytest.fixture(scope="module")
def spec():
    return load_feature_spec()


@pytest.fixture(scope="module", params=[2018, 2019])
def year(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def wide(year) -> pd.DataFrame:
    return battledim.load_scada(year)


# ---------------------------------------------------------------------------------- clean


def test_clean_layer_has_every_raw_reading(con, year):
    raw = con.execute(
        f"SELECT count(*) FROM raw_readings WHERE source = 'battledim' AND year = {year}"
    ).fetchone()[0]
    clean, filled = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE filled) FROM clean_readings "
        f"WHERE source = 'battledim' AND year = {year}"
    ).fetchone()
    assert filled == 0  # BattLeDIM has no missing readings
    assert clean == raw == 12_509_280


def test_clean_stuck_flags_match_reference(con, year, wide, spec):
    got = dict(
        con.execute(
            f"""SELECT sensor_type || '__' || sensor_id, count(*) FROM clean_readings
                WHERE source = 'battledim' AND year = {year} AND stuck GROUP BY 1"""
        ).fetchall()
    )
    exempt = spec.stuck_exempt_types
    expected = {}
    for col in wide.columns:
        if col.split("__")[0] in exempt:
            continue
        n = int(quality.stuck_flags(wide[col], spec.stuck_min_steps).sum())
        if n:
            expected[col] = n
    # The year's first readings have no history from the year before in the reference.
    assert got.keys() == expected.keys()
    for col, n in expected.items():
        assert abs(got[col] - n) <= spec.stuck_min_steps - 1, col
    assert not any(k.split("__")[0] in exempt for k in got)


def test_clean_rows_satisfy_contract(con, year):
    df = lake.query_df(
        con,
        f"SELECT * FROM clean_readings WHERE source = 'battledim' AND date = '{year}-02-10'",
    )
    CLEAN_LAKE_SCHEMA.validate(df[list(CLEAN_LAKE_SCHEMA.columns)])
    assert len(df) == 288 * 119


# ------------------------------------------------------------------------------ baselines


def test_baselines_cover_every_series_and_hour(con, spec):
    df = lake.query_df(con, "SELECT * FROM baselines")
    BASELINES_SCHEMA.validate(df)
    assert set(df["series_id"]) == set(spec.series)
    assert len(df) == len(spec.series) * 168


# ------------------------------------------------------------------------------- features


def test_one_row_per_window(con, year):
    n, distinct, first, last = con.execute(
        f"""SELECT count(*), count(DISTINCT window_start), min(window_start), max(window_start)
            FROM features WHERE source = 'battledim' AND year(window_start) = {year}"""
    ).fetchone()
    assert n == distinct == EXPECTED_WINDOWS[year]
    assert pd.Timestamp(first) == pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
    assert pd.Timestamp(last) == pd.Timestamp(LAST_WINDOW[year], tz="UTC")


def test_every_window_is_complete_and_has_no_nulls(con, year, spec):
    full = spec.steps_per_window * len(spec.series)
    nulls = " + ".join(f'count(*) FILTER (WHERE "{c}" IS NULL)' for c in spec.feature_columns())
    row = con.execute(
        f"""SELECT min(n_readings), max(n_readings), max(n_series_missing), sum(n_filled),
                   {nulls}
            FROM features WHERE source = 'battledim' AND year(window_start) = {year}"""
    ).fetchone()
    assert row[:4] == (full, full, 0, 0)
    assert row[4] == 0


def test_feature_rows_satisfy_contract(con, year, spec):
    df = lake.query_df(
        con, f"SELECT * FROM features WHERE source = 'battledim' AND date = '{year}-07-04'"
    )
    schema = feature_table_schema(spec.feature_columns())
    schema.validate(df[list(schema.columns)])
    assert len(df) == 96


def test_one_window_matches_numpy(con, year, wide):
    start = pd.Timestamp(f"{year}-03-05 14:15", tz="UTC")  # 2018: abrupt leak p673 starts 15:45
    win = wide.loc[start : start + pd.Timedelta(minutes=55)]
    assert len(win) == 12
    row = lake.query_df(
        con,
        f"SELECT * FROM features WHERE source = 'battledim' AND window_start = '{start}'",
    ).iloc[0]
    base = lake.query_df(con, "SELECT * FROM baselines").set_index(["series_id", "hour_of_week"])
    hours = (win.index - win.index[0]).total_seconds().to_numpy() / 3600
    how = win.index.dayofweek * 24 + win.index.hour
    amr = win[[c for c in win.columns if c.startswith("amr__")]].sum(axis=1)
    checks = {
        sensor_column("pressure", "n54"): win["pressure__n54"],
        sensor_column("flow", "p227"): win["flow__p227"],
        sensor_column("level", "T1"): win["level__T1"],
        "amr_sum__C": amr,
    }
    for series_id, values in checks.items():
        v = values.to_numpy()
        baseline = base.loc[[(series_id, h) for h in how], "baseline"].to_numpy()
        expected = {
            "mean": v.mean(),
            "std": v.std(),
            "min": v.min(),
            "max": v.max(),
            "slope": np.polyfit(hours, v, 1)[0],
            "residual": (v - baseline).mean(),
        }
        for stat, value in expected.items():
            assert row[f"{series_id}__{stat}"] == pytest.approx(value, rel=1e-8, abs=1e-8), (
                series_id,
                stat,
            )
