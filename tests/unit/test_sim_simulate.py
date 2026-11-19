"""Scenario simulation (epyt-flow) produces BattLeDIM-shaped data, and a leak shows up in it."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.needs_data]

NO_NOISE = {"pressure_m": 0.0, "level_m": 0.0, "flow_rel": 0.0, "amr_rel": 0.0}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    import os

    from leakwatch.common import paths
    from leakwatch.data import network
    from leakwatch.sim.scenarios import LeakSpec, ScenarioConfig, simulate, tank_areas

    sensors = network.parse_sensor_config()
    inp = paths.network_inp_path()
    area = tank_areas(inp, sensors["level"])
    base = ScenarioConfig("t-base", "2020-01-06T00:00:00+00:00", 1, 1, 0.0, NO_NOISE)
    leak = replace(
        base,
        scenario_id="t-leak",
        leaks=(LeakSpec("p461", "abrupt", 0.025, start_h=6.0, peak_h=6.0, end_h=24.0),),
    )
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("epanet"))  # EPANET writes scratch files to the cwd
    try:
        return sensors, simulate(base, inp, sensors, area), simulate(leak, inp, sensors, area)
    finally:
        os.chdir(cwd)


def test_output_has_battledim_columns_steps_and_rounding(runs):
    sensors, base, _ = runs
    assert base.shape[1] == sum(len(v) for v in sensors.values())
    assert {c.split("__")[0] for c in base.columns} == {"pressure", "flow", "level", "amr"}
    assert 287 <= len(base) <= 288  # one day of 5-minute steps (EPANET may stop one short)
    assert base.index[0] == pd.Timestamp("2020-01-06", tz="UTC")
    assert (base.index[1:] - base.index[:-1] == pd.Timedelta(minutes=5)).all()
    values = base.to_numpy()
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values, np.round(values, 2), atol=1e-9)


def test_a_leak_changes_nothing_before_it_starts_and_lowers_pressure_after(runs):
    _, base, leak = runs
    pressure = [c for c in base.columns if c.startswith("pressure__")]
    before = base.index < pd.Timestamp("2020-01-06 06:00", tz="UTC")
    after = base.index >= pd.Timestamp("2020-01-06 07:00", tz="UTC")
    # equal up to one step of BattLeDIM's 0.01 rounding (solver round-off flips a few roundings)
    np.testing.assert_allclose(leak.loc[before, pressure], base.loc[before, pressure], atol=0.0101)
    drop = (leak.loc[after, pressure] - base.loc[after, pressure]).mean()
    assert drop.min() < -0.05  # some sensor sees the 25 mm leak
