"""Scaling scenarios (D-023): reproducible draws that respect the configured ranges."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from leakwatch.common.config import load_config
from leakwatch.sim.scenarios import ScenarioConfig, leak_table, make_scenarios

PIPES = [f"p{i}" for i in range(1, 51)]


@pytest.fixture(scope="module")
def scale_cfg():
    return load_config(environ={})["sim"]["scale"]


def test_same_seed_same_scenarios_and_different_seed_differs(scale_cfg):
    a = make_scenarios(20, scale_cfg, PIPES, seed=7)
    b = make_scenarios(20, scale_cfg, PIPES, seed=7)
    c = make_scenarios(20, scale_cfg, PIPES, seed=8)
    assert [s.to_json() for s in a] == [s.to_json() for s in b]
    assert [s.to_json() for s in a] != [s.to_json() for s in c]


def test_scenarios_follow_each_other_in_event_time(scale_cfg):
    days = int(scale_cfg["scenario_days"])
    sc = make_scenarios(5, scale_cfg, PIPES, seed=1)
    starts = [pd.Timestamp(s.start) for s in sc]
    assert starts[0] == pd.Timestamp(scale_cfg["start"], tz="UTC")
    assert all(b - a == pd.Timedelta(days=days) for a, b in itertools.pairwise(starts))
    assert len({s.scenario_id for s in sc}) == 5 and len({s.seed for s in sc}) == 5
    assert all(s.steps == days * 288 for s in sc)


def test_leaks_respect_the_configured_ranges(scale_cfg):
    days = int(scale_cfg["scenario_days"])
    lo, hi = scale_cfg["leaks_per_scenario"]
    dmin, dmax = scale_cfg["leak_diameter_m"]
    sc = make_scenarios(200, scale_cfg, PIPES, seed=3)
    counts = [len(s.leaks) for s in sc]
    assert min(counts) >= lo and max(counts) <= hi
    leaks = [leak for s in sc for leak in s.leaks]
    assert {leak.type for leak in leaks} == {"abrupt", "incipient"}
    for leak in leaks:
        assert leak.pipe_id in PIPES
        assert dmin <= leak.diameter_m <= dmax
        assert 48 <= leak.start_h <= (days - 3) * 24  # never in the first 2 or last 3 days
        assert leak.start_h <= leak.peak_h <= leak.end_h <= days * 24
        if leak.type == "abrupt":
            assert leak.peak_h == leak.start_h


def test_json_round_trip_reproduces_the_config(scale_cfg):
    for s in make_scenarios(10, scale_cfg, PIPES, seed=5):
        assert ScenarioConfig.from_json(s.to_json()) == s


def test_leak_table_has_absolute_times(scale_cfg):
    sc = make_scenarios(30, scale_cfg, PIPES, seed=11)
    table = leak_table(sc)
    assert len(table) == sum(len(s.leaks) for s in sc)
    first = next(s for s in sc if s.leaks)
    row = table[table["scenario_id"] == first.scenario_id].iloc[0]
    assert row["leak_id"] == f"{first.scenario_id}-L0"
    assert row["start_ts"] == pd.Timestamp(first.start) + pd.Timedelta(hours=first.leaks[0].start_h)
    assert (table["end_ts"] >= table["start_ts"]).all()
