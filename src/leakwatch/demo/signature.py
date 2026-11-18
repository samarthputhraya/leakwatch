"""Leak signatures for the live demo (PRD §16, D-025).

A demo leak is injected onto replayed BattLeDIM data: two EPANET runs of the nominal L-Town model
from the same time of week — one with an emitter leak at the clicked junction, one without — and
their difference per sensor (pressure m, flow m³/h, tank level m) is added to the real readings.
The detector, baselines and localizer then see real background behaviour plus a physically
simulated leak, instead of a fully simulated network they were never trained on.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from leakwatch.common import paths
from leakwatch.common.schemas import sensor_column

G = 9.81
STEP_S = 300
PATTERN_ORIGIN = pd.Timestamp("2018-01-01", tz="UTC")  # L-Town demand patterns start on a Monday


def _prefix() -> str:
    """Unique EPANET scratch-file prefix (WNTR writes ``<prefix>.inp/.bin/.rpt``)."""
    import uuid

    return str(paths.private_scratch_dir() / f"sig-{uuid.uuid4().hex[:8]}")


@lru_cache(maxsize=1)
def _template(inp: str):  # -> wntr.network.WaterNetworkModel
    import wntr

    wn = wntr.network.WaterNetworkModel(inp)
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.emitter_exponent = 0.5
    wn.options.time.hydraulic_timestep = STEP_S
    wn.options.time.report_timestep = STEP_S
    return wn


def _run(
    inp: str,
    start: pd.Timestamp,
    hours: float,
    sensors: Mapping[str, Sequence[str]],
    leak: tuple[str, float] | None,
    discharge_coeff: float,
) -> pd.DataFrame:
    import wntr

    wn = copy.deepcopy(_template(inp))
    wn.options.time.duration = int(hours * 3600)
    week = 7 * 86400
    wn.options.time.pattern_start = int((start - PATTERN_ORIGIN).total_seconds()) % week
    if leak is not None:
        node, diameter = leak
        area = math.pi * (diameter / 2) ** 2
        wn.get_node(node).emitter_coefficient = discharge_coeff * area * math.sqrt(2 * G)
    prefix = _prefix()
    try:
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)
    finally:
        for ext in (".inp", ".bin", ".rpt"):
            Path(prefix + ext).unlink(missing_ok=True)
    cols = {}
    for s in sensors.get("pressure", []):
        cols[sensor_column("pressure", s)] = res.node["pressure"][s]
    for s in sensors.get("flow", []):
        cols[sensor_column("flow", s)] = res.link["flowrate"][s] * 3600.0  # m3/s -> m3/h
    for s in sensors.get("level", []):
        cols[sensor_column("level", s)] = res.node["pressure"][s]  # tank pressure = water level
    return pd.DataFrame(cols)


def leak_signature(
    node: str,
    diameter_m: float,
    start: pd.Timestamp,
    hours: float = 4.0,
    sensors: Mapping[str, Sequence[str]] | None = None,
    inp: Path | None = None,
    discharge_coeff: float = 0.75,
) -> tuple[pd.DataFrame, float]:
    """Per-step sensor deltas (leak − no leak) from ``start`` for ``hours``, and the leak flow.

    Returns the deltas indexed by step number (0 = ``start``) with wide-table column names, and
    the leak's outflow in m³/h estimated as the increase of the two reservoir inflows.
    """
    from leakwatch.data import network

    sensors = sensors or network.parse_sensor_config()
    inp_s = str(inp or paths.network_inp_path())
    base = _run(inp_s, start, hours, sensors, None, discharge_coeff)
    leak = _run(inp_s, start, hours, sensors, (node, diameter_m), discharge_coeff)
    delta = (leak - base).reset_index(drop=True)
    inflow = [c for c in ("flow__p227", "flow__p235") if c in delta]
    flow = float(delta[inflow].sum(axis=1).iloc[-1]) if inflow else float("nan")
    return delta.astype("float64"), flow


def apply_signature(
    values: np.ndarray, columns: Sequence[str], delta: pd.DataFrame, step: int
) -> np.ndarray:
    """Add the leak's delta at ``step`` (last value held afterwards) to one row of readings."""
    if step < 0:
        return values
    row = delta.iloc[min(step, len(delta) - 1)]
    out = values.copy()
    for j, col in enumerate(columns):
        if col in row.index:
            out[j] = out[j] + float(row[col])
    return out
