"""Simulated L-Town scenarios for the scaling story (PRD §4.2, Phase 4; D-007, D-023).

A scenario is ``days`` of 5-minute readings from the nominal L-Town model, simulated with
epyt-flow (EPANET), with

- ±``demand_uncertainty`` multiplicative noise on the demand patterns (the nominal model carries
  about ±10% parameter uncertainty, PRD §4.2),
- 0–2 random leaks (abrupt or incipient, BattLeDIM-like diameters, random pipe and start),
- Gaussian sensor noise and BattLeDIM's 0.01 rounding,

returned as a wide table with BattLeDIM's columns and units (pressure m, flow m³/h, tank level m,
AMR L/h), so the replayer, the lake and the feature code treat it exactly like benchmark data.
Scenarios follow each other in event time (scenario k starts ``k × days`` after ``start``): the
streaming pipeline needs event time to move forward (D-013).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.common.schemas import sensor_column

STEP_S = 300


@dataclass(frozen=True)
class LeakSpec:
    """One simulated leak (times in hours from the scenario start)."""

    pipe_id: str
    type: str  # abrupt | incipient
    diameter_m: float
    start_h: float
    peak_h: float  # incipient: time the leak reaches full size (= start_h for abrupt)
    end_h: float  # repair time (= scenario end if never repaired)


@dataclass(frozen=True)
class ScenarioConfig:
    """Everything needed to reproduce one scenario."""

    scenario_id: str
    start: str  # ISO timestamp (UTC) of the first step
    days: int
    seed: int
    demand_uncertainty: float
    noise: Mapping[str, float]
    leaks: tuple[LeakSpec, ...] = field(default_factory=tuple)

    def to_json(self) -> str:
        """JSON form (Spark passes configs to workers as strings)."""
        d = asdict(self)
        d["leaks"] = [asdict(x) for x in self.leaks]
        return json.dumps(d)

    @classmethod
    def from_json(cls, text: str) -> ScenarioConfig:
        """Inverse of :meth:`to_json`."""
        d = json.loads(text)
        d["leaks"] = tuple(LeakSpec(**x) for x in d["leaks"])
        return cls(**d)

    @property
    def steps(self) -> int:
        """Time steps in the scenario."""
        return self.days * 86400 // STEP_S


def make_scenarios(
    n: int, cfg: Mapping[str, Any], pipes: Sequence[str], seed: int, prefix: str = "scale"
) -> list[ScenarioConfig]:
    """Draw ``n`` consecutive scenarios (``sim.scale`` config section)."""
    rng = np.random.default_rng(seed)
    days = int(cfg["scenario_days"])
    t0 = pd.Timestamp(cfg["start"], tz="UTC")
    lo, hi = (int(v) for v in cfg["leaks_per_scenario"])
    dmin, dmax = (float(v) for v in cfg["leak_diameter_m"])
    out = []
    for k in range(n):
        leaks = []
        for _ in range(int(rng.integers(lo, hi + 1))):
            start_h = float(rng.uniform(2 * 24, (days - 3) * 24))
            incipient = bool(rng.random() < float(cfg["incipient_share"]))
            peak_h = (
                min(start_h + float(rng.uniform(3, 15)) * 24, days * 24.0) if incipient else start_h
            )
            repaired = bool(rng.random() < 0.3)
            end_h = (
                min(start_h + float(rng.uniform(4, 20)) * 24, days * 24.0)
                if repaired
                else days * 24.0
            )
            leaks.append(
                LeakSpec(
                    pipe_id=str(rng.choice(pipes)),
                    type="incipient" if incipient else "abrupt",
                    diameter_m=round(
                        float(math.exp(rng.uniform(math.log(dmin), math.log(dmax)))), 5
                    ),
                    start_h=round(start_h, 3),
                    peak_h=round(max(peak_h, start_h), 3),
                    end_h=round(max(end_h, peak_h, start_h + 1), 3),
                )
            )
        out.append(
            ScenarioConfig(
                scenario_id=f"{prefix}-{k:04d}",
                start=(t0 + pd.Timedelta(days=days * k)).isoformat(),
                days=days,
                seed=int(seed * 100_003 + k),
                demand_uncertainty=float(cfg["demand_uncertainty"]),
                noise=dict(cfg["noise"]),
                leaks=tuple(leaks),
            )
        )
    return out


def tank_areas(inp_path: Path, tanks: Sequence[str]) -> dict[str, float]:
    """Cross-section area (m²) of each tank, from the model (level = volume / area)."""
    import wntr

    wn = wntr.network.WaterNetworkModel(str(inp_path))
    return {t: math.pi * (float(wn.get_node(t).diameter) / 2) ** 2 for t in tanks}


def simulate(
    config: ScenarioConfig,
    inp_path: Path,
    sensors: Mapping[str, Sequence[str]],
    tank_area: Mapping[str, float],
) -> pd.DataFrame:
    """Run one scenario with epyt-flow; wide table in BattLeDIM columns/units (see module doc)."""
    from epyt_flow.simulation import ScenarioSimulator
    from epyt_flow.simulation.events import AbruptLeakage, IncipientLeakage
    from epyt_flow.uncertainty import ModelUncertainty, RelativeUniformUncertainty

    duration = config.days * 86400
    with ScenarioSimulator(f_inp_in=str(inp_path)) as sim:
        sim.set_general_parameters(
            simulation_duration=duration - STEP_S,
            hydraulic_time_step=STEP_S,
            reporting_time_step=STEP_S,
        )
        sim.set_pressure_sensors(list(sensors["pressure"]))
        sim.set_flow_sensors(list(sensors["flow"]))
        sim.set_tank_sensors(list(sensors["level"]))
        sim.set_demand_sensors(list(sensors["amr"]))
        u = float(config.demand_uncertainty)
        if u > 0:
            sim.set_model_uncertainty(
                ModelUncertainty(
                    global_demand_pattern_uncertainty=RelativeUniformUncertainty(
                        low=1 - u, high=1 + u
                    ),
                    seed=config.seed,
                )
            )
        for leak in config.leaks:
            kwargs = {
                "link_id": leak.pipe_id,
                "diameter": leak.diameter_m,
                "start_time": int(leak.start_h * 3600) // STEP_S * STEP_S,
                "end_time": int(leak.end_h * 3600) // STEP_S * STEP_S,
            }
            if leak.type == "incipient":
                sim.add_leakage(
                    IncipientLeakage(peak_time=int(leak.peak_h * 3600) // STEP_S * STEP_S, **kwargs)
                )
            else:
                sim.add_leakage(AbruptLeakage(**kwargs))
        scada = sim.run_simulation()
    index = pd.date_range(config.start, periods=config.steps, freq=f"{STEP_S}s", name="ts")
    parts = {
        "pressure": scada.get_data_pressures(),
        "flow": scada.get_data_flows(),
        "level": scada.get_data_tanks_water_volume()
        / np.array([tank_area[t] for t in sensors["level"]]),
        "amr": scada.get_data_demands() * 1000.0,  # m3/h -> L/h
    }
    rng = np.random.default_rng(config.seed + 1)
    noise = config.noise
    cols, blocks = [], []
    for sensor_type, values in parts.items():
        values = np.asarray(values, dtype="float64")[: config.steps]
        if sensor_type == "pressure":
            values = values + rng.normal(0, float(noise["pressure_m"]), values.shape)
        elif sensor_type == "level":
            values = values + rng.normal(0, float(noise["level_m"]), values.shape)
        else:
            rel = float(noise["flow_rel"] if sensor_type == "flow" else noise["amr_rel"])
            values = values * (1 + rng.normal(0, rel, values.shape))
            values = np.clip(values, 0.0, None)
        blocks.append(np.round(values, 2))
        cols += [sensor_column(sensor_type, s) for s in sensors[sensor_type]]
    data = np.hstack(blocks)
    if len(data) < len(index):  # EPANET may stop one step short; never pad with invented data
        index = index[: len(data)]
    return pd.DataFrame(data, index=index, columns=cols)


def leak_table(configs: Sequence[ScenarioConfig]) -> pd.DataFrame:
    """Ground truth of all simulated leaks (absolute UTC times)."""
    rows = []
    for c in configs:
        t0 = pd.Timestamp(c.start)
        for i, leak in enumerate(c.leaks):
            rows.append(
                {
                    "scenario_id": c.scenario_id,
                    "leak_id": f"{c.scenario_id}-L{i}",
                    "pipe_id": leak.pipe_id,
                    "type": leak.type,
                    "diameter_m": leak.diameter_m,
                    "start_ts": t0 + pd.Timedelta(hours=leak.start_h),
                    "peak_ts": t0 + pd.Timedelta(hours=leak.peak_h),
                    "end_ts": t0 + pd.Timedelta(hours=leak.end_h),
                }
            )
    return pd.DataFrame(rows)
