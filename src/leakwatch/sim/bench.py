"""``make sim-bench``: time one leak scenario on L-Town with each simulator (PRD §4.2, Phase 0).

For each engine and each duration (default 48 h and 7 days at 5-minute steps) we simulate the
nominal L-Town model with one abrupt pipe leak and record wall time, peak memory and throughput.
Each run happens in a fresh process so peak memory is per run.

Engines:
- ``wntr``: WNTRSimulator, pressure-dependent demand, leak modelled on a junction inserted
  in the middle of the pipe (``wntr.morph.split_pipe`` + ``add_leak``), as PRD §4.2 describes.
- ``epyt_flow``: EPANET-based ScenarioSimulator with its native ``AbruptLeakage`` on the pipe.

Throughput is expressed as "readings per second": simulated time steps x 119 BattLeDIM sensors
per wall-clock second, so it is directly comparable with the 100M-reading target.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import resource
import sys
import time
from typing import Any

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics

log = get_logger("leakwatch.sim.bench")  # explicit: __name__ is "__main__" under `python -m`

BATTLEDIM_SENSORS = 119  # 33 pressure + 3 flow + 1 level + 82 AMR
TARGET_READINGS = 100_000_000  # PRD §4.2 full-scale lake target


def _peak_rss_mb() -> float:
    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_wntr(params: dict[str, Any]) -> dict[str, Any]:
    """Simulate one leak with WNTRSimulator (PDD). Runs inside a child process."""
    import wntr

    wn = wntr.network.WaterNetworkModel(params["inp"])
    wn.options.time.duration = params["hours"] * 3600
    wn.options.time.hydraulic_timestep = params["step_s"]
    wn.options.time.report_timestep = params["step_s"]
    wn.options.hydraulic.demand_model = "PDD"
    pipe = params["leak_pipe"]
    leak_node = f"{pipe}_leak"
    wn = wntr.morph.split_pipe(wn, pipe, f"{pipe}_b", leak_node)
    area = math.pi * (params["leak_diameter_m"] / 2) ** 2
    wn.get_node(leak_node).add_leak(
        wn,
        area=area,
        discharge_coeff=params["discharge_coeff"],
        start_time=params["leak_start_hour"] * 3600,
    )
    t0 = time.perf_counter()
    results = wntr.sim.WNTRSimulator(wn).run_sim()
    elapsed = time.perf_counter() - t0
    pressure = results.node["pressure"]
    leak_m3h = float(results.node["leak_demand"][leak_node].iloc[-1]) * 3600
    return {
        "wall_s": elapsed,
        "timesteps": len(pressure),
        "leak_outflow_end_m3h": leak_m3h,
        "peak_rss_mb": _peak_rss_mb(),
    }


def run_epyt_flow(params: dict[str, Any]) -> dict[str, Any]:
    """Simulate one leak with epyt-flow (EPANET). Runs inside a child process."""
    from epyt_flow.simulation import ScenarioSimulator
    from epyt_flow.simulation.events import AbruptLeakage

    duration_s = params["hours"] * 3600
    with ScenarioSimulator(f_inp_in=params["inp"]) as sim:
        sim.set_general_parameters(
            simulation_duration=duration_s,
            hydraulic_time_step=params["step_s"],
            reporting_time_step=params["step_s"],
        )
        sim.set_pressure_sensors(params["sensors"]["pressure"])
        sim.set_flow_sensors(params["sensors"]["flow"])
        sim.set_tank_sensors(params["sensors"]["level"])
        sim.set_demand_sensors(params["sensors"]["amr"])
        sim.add_leakage(
            AbruptLeakage(
                link_id=params["leak_pipe"],
                diameter=float(params["leak_diameter_m"]),
                start_time=params["leak_start_hour"] * 3600,
                end_time=duration_s,
            )
        )
        t0 = time.perf_counter()
        scada = sim.run_simulation()
        elapsed = time.perf_counter() - t0
    return {
        "wall_s": elapsed,
        "timesteps": int(scada.get_data().shape[0]),
        "leak_outflow_end_m3h": None,  # epyt-flow does not expose the emitter flow directly
        "peak_rss_mb": _peak_rss_mb(),
    }


ENGINES = {"wntr": run_wntr, "epyt_flow": run_epyt_flow}


def run_isolated(engine: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run one benchmark in a fresh spawned process and return its measurements."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(ENGINES[engine], (params,))


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Add derived throughput numbers to one raw measurement."""
    readings = result["timesteps"] * BATTLEDIM_SENSORS
    per_s = readings / result["wall_s"]
    return {
        **result,
        "wall_s": round(result["wall_s"], 2),
        "peak_rss_mb": round(result["peak_rss_mb"], 1),
        "equivalent_readings": readings,
        "readings_per_s": round(per_s, 1),
        "simulated_days_per_wall_minute": round(
            result["timesteps"] * 300 / 86400 / (result["wall_s"] / 60), 2
        ),
        "projected_single_core_hours_for_100M": round(TARGET_READINGS / per_s / 3600, 2),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark grid and write ``reports/metrics/sim_benchmark.json``."""
    parser = argparse.ArgumentParser(description="Time leak simulations with WNTR and epyt-flow.")
    parser.add_argument("--engines", nargs="*", help="subset of engines to run")
    args = parser.parse_args(argv)

    from leakwatch.data.network import parse_sensor_config

    cfg = load_config()["sim"]["bench"]
    inp = paths.network_inp_path()
    if not inp.exists():
        log.error("%s missing: run `make data` first", inp)
        return 1
    engines = args.engines or cfg["engines"]
    base = {
        "inp": str(inp),
        "step_s": int(cfg["hydraulic_timestep_s"]),
        "leak_pipe": cfg["leak_pipe"],
        "leak_diameter_m": float(cfg["leak_diameter_m"]),
        "leak_start_hour": int(cfg["leak_start_hour"]),
        "discharge_coeff": float(cfg["discharge_coeff"]),
        "sensors": parse_sensor_config(),
    }
    runs = []
    for engine in engines:
        for hours in cfg["durations_hours"]:
            for rep in range(int(cfg["repeats"])):
                log.info("running %s, %d h (repeat %d)", engine, hours, rep + 1)
                raw = run_isolated(engine, {**base, "hours": int(hours)})
                row = {"engine": engine, "hours": int(hours), "repeat": rep + 1, **summarise(raw)}
                log.info(
                    "  %s %d h: %.1f s, %d steps, %.0f readings/s, peak %.0f MB",
                    engine,
                    hours,
                    row["wall_s"],
                    row["timesteps"],
                    row["readings_per_s"],
                    row["peak_rss_mb"],
                )
                runs.append(row)

    best: dict[str, float] = {}
    for row in runs:
        best[row["engine"]] = max(best.get(row["engine"], 0.0), row["readings_per_s"])
    path = write_metrics(
        "sim_benchmark",
        {
            "scenario": {k: v for k, v in base.items() if k not in ("inp", "sensors")},
            "machine": {"cpu_count": os.cpu_count()},
            "runs": runs,
            "fastest_engine": max(best, key=best.__getitem__),
            "best_readings_per_s": best,
        },
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
