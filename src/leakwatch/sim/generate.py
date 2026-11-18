"""``make sim-scale``: simulate scenarios in parallel with Spark ``mapPartitions`` (PRD §4.2, §7.5).

Each Spark task simulates one scenario (``sim.scenarios``) with epyt-flow in a Python worker and
writes its wide table to ``lake/sim/scenarios/outputs/scenario_id=…/data.parquet``; the configs
and the leak ground truth go to ``lake/sim/scenarios/configs/``. Scenarios already on disk are
skipped, so an interrupted run resumes where it stopped.

These are simulator *outputs* (PRD §6.2 ``sim/scenarios/``); they reach ``lake/raw`` only through
Kafka (``make sim-replay``), like every other reading (PRD §5).

Writes ``reports/metrics/sim_generation.json``: scenarios, readings, wall time, throughput, and
per-task wall time and peak memory.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.sim.scenarios import ScenarioConfig, leak_table, make_scenarios, simulate, tank_areas

log = get_logger("leakwatch.sim.generate")


def output_path(root: Path, scenario_id: str) -> Path:
    """Where a scenario's wide table lives."""
    return root / "outputs" / f"scenario_id={scenario_id}" / "data.parquet"


def run_partition(
    rows: Iterator[tuple[str, str]],
    root: str,
    inp: str,
    sensors: dict[str, list[str]],
    tank_area: dict[str, float],
) -> Iterator[dict[str, Any]]:
    """Spark task: simulate each scenario of the partition and write it (skip if present)."""
    for scenario_id, text in rows:
        target = output_path(Path(root), scenario_id)
        if target.exists():
            n = pd.read_parquet(target).shape
            yield {"scenario_id": scenario_id, "skipped": True, "steps": n[0], "sensors": n[1]}
            continue
        config = ScenarioConfig.from_json(text)
        t0 = time.perf_counter()
        wide = simulate(config, Path(inp), sensors, tank_area)
        wall = time.perf_counter() - t0
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        wide.to_parquet(tmp, compression="zstd")
        os.replace(tmp, target)
        yield {
            "scenario_id": scenario_id,
            "skipped": False,
            "steps": int(wide.shape[0]),
            "sensors": int(wide.shape[1]),
            "wall_s": round(wall, 2),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "pid": os.getpid(),
        }


def main(argv: list[str] | None = None) -> int:
    """Generate the configured scenarios (``sim.scale``) and write the metrics."""
    parser = argparse.ArgumentParser(description="Spark-parallel scenario simulation.")
    parser.add_argument("--scenarios", type=int, help="override sim.scale.scenarios")
    parser.add_argument("--workers", type=int, help="override sim.scale.workers")
    args = parser.parse_args(argv)
    cfg = load_config()
    scale = cfg["sim"]["scale"]
    n = int(args.scenarios or scale["scenarios"])
    workers = int(args.workers or scale["workers"])

    from leakwatch.data import network
    from leakwatch.stream.spark import get_spark

    sensors = network.parse_sensor_config()
    inp = paths.network_inp_path()
    pipes = sorted(network.load_network().pipe_name_list)
    configs = make_scenarios(n, scale, pipes, int(cfg["project"]["seed"]))
    root = paths.lake_sim_dir()
    conf_dir = paths.ensure_dir(root / "configs")
    (conf_dir / "scenarios.jsonl").write_text(
        "\n".join(c.to_json() for c in configs) + "\n", encoding="utf-8"
    )
    leaks = leak_table(configs)
    leaks.to_parquet(conf_dir / "leaks.parquet")

    spark = get_spark(cfg["spark"], "leakwatch-sim-scale", master=f"local[{workers}]")
    sc = spark.sparkContext
    rdd = sc.parallelize([(c.scenario_id, c.to_json()) for c in configs], numSlices=len(configs))
    t0 = time.perf_counter()
    area = tank_areas(inp, sensors["level"])
    results = rdd.mapPartitions(
        lambda it: run_partition(it, str(root), str(inp), sensors, area)
    ).collect()
    wall = time.perf_counter() - t0
    done = [r for r in results if not r["skipped"]]
    readings = int(sum(r["steps"] * r["sensors"] for r in results))
    new_readings = int(sum(r["steps"] * r["sensors"] for r in done))
    payload = {
        "engine": "epyt-flow (EPANET)",
        "scenarios": len(results),
        "scenarios_simulated_now": len(done),
        "scenario_days": int(scale["scenario_days"]),
        "first_start": configs[0].start,
        "last_end": (
            pd.Timestamp(configs[-1].start) + pd.Timedelta(days=configs[-1].days)
        ).isoformat(),
        "readings": readings,
        "readings_simulated_now": new_readings,
        "leaks": len(leaks),
        "workers": workers,
        "wall_s": round(wall, 1),
        "readings_per_s": round(new_readings / wall, 1) if done else None,
        "task_wall_s": {
            "mean": round(sum(r["wall_s"] for r in done) / len(done), 1) if done else None,
            "max": max((r["wall_s"] for r in done), default=None),
        },
        "peak_rss_mb_per_worker_max": max((r["peak_rss_mb"] for r in done), default=None),
        "config": json.loads(json.dumps(scale)),
    }
    path = write_metrics("sim_generation", payload)
    log.info(
        "%d scenarios (%d simulated now), %d readings in %.0f s -> %s",
        len(results),
        len(done),
        readings,
        wall,
        path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
