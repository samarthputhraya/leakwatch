"""Replay generated scenarios through Kafka in event-time order (PRD §5: Kafka is the only entry).

Scenarios are consecutive in event time (``sim.scenarios``), so publishing them one after another
keeps event time moving forward for the raw ingest's watermark (D-013). Each reading carries
``source="sim"`` and its ``scenario_id``. Run with the raw ingest alongside
(``scripts/sim_ingest.sh`` = ``make sim-ingest``).

Writes ``reports/metrics/replay_sim_scale.json``.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.ingest import kafka_io
from leakwatch.ingest.replayer import Pacer, Replayer, ReplayStats, delivery_error_counter
from leakwatch.sim.generate import output_path
from leakwatch.sim.scenarios import ScenarioConfig

log = get_logger("leakwatch.sim.replay")


def main(argv: list[str] | None = None) -> int:
    """Publish every generated scenario (oldest first) to ``scada.readings``."""
    parser = argparse.ArgumentParser(description="Replay simulated scenarios into Kafka.")
    parser.add_argument("--limit", type=int, help="only the first N scenarios")
    args = parser.parse_args(argv)
    cfg = load_config()
    kafka_cfg = cfg["kafka"]
    kafka_io.wait_for_kafka(kafka_cfg)
    root = paths.lake_sim_dir()
    lines = (root / "configs" / "scenarios.jsonl").read_text(encoding="utf-8").splitlines()
    configs = sorted(
        (ScenarioConfig.from_json(x) for x in lines if x.strip()), key=lambda c: c.start
    )
    if args.limit:
        configs = configs[: args.limit]
    topic = kafka_cfg["topics"]["readings"]["name"]
    stats = ReplayStats()
    producer = kafka_io.make_producer(kafka_cfg, on_error=delivery_error_counter(stats))
    t0 = time.perf_counter()
    for i, c in enumerate(configs):
        path = output_path(root, c.scenario_id)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `make sim-scale` first")
        wide = pd.read_parquet(path)
        Replayer(
            producer,
            topic,
            Pacer(None),
            source="sim",
            scenario_id=c.scenario_id,
            progress_every_steps=0,
            stats=stats,
        ).run(wide)
        log.info("%d/%d %s: %d readings so far", i + 1, len(configs), c.scenario_id, stats.readings)
    wall = time.perf_counter() - t0
    path = write_metrics(
        "replay_sim_scale",
        {
            "topic": topic,
            "source": "sim",
            "scenarios": len(configs),
            "readings": stats.readings,
            "delivery_errors": stats.delivery_errors,
            "wall_s": round(wall, 1),
            "readings_per_s": round(stats.readings / wall, 1) if wall else None,
            "range": {
                "start": configs[0].start,
                "end_of_last": str(
                    pd.Timestamp(configs[-1].start) + pd.Timedelta(days=configs[-1].days)
                ),
            },
        },
    )
    log.info("replayed %d readings in %.0f s -> %s", stats.readings, wall, path)
    return 0 if stats.delivery_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
