"""``make up`` / ``make kafka-reset``: wait for Kafka and create the configured topics."""

from __future__ import annotations

import argparse
import sys

from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.ingest.kafka_io import ensure_topics, wait_for_kafka

log = get_logger("leakwatch.ingest.topics")


def main(argv: list[str] | None = None) -> int:
    """Create missing topics; ``--recreate`` deletes and recreates them (drops their messages)."""
    parser = argparse.ArgumentParser(description="Create the LeakWatch Kafka topics.")
    parser.add_argument("--recreate", action="store_true", help="delete and recreate topics")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for Kafka")
    args = parser.parse_args(argv)

    kafka_cfg = load_config()["kafka"]
    wait_for_kafka(kafka_cfg, timeout_s=args.timeout)
    created = ensure_topics(kafka_cfg, recreate=args.recreate)
    names = [t["name"] for t in kafka_cfg["topics"].values()]
    log.info("topics ready: %s (created now: %s)", ", ".join(names), ", ".join(created) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
