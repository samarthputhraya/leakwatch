"""Replay SCADA readings into Kafka topic ``scada.readings`` in timestamp order (PRD §7.1).

Input is a wide table (UTC index, ``<sensor_type>__<sensor_id>`` columns): a BattLeDIM year from
``make data`` or simulator output. Each non-missing value becomes one JSON message that matches
``schemas.Reading``; the message key is the column name, so every sensor's readings land in one
partition and stay in order.

Speed: ``--speed 360`` replays 360 simulated seconds per wall second (one 5-minute step every
0.83 s); ``--speed max`` publishes as fast as Kafka accepts.

Usage:
    python -m leakwatch.ingest.replayer --year 2018 --speed max
    python -m leakwatch.ingest.replayer --year 2019 --start 2019-01-14 --end 2019-01-17 --speed 360
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
from confluent_kafka import KafkaError, Producer

from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.common.schemas import UNITS, split_sensor_column
from leakwatch.data import battledim
from leakwatch.ingest.kafka_io import make_producer, wait_for_kafka

log = get_logger("leakwatch.ingest.replayer")


def parse_speed(text: str | float | None) -> float | None:
    """``"max"``/``None`` -> ``None`` (no pacing); ``"360"``, ``"360x"`` or ``360`` -> ``360.0``."""
    if text is None:
        return None
    if isinstance(text, int | float):
        speed = float(text)
    else:
        raw = text.strip().lower()
        if raw in ("max", "inf", ""):
            return None
        speed = float(raw.removesuffix("x"))
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"speed must be positive, got {text!r}")
    return speed


@dataclass
class Pacer:
    """Keeps simulated time and wall time in a fixed ratio (``speed``); ``None`` = no waiting."""

    speed: float | None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _wall0: float | None = field(default=None, init=False)
    _sim0: pd.Timestamp | None = field(default=None, init=False)

    def wait_for(self, ts: pd.Timestamp) -> float:
        """Sleep until ``ts`` is due; returns the seconds slept."""
        if self.speed is None:
            return 0.0
        if self._wall0 is None or self._sim0 is None:
            self._wall0, self._sim0 = self.clock(), ts
            return 0.0
        due = self._wall0 + (ts - self._sim0).total_seconds() / self.speed
        delay = due - self.clock()
        if delay > 0:
            self.sleep(delay)
            return delay
        return 0.0

    def set_speed(self, speed: float | None, now_ts: pd.Timestamp) -> None:
        """Change speed mid-replay without a jump: re-anchor at the current simulated time."""
        self.speed = speed
        self._wall0, self._sim0 = (self.clock(), now_ts) if speed is not None else (None, None)


@dataclass(frozen=True)
class SensorMeta:
    """Static part of every message for one wide-table column."""

    key: bytes
    sensor_id: str
    sensor_type: str
    unit: str


def sensor_meta(columns: list[str]) -> list[SensorMeta]:
    """Per-column message metadata, in column order."""
    metas = []
    for col in columns:
        sensor_type, sensor_id = split_sensor_column(col)
        metas.append(SensorMeta(col.encode(), sensor_id, sensor_type, UNITS[sensor_type]))
    return metas


def encode_step(
    ts_iso: str,
    year: int,
    values: np.ndarray,
    metas: list[SensorMeta],
    source: str,
    scenario_id: str | None,
) -> Iterator[tuple[bytes, bytes]]:
    """Yield ``(key, json)`` for every non-missing reading of one time step, in column order."""
    for meta, value in zip(metas, values.tolist(), strict=True):
        if value is None or value != value:  # NaN: a missing reading is not published
            continue
        payload = {
            "ts": ts_iso,
            "sensor_id": meta.sensor_id,
            "sensor_type": meta.sensor_type,
            "value": value,
            "unit": meta.unit,
            "source": source,
            "scenario_id": scenario_id,
            "year": year,
        }
        yield meta.key, orjson.dumps(payload)


@dataclass
class ReplayStats:
    """What a replay did; becomes the metrics file."""

    steps: int = 0
    readings: int = 0
    delivery_errors: int = 0
    slept_s: float = 0.0
    wall_s: float = 0.0
    first_ts: str | None = None
    last_ts: str | None = None

    @property
    def readings_per_s(self) -> float:
        """Published readings per wall-clock second, including the final flush."""
        return self.readings / self.wall_s if self.wall_s > 0 else float("nan")


def delivery_error_counter(stats: ReplayStats) -> Callable[[KafkaError | None, Any], None]:
    """Producer callback that counts failed deliveries (librdkafka calls it only for errors)."""

    def on_delivery(err: KafkaError | None, msg: Any) -> None:
        if err is not None:
            stats.delivery_errors += 1
            if stats.delivery_errors <= 5:
                log.error("delivery failed: %s", err)

    return on_delivery


class Replayer:
    """Publishes a wide table to Kafka in timestamp order with optional pacing."""

    def __init__(
        self,
        producer: Producer,
        topic: str,
        pacer: Pacer,
        source: str = "battledim",
        scenario_id: str | None = None,
        progress_every_steps: int = 8640,
        stats: ReplayStats | None = None,
    ) -> None:
        self.producer = producer
        self.topic = topic
        self.pacer = pacer
        self.source = source
        self.scenario_id = scenario_id
        self.progress_every_steps = progress_every_steps
        self.stats = stats if stats is not None else ReplayStats()

    def _produce(self, key: bytes, value: bytes) -> None:
        while True:
            try:
                self.producer.produce(self.topic, value=value, key=key)
                return
            except BufferError:
                # Local queue full: let librdkafka send, then retry.
                self.producer.poll(0.05)

    def run(self, wide: pd.DataFrame, flush_timeout_s: float = 120.0) -> ReplayStats:
        """Publish every row of ``wide`` (sorted by time). Returns the stats."""
        wide = wide.sort_index()
        metas = sensor_meta(list(wide.columns))
        values = wide.to_numpy(dtype="float64")
        index = wide.index
        ts_iso = index.strftime("%Y-%m-%dT%H:%M:%SZ")
        years = index.year.to_numpy()
        t0 = time.perf_counter()
        for i in range(len(index)):
            self.stats.slept_s += self.pacer.wait_for(index[i])
            for key, payload in encode_step(
                ts_iso[i], int(years[i]), values[i], metas, self.source, self.scenario_id
            ):
                self._produce(key, payload)
                self.stats.readings += 1
            self.producer.poll(0)
            self.stats.steps += 1
            if self.progress_every_steps and self.stats.steps % self.progress_every_steps == 0:
                rate = self.stats.readings / (time.perf_counter() - t0)
                log.info(
                    "replayed up to %s (%d readings, %.0f/s)", ts_iso[i], self.stats.readings, rate
                )
        remaining = self.producer.flush(flush_timeout_s)
        self.stats.wall_s = time.perf_counter() - t0
        if remaining:
            self.stats.delivery_errors += remaining
            log.error("%d messages still undelivered after flush", remaining)
        if len(index):
            self.stats.first_ts, self.stats.last_ts = str(ts_iso[0]), str(ts_iso[-1])
        return self.stats


def load_input(
    year: int | None, input_path: Path | None, start: str | None, end: str | None
) -> pd.DataFrame:
    """Wide table to replay: a BattLeDIM year or a Parquet file, optionally sliced by date."""
    if input_path is not None:
        wide = pd.read_parquet(input_path)
    elif year is not None:
        wide = battledim.load_scada(year)
    else:
        raise ValueError("give --year or --input")
    if start or end:
        wide = wide.loc[start:end]
    if wide.empty:
        raise ValueError("nothing to replay in the selected range")
    return wide


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; writes ``reports/metrics/<metrics-name>.json``."""
    parser = argparse.ArgumentParser(description="Replay SCADA readings into Kafka.")
    parser.add_argument("--year", type=int, help="BattLeDIM year (2018 or 2019)")
    parser.add_argument("--input", type=Path, help="wide Parquet file instead of a BattLeDIM year")
    parser.add_argument("--start", help="first date/time to replay (inclusive), e.g. 2019-01-14")
    parser.add_argument("--end", help="last date/time to replay (inclusive), e.g. 2019-01-17")
    parser.add_argument("--speed", help="simulated seconds per wall second, or 'max'")
    parser.add_argument("--topic", help="override the readings topic")
    parser.add_argument("--source", default="battledim", choices=["battledim", "sim"])
    parser.add_argument("--scenario-id", help="required for --source sim")
    parser.add_argument("--metrics-name", help="metrics file name (default replay_<source>_<year>)")
    args = parser.parse_args(argv)

    cfg = load_config()
    kafka_cfg = cfg["kafka"]
    topic = args.topic or kafka_cfg["topics"]["readings"]["name"]
    speed = parse_speed(args.speed if args.speed is not None else str(cfg["replay"]["speed"]))
    wide = load_input(args.year, args.input, args.start, args.end)
    wait_for_kafka(kafka_cfg)

    stats = ReplayStats()
    replayer = Replayer(
        producer=make_producer(kafka_cfg, on_error=delivery_error_counter(stats)),
        topic=topic,
        pacer=Pacer(speed),
        source=args.source,
        scenario_id=args.scenario_id,
        progress_every_steps=int(cfg["replay"]["progress_every_steps"]),
        stats=stats,
    )
    log.info(
        "replaying %d steps x %d sensors to %s at speed %s",
        len(wide),
        wide.shape[1],
        topic,
        "max" if speed is None else f"{speed:g}x",
    )
    replayer.run(wide)
    log.info(
        "done: %d readings in %.1f s (%.0f readings/s), %d delivery errors",
        stats.readings,
        stats.wall_s,
        stats.readings_per_s,
        stats.delivery_errors,
    )
    label = args.year if args.year is not None else (args.input.stem if args.input else "input")
    name = args.metrics_name or f"replay_{args.source}_{label}"
    path = write_metrics(
        name,
        {
            "topic": topic,
            "source": args.source,
            "scenario_id": args.scenario_id,
            "range": {"start": stats.first_ts, "end": stats.last_ts},
            "speed": "max" if speed is None else speed,
            "steps": stats.steps,
            "sensors": int(wide.shape[1]),
            "readings": stats.readings,
            "delivery_errors": stats.delivery_errors,
            "wall_s": round(stats.wall_s, 2),
            "slept_s": round(stats.slept_s, 2),
            "readings_per_s": round(stats.readings_per_s, 1),
            "target_readings_per_s": 10_000,
            "meets_target": bool(stats.readings_per_s >= 10_000) if speed is None else None,
            "producer": {k: v for k, v in kafka_cfg["producer"].items()},
            "partitions": int(kafka_cfg["topics"]["readings"]["partitions"]),
        },
    )
    log.info("wrote %s", path)
    return 0 if stats.delivery_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
