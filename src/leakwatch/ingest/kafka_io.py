"""Kafka plumbing shared by the replayer, simulator and tests: producer, topics, readiness."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from confluent_kafka import KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from leakwatch.common.logging import get_logger

log = get_logger(__name__)


def producer_config(kafka_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Librdkafka settings for high-throughput, duplicate-free publishing."""
    p = kafka_cfg["producer"]
    return {
        "bootstrap.servers": kafka_cfg["bootstrap_servers"],
        "linger.ms": int(p["linger_ms"]),
        "batch.size": int(p["batch_size_bytes"]),
        "compression.type": str(p["compression"]),
        "acks": str(p["acks"]),
        "enable.idempotence": bool(p["enable_idempotence"]),
        "queue.buffering.max.messages": 1_000_000,
        "queue.buffering.max.kbytes": 512 * 1024,
        # Only failed deliveries trigger the callback: one Python call per error, not per message.
        "delivery.report.only.error": True,
    }


def make_producer(kafka_cfg: Mapping[str, Any], on_error=None) -> Producer:
    """Create a producer; ``on_error(err, msg)`` is called for every failed delivery."""
    cfg = producer_config(kafka_cfg)
    if on_error is not None:
        cfg["on_delivery"] = on_error
    return Producer(cfg)


def admin(kafka_cfg: Mapping[str, Any]) -> AdminClient:
    """Admin client for topic management."""
    return AdminClient({"bootstrap.servers": kafka_cfg["bootstrap_servers"]})


def wait_for_kafka(kafka_cfg: Mapping[str, Any], timeout_s: float = 60.0) -> None:
    """Block until the broker answers metadata requests, or raise ``TimeoutError``."""
    deadline = time.monotonic() + timeout_s
    client = admin(kafka_cfg)
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.list_topics(timeout=5)
            return
        except KafkaException as exc:
            last = exc
            time.sleep(1)
    raise TimeoutError(f"Kafka not reachable at {kafka_cfg['bootstrap_servers']}") from last


def topic_specs(kafka_cfg: Mapping[str, Any]) -> list[NewTopic]:
    """Topics declared in config (PRD §6.1)."""
    specs = []
    for spec in kafka_cfg["topics"].values():
        retention_ms = int(spec["retention_hours"]) * 3600 * 1000
        specs.append(
            NewTopic(
                spec["name"],
                num_partitions=int(spec["partitions"]),
                replication_factor=1,
                config={"retention.ms": str(retention_ms), "compression.type": "producer"},
            )
        )
    return specs


def ensure_topics(
    kafka_cfg: Mapping[str, Any],
    recreate: bool = False,
    names: list[str] | None = None,
) -> list[str]:
    """Create missing topics (all configured ones, or only ``names``). Returns created names.

    With ``recreate=True`` existing topics are deleted first, which also deletes their messages.
    """
    client = admin(kafka_cfg)
    wanted = [t for t in topic_specs(kafka_cfg) if names is None or t.topic in names]
    existing = set(client.list_topics(timeout=10).topics)
    if recreate:
        doomed = [t.topic for t in wanted if t.topic in existing]
        if doomed:
            for name, fut in client.delete_topics(doomed, operation_timeout=30).items():
                fut.result()
                log.info("deleted topic %s", name)
            _wait_until_gone(client, doomed)
            existing -= set(doomed)
    to_create = [t for t in wanted if t.topic not in existing]
    if to_create:
        for name, fut in client.create_topics(to_create, operation_timeout=30).items():
            fut.result()
            log.info("created topic %s", name)
    return [t.topic for t in to_create]


def create_topic(
    kafka_cfg: Mapping[str, Any], name: str, partitions: int, retention_hours: int = 24
) -> None:
    """Create one ad-hoc topic (used by integration tests); no-op if it exists."""
    client = admin(kafka_cfg)
    if name in client.list_topics(timeout=10).topics:
        return
    spec = NewTopic(
        name,
        num_partitions=partitions,
        replication_factor=1,
        config={"retention.ms": str(retention_hours * 3600 * 1000)},
    )
    client.create_topics([spec], operation_timeout=30)[name].result()


def delete_topic(kafka_cfg: Mapping[str, Any], name: str) -> None:
    """Delete one topic if present."""
    client = admin(kafka_cfg)
    if name in client.list_topics(timeout=10).topics:
        client.delete_topics([name], operation_timeout=30)[name].result()


def _wait_until_gone(client: AdminClient, names: list[str], timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not set(names) & set(client.list_topics(timeout=10).topics):
            return
        time.sleep(0.5)
    raise TimeoutError(f"topics still present after delete: {names}")
