"""``make live-replay``: replay part of 2019 through Kafka at demo speed with live K-Means alerts.

Phase 3 Definition of Done: alerts appear live when replaying 2019. This job replays a slice of
BattLeDIM 2019 (default: 17 h around the abrupt leak on p523, 2019-01-15 23:00) at a paced speed
(default 360×) into a throwaway topic, runs the feature stream with live scoring
(``stream.features_stream`` + ``bdt.live``) into a scratch lake under ``data/bench/live``, and
records when each alert is committed.

Latencies (wall clock):

- *detection latency*: from publishing the reading that lets the m-th window above τ be emitted
  (the window's last step + lateness) to the alert being committed to the lake;
- *leak-to-alert*: from publishing the first reading after the leak starts to the alert being
  committed (the number that matters for the 90-second demo, PRD §16).

Publish times come from the replayer's pacing: a reading with event time ``t`` is published at
``wall0 + (t - t0) / speed``. The live alerts are also compared with the batch evaluation's alerts
(``origin=batch``) that start inside the slice.

Writes ``reports/metrics/bdt_live_replay_2019.json``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import uuid
from typing import Any

import pandas as pd

from leakwatch.common import lake, paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.data import battledim
from leakwatch.data.ground_truth import load_ground_truth
from leakwatch.features.baselines import read_baselines
from leakwatch.features.spec import load_feature_spec
from leakwatch.ingest import kafka_io
from leakwatch.ingest.replayer import Pacer, Replayer, ReplayStats, delivery_error_counter
from leakwatch.stream import lake_writer
from leakwatch.stream.features_stream import (
    FeatureBatchWriter,
    FeatureStreamPaths,
    make_live_scoring,
    start_query,
    summarise,
)
from leakwatch.stream.raw_ingest import await_query, kafka_readings
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.stream.live_replay")


def main(argv: list[str] | None = None) -> int:
    """Run the paced live replay and write its metrics."""
    parser = argparse.ArgumentParser(description="Live 2019 replay with K-Means alerts.")
    parser.add_argument("--start", default="2019-01-15 17:00", help="first simulated timestamp")
    # 17 h: 6 h before the leak, then room for the live model's 8 h persistence (m = 32, D-038)
    parser.add_argument("--hours", type=float, default=17.0, help="simulated hours to replay")
    parser.add_argument("--speed", type=float, default=360.0, help="simulated s per wall s")
    parser.add_argument("--leak", default="L2019_p523", help="leak whose start is in the slice")
    parser.add_argument("--keep", action="store_true", help="keep the scratch lake afterwards")
    args = parser.parse_args(argv)

    cfg = load_config()
    spec = load_feature_spec(cfg)
    kafka_cfg = cfg["kafka"]
    kafka_io.wait_for_kafka(kafka_cfg)
    start = pd.Timestamp(args.start, tz="UTC")
    end = start + battledim.TIMESTEP * (round(args.hours * 12) - 1)
    wide = battledim.load_scada(start.year).loc[start:end]
    gt = load_ground_truth().set_index("leak_id")
    leak_start = gt.loc[args.leak, "start_ts"] if args.leak in gt.index else None

    root = paths.data_dir() / "bench" / "live"
    shutil.rmtree(root, ignore_errors=True)
    stream_paths = FeatureStreamPaths.under(root / "lake")
    topic = f"bench.live.{uuid.uuid4().hex[:8]}"
    alerts_topic = f"bench.alerts.{uuid.uuid4().hex[:8]}"
    partitions = int(kafka_cfg["topics"]["readings"]["partitions"])
    kafka_io.create_topic(kafka_cfg, topic, partitions=partitions)
    kafka_io.create_topic(kafka_cfg, alerts_topic, partitions=1)
    published: list[dict[str, Any]] = []
    try:
        spark = get_spark(cfg["spark"], "leakwatch-live-replay", with_kafka=True)
        baselines = read_baselines(spark, paths.lake_baselines_dir()).cache()
        live = make_live_scoring(cfg, spec, alerts_topic=alerts_topic)
        inner = live.publish

        def publish(rows: list[dict[str, Any]]) -> None:
            now = time.time()
            published.extend({**r, "_published_epoch_s": now} for r in rows)
            if inner is not None:
                inner(rows)

        live.publish = publish
        writer = FeatureBatchWriter(stream_paths, spec, baselines, live)
        readings = kafka_readings(spark, cfg, topic, cfg["features"]["stream"])
        query = start_query(readings, writer, cfg["features"]["stream"]["trigger_seconds"])
        stats = ReplayStats()
        producer = kafka_io.make_producer(kafka_cfg, on_error=delivery_error_counter(stats))
        log.info("replaying %s .. %s at %gx", start, end, args.speed)
        wall0 = time.time()
        Replayer(producer, topic, Pacer(args.speed), progress_every_steps=0, stats=stats).run(wide)
        await_query(query, stop_when_idle_s=30)
        markers = lake_writer.read_markers(stream_paths.layout)
        summary = summarise([dict(p) for p in query.recentProgress], markers)
        con = lake.connect(lake_root=root / "lake")
        live_alerts = (
            lake.query_df(con, "SELECT * FROM alerts ORDER BY start_ts")
            if "alerts"
            in {r[0] for r in con.execute("SELECT view_name FROM duckdb_views()").fetchall()}
            else pd.DataFrame()
        )
    finally:
        kafka_io.delete_topic(kafka_cfg, topic)
        kafka_io.delete_topic(kafka_cfg, alerts_topic)

    def publish_wall(ts: pd.Timestamp) -> float:
        return wall0 + (ts - start).total_seconds() / args.speed

    lateness = pd.Timedelta(seconds=spec.lateness_s)
    step = pd.Timedelta(seconds=spec.step_s)
    opened = {}
    for row in published:
        if row["version"] == 1 and row["alert_id"] not in opened:
            opened[row["alert_id"]] = row
    alerts_out = []
    for alert_id, row in opened.items():
        fired = pd.Timestamp(row["start_ts"])
        trigger_reading = fired - step + lateness
        entry = {
            "alert_id": alert_id,
            "start_ts": fired.isoformat(),
            "area": row["area"],
            "top_sensors": row["top_sensors"],
            "kmeans_score_max_at_open": row["kmeans_score_max"],
            "detection_latency_s": round(
                row["_published_epoch_s"] - publish_wall(trigger_reading), 2
            ),
        }
        if leak_start is not None and fired >= leak_start:
            entry["event_time_after_leak_start_h"] = round(
                (fired - leak_start).total_seconds() / 3600, 2
            )
            entry["leak_to_alert_wall_s"] = round(
                row["_published_epoch_s"] - publish_wall(leak_start), 2
            )
        alerts_out.append(entry)

    batch = lake.query_df(
        lake.connect(),
        f"""SELECT alert_id, start_ts FROM alerts WHERE origin = 'batch'
            AND start_ts BETWEEN '{start}' AND '{end}' ORDER BY start_ts""",
    )
    live_starts = sorted(pd.Timestamp(a["start_ts"]) for a in alerts_out)
    batch_starts = sorted(batch["start_ts"])
    payload = {
        "slice": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "simulated_hours": args.hours,
            "speed": args.speed,
            "readings": stats.readings,
            "replay_wall_s": round(stats.wall_s, 1),
        },
        "leak": {
            "leak_id": args.leak,
            "start": None if leak_start is None else leak_start.isoformat(),
        },
        "stream": {
            "batches_with_data": summary["batches_with_data"],
            "windows_emitted": summary["windows_emitted"],
            "late_readings_ignored": summary["late_readings_ignored"],
            "batch_duration_s": summary["batch_duration_s"],
            "steady_state_batch_duration_s": summary["steady_state_excluding_first_batch"][
                "batch_duration_s"
            ],
            "first_batch_duration_s": summary["first_batch_duration_s"],
        },
        "alerts_opened_live": alerts_out,
        "alert_versions_published": len(published),
        "alert_rows_in_scratch_lake": len(live_alerts),
        "batch_alerts_in_slice": [t.isoformat() for t in batch_starts],
        "live_alert_starts_match_batch": live_starts == list(batch_starts),
        "note": "latencies are wall-clock; publish times derived from the replayer's pacing",
    }
    path = write_metrics("bdt_live_replay_2019", payload)
    log.info("%d live alerts (batch: %d) -> %s", len(alerts_out), len(batch_starts), path)
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if alerts_out else 1


if __name__ == "__main__":
    sys.exit(main())
