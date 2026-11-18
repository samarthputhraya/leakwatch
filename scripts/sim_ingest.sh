#!/usr/bin/env bash
# Replay all generated scenarios through Kafka into the raw lake, with the Spark raw ingest running
# alongside (same pattern as ingest_year.sh). The Spark job stops 120 s after the last data.
#
# Writes reports/metrics/replay_sim_scale.json and stream_raw_sim_scale.json.
set -euo pipefail
RUN="uv run --locked --all-extras"

$RUN python -m leakwatch.stream.raw_ingest --stop-when-idle 120 \
  --metrics-name "stream_raw_sim_scale" &
STREAM_PID=$!
trap 'kill "$STREAM_PID" 2>/dev/null || true' INT TERM

$RUN python -m leakwatch.sim.replay
wait "$STREAM_PID"
