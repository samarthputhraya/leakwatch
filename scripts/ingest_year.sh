#!/usr/bin/env bash
# Replay one BattLeDIM year through Kafka into the raw lake, with the Spark streaming job running
# alongside (as it would in production). The Spark job stops 60 s after the last data arrives.
#
# Usage: scripts/ingest_year.sh <year> [speed]     e.g. scripts/ingest_year.sh 2018 max
# Writes reports/metrics/replay_battledim_<year>.json and stream_raw_battledim_<year>.json.
set -euo pipefail
YEAR="${1:?year required}"
SPEED="${2:-max}"
RUN="uv run --locked"

$RUN python -m leakwatch.stream.raw_ingest --stop-when-idle 60 \
  --metrics-name "stream_raw_battledim_${YEAR}" &
STREAM_PID=$!
trap 'kill "$STREAM_PID" 2>/dev/null || true' INT TERM

$RUN python -m leakwatch.ingest.replayer --year "$YEAR" --speed "$SPEED"
wait "$STREAM_PID"
