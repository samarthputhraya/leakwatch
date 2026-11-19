#!/usr/bin/env bash
# `make demo`: the live demo — demo runner (Kafka replay + Spark streams + leak injection) in the
# background and the Streamlit dashboard in the foreground (http://localhost:8501).
# Needs Kafka (`make up`) and the trained models (`make bdt-train gnn`).
set -euo pipefail
RUN="uv run --locked --all-extras"
export PYTHONFAULTHANDLER=1  # a native crash prints the Python stack instead of a bare exit code

$RUN python -m leakwatch.demo.preflight
$RUN python -m leakwatch.demo.runner &
RUNNER_PID=$!
trap 'kill "$RUNNER_PID" 2>/dev/null || true' EXIT INT TERM

$RUN streamlit run src/leakwatch/dashboard/app.py \
  --server.headless true --server.port "${DASHBOARD_PORT:-8501}" --browser.gatherUsageStats false
