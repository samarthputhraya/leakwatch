# LeakWatch — run inside WSL. From PowerShell in the repo folder: `wsl make <target>`.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

export PATH := $(HOME)/.local/bin:$(PATH)
# Keep the venv on the WSL disk, not in OneDrive (docs/DECISIONS.md D-002).
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/leakwatch
RUN := uv run --locked --all-extras
PY_DIRS := src tests scripts notebooks

.PHONY: help setup data test test-all cov lint fmt eda sim-bench check up down demo kafka-reset \
	lake-reset replay stream ingest-year stream-latency lake-stats test-integration features \
	stream-features features-parity features-stream-reset bdt-train bdt-eval live-replay \
	sim-scale sim-ingest lstm-ae gnn-data gnn index dispatch pe-eval demo-runner dashboard \
	demo-check

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup: ## install pinned Python deps (uv.lock) into the WSL venv; create .env if missing
	@command -v uv >/dev/null || { echo "uv not found: see README 'Quick start'"; exit 1; }
	uv sync --locked --all-extras
	@[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }

data: ## download BattLeDIM (MD5-checked), convert to Parquet, build + cross-check ground truth
	$(RUN) python -m leakwatch.data.build

test: ## unit + contract tests (data tests skip if `make data` has not run)
	$(RUN) pytest -m "not slow and not integration"

test-all: ## every test except integration (those need `make up`)
	$(RUN) pytest -m "not integration"

cov: ## tests with coverage report
	$(RUN) pytest -m "not slow and not integration" --cov --cov-report=term-missing

lint: ## ruff lint + format check
	$(RUN) ruff check $(PY_DIRS)
	$(RUN) ruff format --check $(PY_DIRS)

fmt: ## auto-format and fix lint
	$(RUN) ruff format $(PY_DIRS)
	$(RUN) ruff check --fix $(PY_DIRS)

eda: ## execute the EDA notebook headless (figures -> reports/figures/)
	$(RUN) python scripts/run_notebook.py notebooks/01_eda_battledim.py

sim-bench: ## time 48 h + 7-day leak simulations (WNTR vs epyt-flow) -> reports/metrics/sim_benchmark.json
	$(RUN) python -m leakwatch.sim.bench

check: lint test ## what CI runs

up: ## start Kafka (KRaft, Docker) and create the topics
	docker compose up -d --wait
	$(RUN) python -m leakwatch.ingest.topics

down: ## stop Kafka (its messages are dropped; the lake is kept)
	docker compose down

kafka-reset: ## delete and recreate the Kafka topics (drops queued messages)
	$(RUN) python -m leakwatch.ingest.topics --recreate

lake-reset: ## delete the raw lake + live features (and their checkpoints), and empty the topics
	$(RUN) python -m leakwatch.stream.raw_ingest --reset
	$(RUN) python -m leakwatch.stream.features_stream --reset
	$(RUN) python -m leakwatch.ingest.topics --recreate

YEAR ?= 2018
SPEED ?= max
replay: ## publish a BattLeDIM year to Kafka: make replay YEAR=2019 SPEED=360
	$(RUN) python -m leakwatch.ingest.replayer --year $(YEAR) --speed $(SPEED)

stream: ## run the Spark raw-ingest stream until Ctrl+C (Spark UI: http://localhost:4040)
	$(RUN) python -m leakwatch.stream.raw_ingest

ingest-year: ## replay YEAR through Kafka into the raw lake with Spark running alongside
	bash scripts/ingest_year.sh $(YEAR) $(SPEED)

lake-stats: ## rows, files and bytes per lake layer -> reports/metrics/lake_*_summary.json
	$(RUN) python -m leakwatch.common.lake

stream-latency: ## paced replay (360x) -> Spark: steady-state batch time and publish->lake latency
	$(RUN) python -m leakwatch.stream.latency_bench

features: ## Spark batch: cleaned layer, hour-of-week baselines, feature tables for 2018 + 2019
	$(RUN) python -m leakwatch.features.build

stream-features: ## Kafka -> live feature rows (same function as the batch job) until idle
	$(RUN) python -m leakwatch.stream.features_stream --stop-when-idle 60 \
		--metrics-name features_stream_battledim

features-parity: ## compare live (stream) and batch feature tables window by window
	$(RUN) python -m leakwatch.features.parity

bdt-train: ## Spark MLlib K-Means on normal 2018 windows (+ strict ablation), tau and m from 2018
	$(RUN) python -m leakwatch.bdt.train

bdt-eval: ## score 2018-2019, write scores + alerts, evaluate on 2019 -> bdt_kmeans_2019.json
	$(RUN) python -m leakwatch.bdt.evaluate

sim-scale: ## Spark-parallel scenario simulation (epyt-flow) -> lake/sim/scenarios (full: configs/full.yaml)
	$(RUN) python -m leakwatch.sim.generate

sim-ingest: ## replay generated scenarios through Kafka into lake/raw (source=sim), Spark alongside
	bash scripts/sim_ingest.sh

lstm-ae: ## LSTM autoencoder detector + K-Means comparison at matched false alarms -> dl_lstm_ae.json
	$(RUN) python -m leakwatch.dl.lstm_ae

gnn-data: ## single-leak scenarios for the localizer (EPANET emitters) -> data/sim/gnn
	$(RUN) python -m leakwatch.dl.gnn_data

gnn: ## nearest-sensor / MLP / GCN / GAT localizers, 2019 test, gnn_top5 into alerts -> dl_gnn.json
	$(RUN) python -m leakwatch.dl.gnn

live-replay: ## replay part of 2019 at 360x with live K-Means alerts (needs `make up`)
	$(RUN) python -m leakwatch.stream.live_replay

index: ## copilot RAG index: docs/kb + CPHEEO PDFs in data/docs (downloaded if missing) -> data/index/
	$(RUN) python -m leakwatch.copilot.rag.build

dispatch: ## work orders for alerts without one -> lake/work_orders (template when no LLM is set)
	$(RUN) python -m leakwatch.copilot.dispatch

PE_VERSION ?= all
pe-eval: ## copilot evaluation, needs LLM_* in .env: make pe-eval PE_VERSION=v1|v2|v3|all
	$(RUN) python -m leakwatch.copilot.evaluate --version $(PE_VERSION)

features-stream-reset: ## delete the live feature table, its carried state and checkpoint
	$(RUN) python -m leakwatch.stream.features_stream --reset

test-integration: ## end-to-end tests against the running Kafka (`make up` first)
	$(RUN) pytest -m integration

demo: ## live demo: runner (replay + Spark + leak injection) + dashboard on :8501 (needs `make up`)
	bash scripts/demo.sh

demo-runner: ## only the demo backend (replay + streams + leak injection + work orders)
	$(RUN) python -m leakwatch.demo.runner

dashboard: ## only the Streamlit dashboard (http://localhost:8501)
	$(RUN) streamlit run src/leakwatch/dashboard/app.py --server.headless true

demo-check: ## headless demo rehearsal: inject, alert, top 5, work order x3 -> demo_runs.json
	$(RUN) python -m leakwatch.demo.check
