# LeakWatch — run inside WSL. From PowerShell in the repo folder: `wsl make <target>`.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

export PATH := $(HOME)/.local/bin:$(PATH)
# Keep the venv on the WSL disk, not in OneDrive (docs/DECISIONS.md D-002).
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/leakwatch
RUN := uv run --locked
PY_DIRS := src tests scripts notebooks

.PHONY: help setup data test test-all cov lint fmt eda sim-bench check up down demo

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup: ## install pinned Python deps (uv.lock) into the WSL venv; create .env if missing
	@command -v uv >/dev/null || { echo "uv not found: see README 'Quick start'"; exit 1; }
	uv sync --locked
	@[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }

data: ## download BattLeDIM (MD5-checked), convert to Parquet, build + cross-check ground truth
	$(RUN) python -m leakwatch.data.build

test: ## unit + contract tests (data tests skip if `make data` has not run)
	$(RUN) pytest -m "not slow and not integration"

test-all: ## every test, including slow ones
	$(RUN) pytest

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

up: ## start Docker services (Kafka arrives in Phase 1)
	@echo "make up: no Docker services yet. Kafka (KRaft) is added in Phase 1; see PROGRESS.md."; exit 1

down: ## stop Docker services
	@echo "make down: no Docker services yet (Phase 1)."; exit 1

demo: ## run the live demo (Phase 7)
	@echo "make demo: the dashboard demo is built in Phase 7; see PROGRESS.md."; exit 1
