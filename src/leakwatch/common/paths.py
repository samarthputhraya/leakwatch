"""Single source of truth for every filesystem path used by LeakWatch.

Code never builds data paths by hand: it calls the functions below. Data lives under
``LEAKWATCH_DATA_DIR`` when that variable is set (in ``.env`` or the environment), otherwise under
``<repo>/data``. See docs/DECISIONS.md D-002 for why data sits on the WSL disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIGS_DIR: Path = REPO_ROOT / "configs"
REPORTS_DIR: Path = REPO_ROOT / "reports"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"
TESTS_FIXTURES_DIR: Path = REPO_ROOT / "tests" / "fixtures"

# Load .env once so LEAKWATCH_DATA_DIR / LEAKWATCH_CONFIG are visible to every module.
# Values already set in the real environment win (override=False).
load_dotenv(REPO_ROOT / ".env", override=False)


def data_dir() -> Path:
    """Return the root data directory (``LEAKWATCH_DATA_DIR`` or ``<repo>/data``)."""
    env = os.environ.get("LEAKWATCH_DATA_DIR", "").strip()
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


def raw_dir() -> Path:
    """Downloaded source files, exactly as published."""
    return data_dir() / "raw"


def battledim_raw_dir() -> Path:
    """Original BattLeDIM files from Zenodo record 4017659."""
    return raw_dir() / "battledim"


def battledim_parquet_dir() -> Path:
    """Parquet conversions of the BattLeDIM CSVs (inputs to the replayer, not the lake)."""
    return battledim_raw_dir() / "parquet"


def scada_parquet_path(year: int) -> Path:
    """Wide SCADA table for one year: UTC timestamp index, one column per sensor."""
    return battledim_parquet_dir() / f"scada_{year}.parquet"


def leak_flows_parquet_path(year: int) -> Path:
    """Per-leak outflow time series (m3/h) published with the benchmark for one year."""
    return battledim_parquet_dir() / f"leak_flows_{year}.parquet"


def ground_truth_path() -> Path:
    """Tidy table of every BattLeDIM leak (built by ``make data``)."""
    return battledim_parquet_dir() / "ground_truth.parquet"


def network_inp_path() -> Path:
    """EPANET model of L-Town with nominal parameters."""
    return battledim_raw_dir() / "L-TOWN.inp"


def lake_dir() -> Path:
    """Parquet data lake (PRD §6.2). Only Spark writes here."""
    return data_dir() / "lake"


def lake_raw_dir() -> Path:
    """Raw readings layer: ``raw/source=…/year=…/date=…/sensor_type=…/`` (PRD §6.2)."""
    return lake_dir() / "raw"


def lake_clean_dir() -> Path:
    """Cleaned readings: ``clean/source=…/year=…/date=…/sensor_type=…/`` (built in batch, D-017)."""
    return lake_dir() / "clean"


def lake_baselines_dir() -> Path:
    """Hour-of-week medians per feature series from normal 2018 data (PRD §6.2, D-015)."""
    return lake_dir() / "baselines" / "hour_of_week"


def lake_features_dir() -> Path:
    """Batch feature tables: ``features/source=…/window=1h/date=…/`` (PRD §6.2)."""
    return lake_dir() / "features"


def lake_features_stream_dir() -> Path:
    """Features computed live by the streaming job (same schema as the batch tables, D-018)."""
    return lake_dir() / "features_stream"


def lake_scores_dir() -> Path:
    """Anomaly scores per window: ``scores/model=…/origin=…/date=…/`` (PRD §6.2)."""
    return lake_dir() / "scores"


def lake_alerts_dir() -> Path:
    """Alert versions (append-only event log; latest version per alert is current, PRD §6.3)."""
    return lake_dir() / "alerts"


def lake_work_orders_dir() -> Path:
    """Work orders written by the copilot, one file per order (PRD §9.3, D-021)."""
    return lake_dir() / "work_orders"


def lake_state_dir(job: str) -> Path:
    """State a streaming job carries between micro-batches (committed with each batch)."""
    return lake_dir() / "_state" / job


def lake_commits_dir(layer: str) -> Path:
    """One marker file per committed micro-batch of a lake layer (exactly-once bookkeeping)."""
    return lake_dir() / "_commits" / layer


def lake_staging_dir(layer: str) -> Path:
    """Scratch area where a micro-batch is written before its files are moved into the lake."""
    return lake_dir() / "_staging" / layer


def lake_rejects_dir() -> Path:
    """Messages that could not be parsed into readings (kept for inspection)."""
    return lake_dir() / "_rejects"


def checkpoints_dir() -> Path:
    """Spark Structured Streaming checkpoints (Kafka offsets, state store)."""
    return data_dir() / "checkpoints"


def sim_dir() -> Path:
    """Simulator outputs and scenario configs."""
    return data_dir() / "sim"


def docs_data_dir() -> Path:
    """Manuals for the copilot's RAG index (PRD §4.3)."""
    return data_dir() / "docs"


def index_dir() -> Path:
    """Vector index for RAG."""
    return data_dir() / "index"


def models_dir() -> Path:
    """Trained model artifacts."""
    return data_dir() / "models"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
