from __future__ import annotations

from pathlib import Path

import pytest

from leakwatch.common import paths

FIXTURES = paths.TESTS_FIXTURES_DIR / "battledim"


def _have_data() -> bool:
    return (
        paths.ground_truth_path().exists()
        and all(paths.scada_parquet_path(y).exists() for y in (2018, 2019))
        and paths.network_inp_path().exists()
    )


def _have_raw_lake_2018() -> bool:
    return (paths.lake_raw_dir() / "source=battledim" / "year=2018").exists()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests whose inputs are missing on this machine (e.g. in CI)."""
    have_data, have_lake = _have_data(), _have_raw_lake_2018()
    skip_data = pytest.mark.skip(reason="needs `make data` (BattLeDIM files not on this machine)")
    skip_lake = pytest.mark.skip(reason="needs the 2018 raw lake (`make ingest-year YEAR=2018`)")
    for item in items:
        if "needs_data" in item.keywords and not have_data:
            item.add_marker(skip_data)
        if "needs_lake" in item.keywords and not have_lake:
            item.add_marker(skip_lake)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def official_config() -> Path:
    return FIXTURES / "dataset_configuration.yaml"


@pytest.fixture(scope="session")
def fixed_report() -> Path:
    return FIXTURES / "2018_Fixed_Leakages_Report.txt"


@pytest.fixture(scope="session")
def spark():
    """Small local SparkSession shared by all Spark tests (no Kafka connector)."""
    from leakwatch.common.config import load_config
    from leakwatch.stream.spark import get_spark

    cfg = load_config(environ={})["spark"]
    session = get_spark(
        {**cfg, "shuffle_partitions": 2, "ui_enabled": False, "driver_memory": "1g"},
        "leakwatch-tests",
        master="local[2]",
    )
    yield session
    session.stop()
