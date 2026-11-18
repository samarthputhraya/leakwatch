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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``needs_data`` tests when `make data` has not been run (e.g. in CI)."""
    if _have_data():
        return
    skip = pytest.mark.skip(reason="needs `make data` (BattLeDIM files not on this machine)")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def official_config() -> Path:
    return FIXTURES / "dataset_configuration.yaml"


@pytest.fixture(scope="session")
def fixed_report() -> Path:
    return FIXTURES / "2018_Fixed_Leakages_Report.txt"
