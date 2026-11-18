from __future__ import annotations

from pathlib import Path

import pytest

from leakwatch.common import paths


def test_data_dir_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LEAKWATCH_DATA_DIR", str(tmp_path))
    root = tmp_path.resolve()
    assert paths.data_dir() == root
    assert paths.scada_parquet_path(2019) == root / "raw/battledim/parquet/scada_2019.parquet"
    assert paths.lake_dir() == root / "lake"


def test_data_dir_falls_back_to_repo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LEAKWATCH_DATA_DIR", "")
    assert paths.data_dir() == paths.REPO_ROOT / "data"


def test_repo_root_contains_project_files():
    assert (paths.REPO_ROOT / "pyproject.toml").exists()
    assert (paths.CONFIGS_DIR / "default.yaml").exists()
