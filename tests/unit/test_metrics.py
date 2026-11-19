from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from leakwatch.common.metrics import git_commit, latency_stats, read_metrics, write_metrics
from leakwatch.common.paths import REPO_ROOT


def test_latency_stats_nearest_rank_and_empty():
    stats = latency_stats([3.0, None, 1.0, 2.0, 10.0])
    assert stats == {"n": 4, "mean": 4.0, "p50": 3.0, "p95": 10.0, "max": 10.0}
    assert latency_stats([None]) == {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}


def test_write_metrics_adds_provenance_and_cleans_values(tmp_path: Path):
    path = write_metrics(
        "demo",
        {
            "nan": float("nan"),
            "inf": math.inf,
            "np_int": np.int64(3),
            "np_float": np.float32(1.5),
            "nested": {"path": Path("/x/y"), "list": [np.float64(2.0), float("nan")]},
        },
        out_dir=tmp_path,
    )
    assert path == tmp_path / "demo.json"
    doc = json.loads(path.read_text(encoding="utf-8"))  # strict JSON: no NaN tokens
    assert doc["nan"] is None
    assert doc["inf"] is None
    assert doc["np_int"] == 3
    assert doc["np_float"] == 1.5
    assert doc["nested"] == {"path": "/x/y", "list": [2.0, None]}
    assert set(doc["meta"]) >= {"generated_at", "git_commit", "leakwatch_version", "python"}
    assert read_metrics("demo", out_dir=tmp_path) == doc


@pytest.mark.parametrize("pattern", ["reports/metrics/*.json", "reports/figures/*.png"])
def test_git_commit_ignores_report_outputs(pattern):
    """Rewriting a tracked metrics file or figure must not mark the code as dirty."""
    tracked = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    state = git_commit()
    if not tracked or state == "unknown" or state.endswith("+dirty"):
        pytest.skip("needs a clean git checkout with tracked report outputs")
    path = REPO_ROOT / tracked[0]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        assert git_commit() == state
    finally:
        path.write_bytes(original)
