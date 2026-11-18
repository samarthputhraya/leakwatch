from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from leakwatch.common.metrics import read_metrics, write_metrics


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
