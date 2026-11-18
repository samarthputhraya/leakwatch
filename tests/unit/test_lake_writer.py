"""Exactly-once commit protocol for lake micro-batches."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from leakwatch.stream import lake_writer as lw


@pytest.fixture
def layout(tmp_path: Path) -> lw.LakeLayout:
    lay = lw.LakeLayout(tmp_path / "raw", tmp_path / "_staging/raw", tmp_path / "_commits/raw")
    lay.ensure()
    return lay


def stage(layout: lw.LakeLayout, batch_id: int, parts: dict[str, list[str]]) -> None:
    for rel, names in parts.items():
        d = layout.staging_for(batch_id) / rel
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"PAR1")
    (layout.staging_for(batch_id) / "_SUCCESS").write_text("")


def lake_files(layout: lw.LakeLayout) -> list[str]:
    return sorted(str(p.relative_to(layout.layer_dir)) for p in layout.layer_dir.rglob("*.parquet"))


def test_commit_moves_files_into_partitions_and_writes_marker(layout):
    stage(layout, 3, {"source=x/date=2019-01-01": ["part-0.parquet", "part-1.parquet"]})
    moved = lw.commit_batch(layout, 3, {"rows": 10})
    assert moved == 2
    assert lake_files(layout) == [
        "source=x/date=2019-01-01/b0000000003-part-0.parquet",
        "source=x/date=2019-01-01/b0000000003-part-1.parquet",
    ]
    assert lw.is_committed(layout, 3)
    assert not layout.staging_for(3).exists()
    (marker,) = lw.read_markers(layout)
    assert marker["rows"] == 10 and marker["files"] == 2 and marker["batch_id"] == 3


def test_retry_after_partial_publish_does_not_duplicate(layout):
    # Attempt 1 published one file, then crashed before the marker.
    stage(layout, 7, {"d=1": ["part-a.parquet", "part-b.parquet"]})
    stage_dir = layout.staging_for(7)
    first = next(stage_dir.rglob("part-a.parquet"))
    target = layout.layer_dir / "d=1"
    target.mkdir(parents=True)
    first.rename(target / f"{lw.file_prefix(7)}part-a.parquet")
    assert not lw.is_committed(layout, 7)
    # Attempt 2 restages with different file names (Spark uses new UUIDs) and commits.
    shutil.rmtree(stage_dir)
    stage(layout, 7, {"d=1": ["part-c.parquet", "part-d.parquet"]})
    lw.commit_batch(layout, 7, {"rows": 4})
    assert lake_files(layout) == [
        "d=1/b0000000007-part-c.parquet",
        "d=1/b0000000007-part-d.parquet",
    ]


def test_other_batches_are_untouched(layout):
    stage(layout, 1, {"d=1": ["p.parquet"]})
    lw.commit_batch(layout, 1, {})
    stage(layout, 2, {"d=1": ["p.parquet"]})
    lw.commit_batch(layout, 2, {})
    assert lake_files(layout) == ["d=1/b0000000001-p.parquet", "d=1/b0000000002-p.parquet"]
    assert [m["batch_id"] for m in lw.read_markers(layout)] == [1, 2]


def test_empty_batch_commits_marker_only(layout):
    assert lw.commit_batch(layout, 9, {"rows": 0}) == 0
    assert lw.is_committed(layout, 9)
    assert lake_files(layout) == []
