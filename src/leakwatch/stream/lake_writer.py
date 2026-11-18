"""Exactly-once micro-batch writes into a Parquet lake layer.

Spark's ``foreachBatch`` is at-least-once: after a crash the last micro-batch can run again, and a
plain ``append`` would then duplicate rows. Each batch therefore goes through a tiny commit
protocol keyed by its ``batch_id``:

1. skip the batch if its commit marker already exists (it was fully written before);
2. write it to ``_staging/<layer>/batch=<id>`` (overwrite, so a retry starts clean);
3. delete any files a previous, interrupted attempt of this batch left in the lake
   (all lake files carry the prefix ``b<id>-``), then move the staged files into place
   (same filesystem, so each move is an atomic rename);
4. write the commit marker last.

Readers (DuckDB, Spark batch jobs) only ever see whole batches plus, at worst, the files of one
batch that is being moved; they never see a batch twice.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leakwatch.common.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LakeLayout:
    """Where one lake layer lives."""

    layer_dir: Path
    staging_dir: Path
    commits_dir: Path

    def staging_for(self, batch_id: int) -> Path:
        """Scratch directory for one batch."""
        return self.staging_dir / f"batch={batch_id}"

    def marker_for(self, batch_id: int) -> Path:
        """Commit marker of one batch."""
        return self.commits_dir / f"{batch_id:010d}.json"

    def ensure(self) -> None:
        """Create the layer, staging and commit directories."""
        for d in (self.layer_dir, self.staging_dir, self.commits_dir):
            d.mkdir(parents=True, exist_ok=True)


def file_prefix(batch_id: int) -> str:
    """Prefix of every lake file written by ``batch_id``."""
    return f"b{batch_id:010d}-"


def is_committed(layout: LakeLayout, batch_id: int) -> bool:
    """True if the batch was fully written before."""
    return layout.marker_for(batch_id).exists()


def remove_partial(layout: LakeLayout, batch_id: int) -> int:
    """Delete lake files left by an interrupted attempt of ``batch_id``. Returns files removed."""
    prefix = file_prefix(batch_id)
    removed = 0
    for f in layout.layer_dir.rglob(f"{prefix}*"):
        f.unlink()
        removed += 1
    return removed


def publish_staged(layout: LakeLayout, batch_id: int) -> int:
    """Move staged Parquet files into the layer, keeping partition dirs. Returns files moved."""
    stage = layout.staging_for(batch_id)
    moved = 0
    if not stage.exists():
        return 0
    for f in sorted(stage.rglob("*.parquet")):
        rel_dir = f.parent.relative_to(stage)
        target_dir = layout.layer_dir / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        os.replace(f, target_dir / f"{file_prefix(batch_id)}{f.name}")
        moved += 1
    shutil.rmtree(stage, ignore_errors=True)
    return moved


def write_marker(layout: LakeLayout, batch_id: int, info: dict[str, Any]) -> Path:
    """Write the commit marker atomically (temp file + rename)."""
    marker = layout.marker_for(batch_id)
    tmp = marker.with_suffix(".tmp")
    payload = {"batch_id": batch_id, "committed_at": datetime.now(UTC).isoformat(), **info}
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(tmp, marker)
    return marker


def read_markers(layout: LakeLayout) -> list[dict[str, Any]]:
    """All commit markers of the layer, in batch order."""
    if not layout.commits_dir.exists():
        return []
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(layout.commits_dir.glob("*.json"))
    ]


def commit_batch(layout: LakeLayout, batch_id: int, info: dict[str, Any]) -> int:
    """Steps 3–4 of the protocol once the batch is staged. Returns files published."""
    removed = remove_partial(layout, batch_id)
    if removed:
        log.warning("batch %d: removed %d files from an interrupted attempt", batch_id, removed)
    moved = publish_staged(layout, batch_id)
    write_marker(layout, batch_id, {**info, "files": moved})
    return moved
