"""Write result files under ``reports/metrics/``: the only source for numbers in reports."""

from __future__ import annotations

import json
import math
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leakwatch import __version__
from leakwatch.common.paths import METRICS_DIR, REPO_ROOT, ensure_dir


def git_commit() -> str:
    """Short hash of HEAD (``+dirty`` if the tree has changes), or ``unknown`` outside git."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}+dirty" if dirty else head


def _clean(value: Any) -> Any:
    """Make values JSON-safe: NaN/inf -> None, numpy scalars -> Python, paths -> str."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_metadata() -> dict[str, Any]:
    """Provenance attached to every metrics file."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "leakwatch_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def write_metrics(name: str, payload: dict[str, Any], out_dir: Path | None = None) -> Path:
    """Write ``reports/metrics/<name>.json`` with a ``meta`` block and the given payload."""
    target_dir = ensure_dir(out_dir or METRICS_DIR)
    path = target_dir / f"{name}.json"
    document = {"meta": run_metadata(), **_clean(payload)}
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_metrics(name: str, out_dir: Path | None = None) -> dict[str, Any]:
    """Read a metrics file written by :func:`write_metrics`."""
    path = (out_dir or METRICS_DIR) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
