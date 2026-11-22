"""Write result files under ``reports/metrics/``: the only source for numbers in reports."""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leakwatch import __version__
from leakwatch.common.paths import METRICS_DIR, REPO_ROOT, ensure_dir


def git_commit() -> str:
    """Short hash of HEAD (``+dirty`` if code or docs changed), or ``unknown`` outside git.

    Changes under ``reports/`` (metrics, figures, generated report drafts) do not count: they are
    outputs, and a job that writes a figure and then its metrics file, or several metrics files,
    must not stamp its own results dirty.
    """
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
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                f":(exclude){METRICS_DIR.parent.relative_to(REPO_ROOT).as_posix()}",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            # From WSL on the Windows disk, git re-reads every file after Windows git touched the
            # index or the laptop woke up: 29 s was measured once, so 30 s stamped "unknown".
            timeout=180,
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


def latency_stats(values: Iterable[float | None]) -> dict[str, Any]:
    """Count, mean, p50, p95 and max of latencies in seconds (nearest rank; ``None`` dropped)."""
    lat = sorted(float(v) for v in values if v is not None)
    if not lat:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def pct(q: float) -> float:
        return round(lat[min(len(lat) - 1, round(q * (len(lat) - 1)))], 3)

    return {
        "n": len(lat),
        "mean": round(statistics.fmean(lat), 3),
        "p50": pct(0.5),
        "p95": pct(0.95),
        "max": lat[-1],
    }


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
