"""Checks before the live demo starts: everything it reads must exist (``make demo``).

Prints one line per item and, for anything missing, the make target that builds it. Exit code 0
when the demo can run, 1 otherwise.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger

log = get_logger("leakwatch.demo.preflight")


def _kafka_up() -> bool:
    from leakwatch.ingest import kafka_io

    try:
        kafka_io.admin(load_config()["kafka"]).list_topics(timeout=5)
        return True
    except Exception:
        return False


def _has_files(folder: Path, pattern: str = "*.parquet") -> bool:
    return folder.exists() and any(folder.rglob(pattern))


def checks() -> list[tuple[str, Callable[[], bool], str, bool]]:
    """(item, check, how to fix, required) for every demo prerequisite."""
    cfg = load_config()
    year = int(str(cfg["demo"]["background_start"])[:4])
    model = paths.models_dir() / str(cfg["bdt"]["live_model"])
    return [
        ("Kafka broker", _kafka_up, "make up", True),
        (
            f"BattLeDIM {year} SCADA (replayed by the demo)",
            lambda: paths.scada_parquet_path(year).exists(),
            "make data",
            True,
        ),
        (
            "hour-of-week baselines",
            lambda: _has_files(paths.lake_baselines_dir()),
            "make prepare (or make features)",
            True,
        ),
        (
            f"live K-Means model ({model.name})",
            lambda: (model / "params.json").exists(),
            "make prepare (or make bdt-train)",
            True,
        ),
        (
            "leak localizer",
            lambda: (
                (paths.models_dir() / "gnn" / "meta.json").exists()
                and (paths.sim_dir() / "gnn" / "graph.npz").exists()
            ),
            "make gnn-data gnn (without it alerts have no top-5 candidates)",
            False,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the checks; exit 1 if a required item is missing."""
    ok = True
    for item, check, fix, required in checks():
        good = bool(check())
        if good:
            log.info("ok       %s", item)
        elif required:
            ok = False
            log.error("MISSING  %s -> run `%s`", item, fix)
        else:
            log.warning("optional %s missing -> `%s`", item, fix)
    if not ok:
        log.error("the demo cannot start yet; see the commands above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
