"""Project-wide logging setup. Library code logs through ``get_logger``; it never prints."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    level_name = os.environ.get("LEAKWATCH_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger("leakwatch")
    root.addHandler(handler)
    root.setLevel(getattr(logging, level_name, logging.INFO))
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``leakwatch`` namespace (configured on first use)."""
    _configure()
    if not name.startswith("leakwatch"):
        name = f"leakwatch.{name}"
    return logging.getLogger(name)
