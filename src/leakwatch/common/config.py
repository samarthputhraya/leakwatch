"""Configuration: ``configs/default.yaml`` + optional overlay file + environment overrides.

Precedence (later wins):
1. ``configs/default.yaml`` (laptop-safe defaults)
2. the file named by ``LEAKWATCH_CONFIG`` or passed as ``overlay`` (e.g. ``configs/full.yaml``)
3. environment variables ``LEAKWATCH__SECTION__KEY=value`` (value parsed as YAML,
   so ``LEAKWATCH__SIM__BENCH__DURATIONS_HOURS="[24]"`` gives a list)
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from leakwatch.common.paths import CONFIGS_DIR, REPO_ROOT

DEFAULT_CONFIG_PATH: Path = CONFIGS_DIR / "default.yaml"
ENV_OVERRIDE_PREFIX = "LEAKWATCH__"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged recursively into ``base``."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Turn ``LEAKWATCH__A__B=value`` variables into a nested ``{"a": {"b": value}}`` dict."""
    result: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_OVERRIDE_PREFIX):
            continue
        keys = [k.lower() for k in name[len(ENV_OVERRIDE_PREFIX) :].split("__") if k]
        if not keys:
            continue
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return content


def load_config(
    overlay: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load the merged configuration.

    Args:
        overlay: optional YAML file merged over the defaults. Falls back to ``LEAKWATCH_CONFIG``.
        environ: environment mapping to read overrides from (defaults to ``os.environ``).

    Returns:
        The merged configuration as a nested dict.

    """
    env = os.environ if environ is None else environ
    cfg = _read_yaml(DEFAULT_CONFIG_PATH)
    overlay = overlay or env.get("LEAKWATCH_CONFIG") or None
    if overlay:
        overlay_path = Path(overlay)
        if not overlay_path.is_absolute():
            overlay_path = REPO_ROOT / overlay_path
        cfg = deep_merge(cfg, _read_yaml(overlay_path))
    return deep_merge(cfg, env_overrides(env))
