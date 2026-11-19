"""Copilot settings: the ``copilot`` section of the config plus the LLM endpoint from ``.env``.

The endpoint is any OpenAI-compatible server (PRD §9.1): ``LLM_BASE_URL``, ``LLM_API_KEY`` and
``LLM_MODEL``. A local server (Ollama) needs no key; a hosted one needs a key and may leave the base
URL empty (the client's default). Keys are never logged or written to result files. ``LLM_RPM``
(optional) caps requests per minute for a rate-limited free endpoint.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from leakwatch.common import paths  # noqa: F401  (loads .env so LLM_* are visible)
from leakwatch.common.config import load_config

#: Shown wherever an answer needs a model and none is configured. Never a fake answer.
NOT_CONFIGURED_MESSAGE = (
    "The copilot needs a language model, and none is configured. Set LLM_BASE_URL, LLM_API_KEY "
    "and LLM_MODEL in .env (any OpenAI-compatible endpoint). Without a model, LeakWatch still "
    "writes template work orders (`make dispatch`) and matches resident complaints with rules."
)


@dataclass(frozen=True)
class LLMSettings:
    """Where the OpenAI-compatible endpoint is and which model to call."""

    base_url: str | None
    api_key: str | None
    model: str | None
    rpm: float | None = None  # requests per minute to stay under (free tiers)

    @property
    def configured(self) -> bool:
        """True when a model is named and either a server URL or a key is given."""
        return bool(self.model) and bool(self.base_url or self.api_key)

    def describe(self) -> dict[str, Any]:
        """Safe description for logs and metrics files: model and host, never the key."""
        host = urlparse(self.base_url).netloc if self.base_url else "default (api.openai.com)"
        return {"model": self.model, "endpoint_host": host, "configured": self.configured}


def llm_settings(environ: Mapping[str, str] | None = None) -> LLMSettings:
    """Read ``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL`` and ``LLM_RPM`` (empty = unset)."""
    env = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        value = (env.get(name) or "").strip()
        return value or None

    rpm = get("LLM_RPM")
    try:
        rpm_value = float(rpm) if rpm else None
    except ValueError as exc:
        raise ValueError(f"LLM_RPM must be a number, got {rpm!r}") from exc
    return LLMSettings(
        get("LLM_BASE_URL"),
        get("LLM_API_KEY"),
        get("LLM_MODEL"),
        rpm_value if rpm_value and rpm_value > 0 else None,
    )


def copilot_config(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ``copilot`` section of the merged configuration."""
    merged = dict(cfg) if cfg is not None else load_config()
    if "copilot" not in merged:
        raise KeyError("configs/default.yaml has no `copilot` section")
    return dict(merged["copilot"])
