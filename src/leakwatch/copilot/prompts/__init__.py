"""Prompt versions (PRD §9.6) and the task prompts for work orders and complaints.

- ``v1.yaml`` zero-shot: role and task only.
- ``v2.yaml`` few-shot: v1 + guidance + worked examples (``few_shot.yaml``).
- ``v3.yaml`` few-shot + schema + self-correction + guardrails: v2 + the lake schema in the prompt,
  SQL error hints and retries, and the answer guardrails of ``copilot.guardrails``.

Placeholders in prompt text are ``{name}`` tokens replaced by :func:`fill` (only the names given;
other braces, such as JSON in examples, are left alone).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent
VERSIONS: tuple[str, ...] = ("v1", "v2", "v3")


def fill(template: str, values: Mapping[str, Any]) -> str:
    """Replace ``{name}`` for each given name."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


@dataclass(frozen=True)
class FewShotExample:
    """One worked example: a question, the tool calls made, and the answer."""

    question: str
    steps: tuple[dict[str, Any], ...]
    answer: str
    versions: tuple[str, ...]

    def render(self, number: int) -> str:
        """The example as prompt text."""
        lines = [f"Example {number}", f"Operator: {self.question}"]
        for step in self.steps:
            args = json.dumps(step.get("args", {}), ensure_ascii=False)
            lines.append(f"Tool call: {step['tool']}({args})")
            if step.get("result") is not None:
                lines.append(f"Tool result: {step['result']}")
        lines.append(f"Answer: {self.answer}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PromptSpec:
    """One prompt version."""

    version: str
    technique: str
    description: str
    system: str
    include_schema: bool
    sql_hints: bool
    guardrails: bool
    max_sql_retries: int
    examples: tuple[FewShotExample, ...] = field(default_factory=tuple)

    def render_system(self, values: Mapping[str, Any]) -> str:
        """System prompt with placeholders filled and the examples appended."""
        text = fill(self.system, values).rstrip()
        if self.examples:
            shots = "\n\n".join(ex.render(i) for i, ex in enumerate(self.examples, start=1))
            text += "\n\nWorked examples (ids, numbers and times are illustrative):\n\n" + shots
        return text


def _read(name: str) -> dict[str, Any]:
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no prompt file {path.name}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return data


@lru_cache(maxsize=8)
def load_prompt(version: str) -> PromptSpec:
    """Load ``v1``, ``v2`` or ``v3``."""
    if version not in VERSIONS:
        raise ValueError(f"unknown prompt version {version!r}; use one of {', '.join(VERSIONS)}")
    data = _read(version)
    examples: list[FewShotExample] = []
    if data.get("few_shot"):
        for item in _read(str(data["few_shot"])).get("examples", []):
            versions = tuple(item.get("versions") or VERSIONS)
            if version not in versions:
                continue
            examples.append(
                FewShotExample(
                    question=str(item["question"]),
                    steps=tuple(item.get("steps") or ()),
                    answer=str(item["answer"]).strip(),
                    versions=versions,
                )
            )
    return PromptSpec(
        version=version,
        technique=str(data["technique"]),
        description=str(data["description"]).strip(),
        system=str(data["system"]),
        include_schema=bool(data.get("include_schema", False)),
        sql_hints=bool(data.get("sql_hints", False)),
        guardrails=bool(data.get("guardrails", False)),
        max_sql_retries=int(data.get("max_sql_retries", 0)),
        examples=tuple(examples),
    )


@lru_cache(maxsize=8)
def load_named_prompt(name: str) -> dict[str, Any]:
    """A task prompt file (``work_order``, ``complaint``) as a dict."""
    return _read(name)


def few_shot_sql() -> list[str]:
    """Every SQL query used in the few-shot examples (tests check them against the guard)."""
    out = []
    for item in _read("few_shot").get("examples", []):
        for step in item.get("steps") or []:
            if step.get("tool") == "run_sql":
                out.append(str(step["args"]["query"]))
    return out
