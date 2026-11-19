"""LLM access for the copilot (PRD §9.1): an OpenAI-compatible client, a mock, a tool loop.

- :class:`OpenAICompatibleLLM` talks to any server that speaks the OpenAI chat-completions API
  (hosted providers, Ollama, vLLM …) with the ``openai`` package. Timeouts and retries
  (connection errors, 429, 5xx, with backoff) come from ``copilot.llm`` in the config. For
  free, rate-limited endpoints it can pace its own requests (``LLM_RPM`` or
  ``copilot.llm.requests_per_minute``) and waits out "429 Too Many Requests" answers, honouring
  the server's ``Retry-After``; errors that a later retry may fix are marked ``transient``.
- :class:`ScriptedLLM` returns predefined replies and tool calls in order and records every request.
  Unit tests and the evaluation's ``--mock`` mode use it; it never pretends to be a real model in
  result files.
- :func:`run_tool_loop` runs model → tools → model until the model answers without tool calls.

Messages use the OpenAI format (``{"role": …, "content": …}``, assistant ``tool_calls``, ``tool``
messages with ``tool_call_id``).
"""

from __future__ import annotations

import copy
import itertools
import json
import time
from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from leakwatch.common.logging import get_logger
from leakwatch.copilot.settings import LLMSettings

log = get_logger("leakwatch.copilot.llm")


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable.

    ``transient`` is True for failures a later attempt may not have (rate limit, timeout,
    connection error, server error), False for a rejected request or an unusable reply.
    """

    def __init__(self, message: str, transient: bool = False) -> None:
        """Create the error."""
        super().__init__(message)
        self.transient = transient


def retry_after_s(exc: BaseException) -> float | None:
    """Seconds from a ``Retry-After`` / ``retry-after-ms`` header of an API error, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        if headers.get("retry-after-ms"):
            return float(headers["retry-after-ms"]) / 1000.0
        if headers.get("retry-after"):
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        return None
    return None


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""

    def as_message_part(self) -> dict[str, Any]:
        """The ``tool_calls`` entry of an assistant message."""
        raw = self.raw_arguments or json.dumps(self.arguments, ensure_ascii=False)
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": raw},
        }


@dataclass(frozen=True)
class LLMResponse:
    """A model reply: text, tool calls, or both."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, Any] | None = None

    def as_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [c.as_message_part() for c in self.tool_calls]
        return msg


class LLMClient(Protocol):
    """What the copilot needs from a model."""

    model: str

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send the conversation (and tool schemas) and return the reply."""
        ...


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments as a dict; invalid JSON is kept under ``__invalid_json__``."""
    text = (raw or "").strip() or "{}"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"__invalid_json__": text}
    return value if isinstance(value, dict) else {"__invalid_json__": text}


def json_object_from_text(text: str) -> dict[str, Any]:
    """The JSON object in a model reply (tolerates code fences and prose around it).

    Raises:
        ValueError: no JSON object could be read.

    """
    s = (text or "").strip()
    try:
        value = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the reply") from None
        try:
            value = json.loads(s[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in the reply: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("the reply is not a JSON object")
    return value


class OpenAICompatibleLLM:
    """Chat completions with tool calling against an OpenAI-compatible endpoint."""

    def __init__(self, settings: LLMSettings, cfg: Mapping[str, Any]) -> None:
        """Create the client.

        Args:
            settings: endpoint and model (``settings.llm_settings()``).
            cfg: ``copilot.llm`` config (temperature, max_tokens, timeout_s, max_retries, seed,
                requests_per_minute, rate_limit_retries, rate_limit_max_wait_s).

        """
        if not settings.configured:
            raise LLMError("no LLM configured (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)")
        from openai import OpenAI

        self.model = str(settings.model)
        self.cfg = dict(cfg)
        self._client = OpenAI(
            base_url=settings.base_url or None,
            api_key=settings.api_key or "not-needed",  # local servers ignore the key
            timeout=float(cfg.get("timeout_s", 60)),
            max_retries=int(cfg.get("max_retries", 2)),
        )
        rpm = settings.rpm or cfg.get("requests_per_minute")
        self.min_interval_s = 60.0 / float(rpm) if rpm else 0.0
        self.wait_s = 0.0  # time spent pacing and waiting out rate limits (not model latency)
        self._last_start = -1e9

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)
            self.wait_s += seconds

    def _create(self, kwargs: Mapping[str, Any]) -> Any:
        """``chat.completions.create`` with pacing and patient retries on rate limits."""
        import openai

        retries = int(self.cfg.get("rate_limit_retries", 0))
        max_wait = float(self.cfg.get("rate_limit_max_wait_s", 60))
        for attempt in range(retries + 1):
            self._sleep(self._last_start + self.min_interval_s - time.monotonic())
            self._last_start = time.monotonic()
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                if attempt == retries:
                    raise LLMError(f"rate limited: {str(exc)[:300]}", transient=True) from exc
                delay = min(max_wait, retry_after_s(exc) or 5.0 * 2**attempt)
                log.warning(
                    "rate limited; waiting %.0f s (retry %d of %d)", delay, attempt + 1, retries
                )
                self._sleep(delay)
            except (openai.APIConnectionError, openai.InternalServerError) as exc:
                # APITimeoutError is an APIConnectionError
                raise LLMError(f"{type(exc).__name__}: {str(exc)[:300]}", transient=True) from exc
            except openai.APIError as exc:
                raise LLMError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        raise LLMError("unreachable")  # pragma: no cover

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        """One chat-completions call; raises :class:`LLMError` on any API failure."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "max_tokens": int(self.cfg.get("max_tokens", 1200)),
        }
        if self.cfg.get("seed") is not None:
            kwargs["seed"] = int(self.cfg["seed"])
        if tools:
            kwargs["tools"] = [dict(t) for t in tools]
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._create(kwargs)
        if not resp.choices:
            raise LLMError("the model returned no choices")
        msg = resp.choices[0].message
        calls = tuple(
            ToolCall(
                id=tc.id or f"call_{i}",
                name=tc.function.name,
                arguments=parse_arguments(tc.function.arguments),
                raw_arguments=tc.function.arguments or "{}",
            )
            for i, tc in enumerate(msg.tool_calls or [])
            if getattr(tc, "function", None) is not None
        )
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) is not None else None
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)


def build_llm(settings: LLMSettings, cfg: Mapping[str, Any]) -> OpenAICompatibleLLM | None:
    """The configured client, or None when no endpoint is set."""
    return OpenAICompatibleLLM(settings, cfg) if settings.configured else None


# ------------------------------------------------------------------------------------------
# Scripted model for tests
# ------------------------------------------------------------------------------------------

ScriptStep = (
    LLMResponse | Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], LLMResponse]
)
_call_ids = itertools.count(1)


def tool_call(name: str, **arguments: Any) -> ToolCall:
    """A tool call for a script (ids are unique within the process)."""
    return ToolCall(id=f"call_{next(_call_ids)}", name=name, arguments=arguments)


def calls(*tool_calls: ToolCall, content: str | None = None) -> LLMResponse:
    """A scripted reply that requests tool calls."""
    return LLMResponse(content=content, tool_calls=tuple(tool_calls))


def reply(text: str) -> LLMResponse:
    """A scripted final answer."""
    return LLMResponse(content=text)


class ScriptedLLM:
    """Deterministic stand-in for a model: returns the scripted steps in order.

    A step is an :class:`LLMResponse` or a function ``(messages, tools) -> LLMResponse`` (to answer
    from what the tools returned). Every request is recorded in :attr:`requests`.
    """

    def __init__(self, script: Iterable[ScriptStep], model: str = "scripted") -> None:
        """Create the mock with its steps."""
        self.script: list[ScriptStep] = list(script)
        self.model = model
        self.requests: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return the next scripted step (raises :class:`LLMError` when the script is used up)."""
        msgs = [copy.deepcopy(dict(m)) for m in messages]
        tool_list = [dict(t) for t in tools] if tools else None
        self.requests.append({"messages": msgs, "tools": tool_list, "json_mode": json_mode})
        if not self.script:
            raise LLMError("ScriptedLLM: script exhausted")
        step = self.script.pop(0)
        return step(msgs, tool_list) if callable(step) else step

    @property
    def remaining(self) -> int:
        """Steps not used yet."""
        return len(self.script)


def last_tool_output(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Parsed content of the most recent ``tool`` message (helper for scripted answers)."""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            try:
                value = json.loads(str(msg.get("content") or "{}"))
            except json.JSONDecodeError:
                return {"text": msg.get("content")}
            return value if isinstance(value, dict) else {"value": value}
    return {}


# ------------------------------------------------------------------------------------------
# Tool loop
# ------------------------------------------------------------------------------------------


@dataclass
class LoopOutcome:
    """How a tool loop ended."""

    content: str
    rounds: int
    forced_final: bool  # the round limit was hit and a final answer was requested without tools


def run_tool_loop(
    llm: LLMClient,
    messages: MutableSequence[dict[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    execute: Callable[[ToolCall], str],
    max_rounds: int,
) -> LoopOutcome:
    """Model → tools → model until the model answers without tool calls.

    Args:
        llm: the model.
        messages: conversation so far; assistant and tool messages are appended in place.
        tools: tool schemas (OpenAI format).
        execute: runs one tool call and returns the text sent back to the model.
        max_rounds: tool rounds allowed before a final answer is requested without tools.

    """
    for round_no in range(1, max_rounds + 1):
        resp = llm.chat(messages, tools)
        messages.append(resp.as_message())
        if not resp.tool_calls:
            return LoopOutcome(content=resp.content or "", rounds=round_no, forced_final=False)
        for call in resp.tool_calls:
            content = execute(call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    log.info("tool-round limit (%d) reached; asking for a final answer", max_rounds)
    messages.append(
        {
            "role": "user",
            "content": "Answer now with the tool results you already have; do not call tools.",
        }
    )
    resp = llm.chat(messages, None)
    messages.append(resp.as_message())
    return LoopOutcome(content=resp.content or "", rounds=max_rounds + 1, forced_final=True)
