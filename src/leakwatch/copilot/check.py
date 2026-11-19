"""``make llm-check``: a one-minute test that the configured LLM can serve the copilot.

Run it after putting a provider's ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` (and, for a
free tier, ``LLM_RPM``) into ``.env`` and before the long ``make pe-eval``. Two probes:

1. **Tool calling.** One request with the copilot's real tool schemas and a question that needs
   the lake. Passes when the model answers with a call to one of those tools and arguments that
   parse as JSON (some models or servers reject the schemas or reply in plain text instead).
2. **End to end.** One data question through the full v3 agent (tools, SQL on the lake,
   guardrails), as the dashboard and the evaluation ask it. Skipped when the lake is not built.

It prints the latency of each probe and a rough duration of the full evaluation at the
configured pace. Nothing is written to ``reports/``.
"""

from __future__ import annotations

import sys
import time

from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.copilot.agent import CopilotAgent
from leakwatch.copilot.llm import LLMError, build_llm
from leakwatch.copilot.prompts import VERSIONS
from leakwatch.copilot.settings import NOT_CONFIGURED_MESSAGE, llm_settings
from leakwatch.copilot.tools import ToolContext, ToolRegistry

log = get_logger("leakwatch.copilot.check")

TOOL_QUESTION = "How many leak alerts are stored in the lake? Use a tool to find out."
AGENT_QUESTION = "How many alerts are in the alerts table?"
CALLS_PER_ITEM = 3  # model calls per evaluation item, a rough figure for the duration estimate


def main(argv: list[str] | None = None) -> int:
    """Run the probes; exit 0 when both pass, 1 when one fails, 2 when no model is configured."""
    settings = llm_settings()
    if not settings.configured:
        log.error("%s", NOT_CONFIGURED_MESSAGE)
        return 2
    cfg = load_config()
    llm = build_llm(settings, cfg["copilot"]["llm"])
    desc = settings.describe()
    log.info(
        "model %s at %s, pace %s requests/min",
        desc["model"],
        desc["endpoint_host"],
        settings.rpm or cfg["copilot"]["llm"].get("requests_per_minute") or "unlimited",
    )
    ok = True

    ctx = ToolContext(full_cfg=cfg)
    schemas = ToolRegistry(ctx).schemas()
    names = {s["function"]["name"] for s in schemas}
    t0 = time.perf_counter()
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": "You answer with tool calls when data is needed."},
                {"role": "user", "content": TOOL_QUESTION},
            ],
            schemas,
        )
        called = [c.name for c in resp.tool_calls]
        good = bool(called) and all(n in names for n in called)
        log.info(
            "probe 1 tool calling: %s in %.1f s (called %s)",
            "OK" if good else "FAILED",
            time.perf_counter() - t0,
            called or "no tool; replied with text",
        )
        ok &= good
    except LLMError as exc:
        log.error("probe 1 tool calling: FAILED (%s)", exc)
        ok = False

    try:
        ans = CopilotAgent(llm, ctx, "v3").ask(AGENT_QUESTION)
        good = ans.error is None and bool(ans.tool_calls) and not ans.refused
        log.info(
            "probe 2 end to end: %s in %.1f s, tools %s, answer: %s",
            "OK" if good else "FAILED",
            ans.latency_s,
            [c.name for c in ans.tool_calls],
            ans.answer[:200].replace("\n", " "),
        )
        if ans.error:
            log.error("probe 2 error: %s", ans.error)
        ok &= good
    except Exception as exc:  # the lake may not be built on this machine
        log.warning("probe 2 skipped: %s: %s", type(exc).__name__, str(exc)[:200])

    rpm = settings.rpm or cfg["copilot"]["llm"].get("requests_per_minute")
    if rpm:
        from leakwatch.copilot.evaluate import load_items

        calls = len(load_items()) * len(VERSIONS) * CALLS_PER_ITEM
        log.info(
            "full evaluation: about %d model calls -> at least %.0f min at %s requests/min",
            calls,
            calls / float(rpm),
            rpm,
        )
    log.info("verdict: %s", "ready for `make pe-eval`" if ok else "fix the failures above first")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
