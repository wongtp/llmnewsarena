"""Provider abstraction for the multi-LLM arena: a uniform tool-calling contract over the
non-Anthropic model providers.

The Anthropic/Claude path stays NATIVE in analyzer.py (unchanged — it keeps prompt caching,
strict-tool, effort, the owned retry loop and the existing test coverage). These adapters add
OpenAI-compatible (GPT, DeepSeek, xAI/Grok, Zhipu/GLM) and Gemini behind one `call_tool()`
contract so the analyzer can route any non-Anthropic model identically. Usage is normalized to
Anthropic's attribute names so `pricing.response_cost_usd` and `usage_ledger.record` consume a
provider result with NO changes.
"""
from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

log = logging.getLogger("hlbot.providers")


class ProviderConfigError(RuntimeError):
    """A model was routed to a provider whose API key/spec is missing or unknown. Raised at
    routing time; Analyzer.analyze() catches it and degrades to a 'none' verdict, so a
    misconfigured arena entrant becomes a missed trade rather than a pipeline crash."""


class ProviderResponseError(RuntimeError):
    """The provider returned NO usable structured output (empty/truncated/refused/parse-fail).
    Distinct from a valid 'none' verdict (which is a non-empty tool result). Raised so the
    analyzer flags it as an error and the backtest does NOT cache it — otherwise a truncated
    thinking-model response would be stored as a permanent false 'none' and skipped on resume.
    Not auto-retried (a retry hits the same cap); a resume re-runs it, and the backtest's
    consecutive-failure breaker stops the run if it's systemic (e.g. out of output budget)."""


@dataclass
class NormalizedUsage:
    """Token usage normalized to Anthropic's attribute names so `response_cost_usd()` and
    `usage_ledger.record()` consume it unchanged. Each provider maps its own taxonomy onto
    these four fields (input EXCLUDES cache reads, matching Anthropic semantics)."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class CallResult:
    """A provider call's structured tool output plus its normalized usage. `.usage` mirrors the
    shape of an Anthropic response's `.usage`, so `Analyzer._record()` handles it identically."""
    tool_input: dict
    usage: NormalizedUsage


def system_text(system_blocks) -> str:
    """Flatten Anthropic-style system blocks ([{type,text,cache_control}, ...]) into one plain
    string for providers that take a single system message (cache_control is Anthropic-only).
    Accepts a plain string too."""
    if isinstance(system_blocks, str):
        return system_blocks
    parts = []
    for b in system_blocks or []:
        if isinstance(b, dict) and b.get("text"):
            parts.append(b["text"])
        elif isinstance(b, str):
            parts.append(b)
    return "\n\n".join(parts)


def is_retryable(exc: Exception) -> bool:
    """Transient-failure predicate shared by the provider retry loops (mirrors
    analyzer._is_retryable): retry connection/timeout blips, rate limits and 5xx; fail fast on
    other 4xx (a 400 bad-request won't fix itself). Matches by SDK class name AND status code so
    it works across the OpenAI and google-genai error hierarchies without importing either."""
    if isinstance(exc, ProviderResponseError):
        return True   # an empty/dropped tool call is usually a transient model hiccup — retry it
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                "InternalServerError", "ServerError"):
        return True
    status = (getattr(exc, "status_code", None) or getattr(exc, "code", None)
              or getattr(getattr(exc, "response", None), "status_code", None))
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    return False


class Provider(ABC):
    def __init__(self, *, max_retries: int = 4, retry_base_delay: float = 0.5):
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def supports_temperature(self, model: str) -> bool:  # noqa: ARG002 - overridden per provider
        return True

    @abstractmethod
    async def call_tool(self, *, model: str, system_blocks, user_text: str, tool: dict,
                        tool_name: str, max_tokens: int,
                        temperature: float | None = None) -> CallResult:
        """Force the model to return `tool`'s structured output; return its parsed arguments
        (best-effort dict — never raise on a malformed/empty result; the analyzer's
        _to_analysis sanitizer is the backstop) plus normalized usage."""

    async def _retry(self, make_call):
        """Run make_call() with bounded exponential backoff (mirrors Analyzer._create): a
        one-shot breaking-news catalyst can't be re-fetched, so survive transient blips; a
        non-retryable error raises immediately on the first try."""
        attempts = max(1, self.max_retries)
        for i in range(1, attempts + 1):
            try:
                return await make_call()
            except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
                if i >= attempts or not is_retryable(exc):
                    raise
                delay = self.retry_base_delay * (2 ** (i - 1)) * (0.5 + random.random())
                log.warning("%s call failed (%s); retry %d/%d in %.1fs",
                            type(self).__name__, type(exc).__name__, i, attempts - 1, delay)
                await asyncio.sleep(delay)
