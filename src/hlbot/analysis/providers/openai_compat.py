"""OpenAI Chat Completions-compatible provider. One adapter covers OpenAI (GPT), DeepSeek,
xAI/Grok and Zhipu/GLM — they differ only by base_url + key (+ a couple of param quirks the
registry sets). Structured output comes from forced function-calling; provider-side strict
schema validation is NOT required (uneven across these vendors) because the analyzer's
_to_analysis sanitizer already clamps/validates every field and a content-JSON fallback
covers vendors that ignore a forced tool.
"""
from __future__ import annotations

import json
import logging

from .base import CallResult, NormalizedUsage, Provider, ProviderResponseError, system_text

log = logging.getLogger("hlbot.providers.openai")


def to_function_tool(tool: dict) -> dict:
    """Anthropic tool ({name, description, input_schema}) -> OpenAI function-tool definition.
    The JSON Schema is reused verbatim (ANALYSIS_TOOL is already strict-compatible)."""
    return {"type": "function",
            "function": {"name": tool["name"],
                         "description": tool.get("description", ""),
                         "parameters": tool["input_schema"]}}


def _loads(s) -> dict:
    """Parse a JSON object string to a dict; {} on anything malformed/empty/non-object."""
    if not s:
        return {}
    try:
        out = json.loads(s)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _json_instruction(tool: dict) -> str:
    """Schema instruction appended to the system prompt in JSON mode. Must contain the literal
    'json' (DeepSeek's response_format=json_object requires it). Delivers the SAME schema the
    function-calling models get via their tool definition, just as text."""
    return ("Respond with ONLY a single JSON object conforming to this JSON Schema "
            "(no markdown, no commentary, no other text):\n"
            + json.dumps(tool["input_schema"]))


class OpenAICompatProvider(Provider):
    def __init__(self, *, api_key: str, base_url: str | None = None,
                 token_param: str = "max_tokens", send_temperature: bool = True,
                 tool_choice_mode: str = "named", extra_body: dict | None = None,
                 max_retries: int = 4, retry_base_delay: float = 0.5,
                 timeout: float = 30.0, client=None):
        super().__init__(max_retries=max_retries, retry_base_delay=retry_base_delay)
        # Optional vendor-specific request extras merged into every call (e.g. a thinking-budget
        # toggle). Unused by default: every arena entrant runs at its provider's DEFAULT reasoning
        # for cross-model consistency; this is just a hook for any future per-vendor need.
        self.extra_body = extra_body
        # GPT-5/o-series want `max_completion_tokens` and reject `max_tokens`; the OpenAI-compat
        # vendors (DeepSeek/xAI/Zhipu) take `max_tokens`. Set per kind by the registry.
        self.token_param = token_param
        # GPT-5 reasoning models reject a custom temperature; the others honor 0.0 for
        # determinism. The registry sets this per kind.
        self.send_temperature = send_temperature
        # How to get structured output:
        #   "named"    = force this specific function (default; GPT/Grok/GLM)
        #   "required" = force SOME tool call (we expose only one, so it's still ours)
        #   "auto"     = let the model decide (unreliable on thinking models)
        #   "json"     = no tools; response_format=json_object + schema in the prompt, parse
        #                content. Needed by DeepSeek V4 Pro, whose thinking mode rejects every
        #                forced tool_choice but emits clean JSON reliably.
        self.tool_choice_mode = tool_choice_mode
        if client is not None:
            self.client = client   # tests inject a fake; no SDK / network needed
        else:
            from openai import AsyncOpenAI   # lazy: package imports fine without the SDK present
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url,
                                      max_retries=0, timeout=timeout)

    def supports_temperature(self, model: str) -> bool:  # noqa: ARG002
        return self.send_temperature

    async def call_tool(self, *, model, system_blocks, user_text, tool, tool_name,
                        max_tokens, temperature=None) -> CallResult:
        if self.tool_choice_mode == "json":
            return await self._call_json(model, system_blocks, user_text, tool, max_tokens,
                                         temperature)
        tool_choice = ({"type": "function", "function": {"name": tool_name}}
                       if self.tool_choice_mode == "named" else self.tool_choice_mode)
        kwargs = {
            "model": model,
            self.token_param: max_tokens,
            "messages": [{"role": "system", "content": system_text(system_blocks)},
                         {"role": "user", "content": user_text}],
            "tools": [to_function_tool(tool)],
            "tool_choice": tool_choice,
        }
        if temperature is not None and self.send_temperature:
            kwargs["temperature"] = temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        async def _attempt():
            resp = await self.client.chat.completions.create(**kwargs)
            ti = self._parse(resp, tool_name)
            if not ti:   # empty/dropped tool call -> retryable (intermittent on thinking models)
                raise ProviderResponseError(f"{model}: empty tool output (truncated/refused/unparseable)")
            return resp, ti

        resp, tool_input = await self._retry(_attempt)
        return CallResult(tool_input, self._usage(resp))

    async def _call_json(self, model, system_blocks, user_text, tool, max_tokens,
                         temperature) -> CallResult:
        """JSON-mode structured output: no tools, response_format=json_object, the tool's JSON
        Schema appended to the system text. For thinking models that reject forced tool_choice
        but reliably emit a clean JSON object (DeepSeek V4 Pro). The reasoning goes to a separate
        reasoning_content field; message.content holds the JSON."""
        sys_text = system_text(system_blocks) + "\n\n" + _json_instruction(tool)
        kwargs = {
            "model": model,
            self.token_param: max_tokens,
            "messages": [{"role": "system", "content": sys_text},
                         {"role": "user", "content": user_text}],
            "response_format": {"type": "json_object"},
        }
        if temperature is not None and self.send_temperature:
            kwargs["temperature"] = temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        async def _attempt():
            resp = await self.client.chat.completions.create(**kwargs)
            try:
                content = resp.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                content = None
            ti = _loads(content)
            if not ti:   # empty/unparseable JSON -> retryable
                raise ProviderResponseError(f"{model}: empty JSON output (truncated/refused/unparseable)")
            return resp, ti

        resp, tool_input = await self._retry(_attempt)
        return CallResult(tool_input, self._usage(resp))

    @staticmethod
    def _parse(resp, tool_name) -> dict:
        try:
            msg = resp.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return {}
        for call in (getattr(msg, "tool_calls", None) or []):
            fn = getattr(call, "function", None)
            if fn is not None and getattr(fn, "name", None) == tool_name:
                return _loads(getattr(fn, "arguments", None))
        # Some compat vendors ignore a forced tool and put the JSON in message.content instead.
        return _loads(getattr(msg, "content", None))

    @staticmethod
    def _usage(resp) -> NormalizedUsage:
        u = getattr(resp, "usage", None)
        if not u:
            return NormalizedUsage()
        prompt = getattr(u, "prompt_tokens", 0) or 0
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return NormalizedUsage(
            input_tokens=max(0, prompt - cached),   # match Anthropic: input excludes cache reads
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=0,
        )
