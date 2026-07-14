"""Gemini provider (google-genai). Forced function-calling for structured output; usage from
`usage_metadata`. Gemini's schema dialect is an OpenAPI subset that rejects
`additionalProperties`, so the tool schema is cleaned before sending. Config is passed as plain
dicts (the SDK coerces them) so this module imports without `google.genai.types` and stays
unit-testable with an injected fake client.
"""
from __future__ import annotations

import asyncio
import logging

from .base import CallResult, NormalizedUsage, Provider, ProviderResponseError, system_text

log = logging.getLogger("hlbot.providers.gemini")


def clean_schema(schema):
    """Recursively drop JSON-Schema keys Gemini's OpenAPI subset rejects (additionalProperties)."""
    if isinstance(schema, dict):
        return {k: clean_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [clean_schema(v) for v in schema]
    return schema


class GeminiProvider(Provider):
    def __init__(self, *, api_key: str, max_retries: int = 4, retry_base_delay: float = 0.5,
                 timeout: float = 30.0, client=None):
        super().__init__(max_retries=max_retries, retry_base_delay=retry_base_delay)
        # Enforced per attempt via asyncio.wait_for (SDK-agnostic): a wedged Gemini call must
        # not hold an arena semaphore slot indefinitely. TimeoutError is retryable in _retry.
        self.timeout = timeout
        if client is not None:
            self.client = client   # tests inject a fake; no SDK / network needed
        else:
            from google import genai   # lazy: package imports fine without the SDK present
            self.client = genai.Client(api_key=api_key)

    async def call_tool(self, *, model, system_blocks, user_text, tool, tool_name,
                        max_tokens, temperature=None) -> CallResult:
        fd = {"name": tool["name"], "description": tool.get("description", ""),
              "parameters": clean_schema(tool["input_schema"])}
        config = {
            "system_instruction": system_text(system_blocks),
            "tools": [{"function_declarations": [fd]}],
            "tool_config": {"function_calling_config": {
                "mode": "ANY", "allowed_function_names": [tool_name]}},
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            config["temperature"] = temperature
        async def _attempt():
            resp = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=model, contents=user_text, config=config),
                self.timeout)
            ti = self._parse(resp, tool_name)
            if not ti:   # empty/dropped function call -> retryable
                raise ProviderResponseError(f"{model}: empty function-call output (truncated/refused)")
            return resp, ti

        resp, tool_input = await self._retry(_attempt)
        return CallResult(tool_input, self._usage(resp))

    @staticmethod
    def _parse(resp, tool_name) -> dict:
        try:
            parts = resp.candidates[0].content.parts or []
        except (AttributeError, IndexError, TypeError):
            return {}
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None) == tool_name:
                args = getattr(fc, "args", None)
                try:
                    return dict(args) if args else {}
                except (TypeError, ValueError):
                    return {}
        return {}

    @staticmethod
    def _usage(resp) -> NormalizedUsage:
        um = getattr(resp, "usage_metadata", None)
        if not um:
            return NormalizedUsage()
        prompt = getattr(um, "prompt_token_count", 0) or 0
        cached = getattr(um, "cached_content_token_count", 0) or 0
        # thoughts_token_count is billed as output but reported separately from candidates.
        thoughts = getattr(um, "thoughts_token_count", 0) or 0
        return NormalizedUsage(
            input_tokens=max(0, prompt - cached),
            output_tokens=(getattr(um, "candidates_token_count", 0) or 0) + thoughts,
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=0,
        )
