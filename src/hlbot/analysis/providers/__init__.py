"""Multi-LLM provider adapters for the news-trading arena. See base.py for the contract.

Anthropic/Claude is served natively by analyzer.py; this package adds OpenAI-compatible
(GPT, DeepSeek, xAI/Grok, Zhipu/GLM) and Gemini providers behind one `call_tool()` interface,
routed by `ProviderRouter` on a `provider:model_id` prefix.
"""
from __future__ import annotations

from .base import (
    CallResult,
    NormalizedUsage,
    Provider,
    ProviderConfigError,
    ProviderResponseError,
    is_retryable,
    system_text,
)
from .gemini_provider import GeminiProvider, clean_schema
from .openai_compat import OpenAICompatProvider, to_function_tool
from .registry import ProviderRouter, is_anthropic, split_model

__all__ = [
    "CallResult", "NormalizedUsage", "Provider", "ProviderConfigError", "ProviderResponseError",
    "is_retryable", "system_text", "GeminiProvider", "clean_schema", "OpenAICompatProvider",
    "to_function_tool", "ProviderRouter", "is_anthropic", "split_model",
]
