"""Prefix-based model routing: `provider:model_id` -> the right Provider.

Bare strings and `claude-*` / `anthropic:*` are Anthropic and handled NATIVELY in analyzer.py
(not here). One Provider instance is built per kind and cached on the router for the session.
"""
from __future__ import annotations

from .base import Provider, ProviderConfigError
from .gemini_provider import GeminiProvider
from .openai_compat import OpenAICompatProvider

# Friendly aliases -> canonical provider kind (so MODELS / config can use either form).
_ALIASES = {"grok": "xai", "glm": "zhipu", "gpt": "openai", "gemini": "google",
            "claude": "anthropic"}

# OpenAI-compatible kinds. Verified against each vendor's live API via
# scripts/smoke_providers.py. base_url=None => OpenAI's default endpoint. Notes:
#   - openai: GPT-5/o-series want max_completion_tokens and reject a custom temperature.
#   - deepseek (V4 Pro): thinking mode rejects every forced tool_choice -> "json" mode.
#   - zhipu (GLM 5.2): thinking DISABLED via extra_body. z.ai's reasoning path intermittently
#     drops the tool call under sustained sequential load (systematic empty responses that stall
#     the run); the direct/non-thinking path is reliable + faster. This is the ONE entrant not at
#     provider-default reasoning — a documented consistency asterisk, accepted to make GLM usable.
# Otherwise every entrant runs at its provider default reasoning (no reasoning_effort set).
_OPENAI_KINDS = {
    "openai": {"key": "openai_api_key", "base_url": None,
               "token_param": "max_completion_tokens", "temperature": False,
               "tool_choice": "named"},
    "deepseek": {"key": "deepseek_api_key", "base_url": "https://api.deepseek.com",
                 "token_param": "max_tokens", "temperature": True, "tool_choice": "json"},
    "xai": {"key": "xai_api_key", "base_url": "https://api.x.ai/v1",
            "token_param": "max_tokens", "temperature": True, "tool_choice": "named"},
    "zhipu": {"key": "zhipu_api_key", "base_url": "https://api.z.ai/api/paas/v4",
              "token_param": "max_tokens", "temperature": True, "tool_choice": "named",
              "extra_body": {"thinking": {"type": "disabled"}}},
}


def split_model(model_string: str) -> tuple[str, str]:
    """'openai:gpt-5.4' -> ('openai', 'gpt-5.4'); bare / 'claude-...' -> ('anthropic', <as-is>)."""
    if ":" in model_string:
        prov, bare = model_string.split(":", 1)
        return _ALIASES.get(prov.lower(), prov.lower()), bare
    return "anthropic", model_string


def is_anthropic(model_string: str) -> bool:
    """True for the native Claude path (bare strings, `claude-*`, or an explicit `anthropic:`)."""
    return split_model(model_string)[0] == "anthropic"


class ProviderRouter:
    """Resolves a model string to (Provider, bare_model_id), building each provider once and
    caching it. Keys/base_urls come from `secrets`; retry/timeout knobs from the AnalyzerConfig."""

    def __init__(self, secrets, cfg):
        self.secrets = secrets
        self.cfg = cfg
        self._cache: dict[str, Provider] = {}

    def provider_for(self, model_string: str) -> tuple[Provider, str]:
        kind, bare = split_model(model_string)
        if kind == "anthropic":
            raise ProviderConfigError(
                "Anthropic models are served by the native analyzer path, not the router")
        prov = self._cache.get(kind)
        if prov is None:
            prov = self._build(kind)
            self._cache[kind] = prov
        return prov, bare

    def _key(self, attr: str) -> str:
        key = getattr(self.secrets, attr, "") or ""
        if not key:
            raise ProviderConfigError(f"{attr.upper()} not set in .env")
        return key

    def _build(self, kind: str) -> Provider:
        common = dict(max_retries=self.cfg.max_retries,
                      retry_base_delay=self.cfg.retry_base_delay,
                      timeout=self.cfg.request_timeout_seconds)
        if kind == "google":
            return GeminiProvider(api_key=self._key("gemini_api_key"), **common)
        spec = _OPENAI_KINDS.get(kind)
        if spec is None:
            raise ProviderConfigError(f"unknown provider kind {kind!r}")
        return OpenAICompatProvider(api_key=self._key(spec["key"]), base_url=spec["base_url"],
                                    token_param=spec["token_param"],
                                    send_temperature=spec["temperature"],
                                    tool_choice_mode=spec["tool_choice"],
                                    extra_body=spec.get("extra_body"), **common)
