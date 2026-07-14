"""Speed/cost hardening: model-aware sampling params (Opus 4.7+/Fable reject temperature),
strict-tool + effort fallbacks, and the prompt-cache keep-warm gating. All offline — no
client construction, same style as test_analyzer_retry."""
import asyncio
import types

import httpx
import pytest
from anthropic import BadRequestError

from hlbot.analysis.analyzer import Analyzer
from hlbot.analysis.prompts import ANALYSIS_TOOL, supports_temperature


def _bad_request(msg="bad"):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def _cfg(**over):
    base = dict(model_fast="claude-sonnet-4-6", model_smart="claude-opus-4-8",
                temperature=0.0, max_tokens=1024, max_retries=1, retry_base_delay=0.0,
                strict_tool=True, effort="", cache_ttl="1h", include_crypto_universe=True,
                triage_model="claude-haiku-4-5-20251001")
    base.update(over)
    return types.SimpleNamespace(**base)


def _analyzer(create, **cfg_over):
    a = object.__new__(Analyzer)  # bypass __init__ (no real client / api key)
    a.cfg = _cfg(**cfg_over)
    a.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    a.regime_context = "risk-on"
    a.exemplars = ""
    a.context_bridge = ""
    a.recent_catalysts = ""
    a._strict_ok = True
    a._effort_ok = True
    a._warm_key = None
    a._prefix_used_at = 0.0
    a._record = lambda model, resp, **kw: None   # shadow the method; no ledger/usage in these tests
    return a


def _tool_resp(**input_kw):
    block = types.SimpleNamespace(type="tool_use", name="submit_analysis",
                                  input=input_kw or {"ticker": "BTC"})
    return types.SimpleNamespace(content=[block], usage=None)


def test_supports_temperature_by_model_family():
    assert supports_temperature("claude-sonnet-4-6")
    assert supports_temperature("claude-haiku-4-5-20251001")
    assert not supports_temperature("claude-opus-4-8")
    assert not supports_temperature("claude-opus-4-7")
    assert not supports_temperature("claude-fable-5")


def test_analysis_tool_schema_is_strict_compatible():
    schema = ANALYSIS_TOOL["input_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])   # strict mode needs all fields


def test_call_omits_temperature_on_opus():
    seen = []

    async def create(**kw):
        seen.append(kw)
        return _tool_resp()

    a = _analyzer(create)
    sys_blocks = [{"type": "text", "text": "s"}]
    asyncio.run(a._call("claude-sonnet-4-6", sys_blocks, "news"))
    asyncio.run(a._call("claude-opus-4-8", sys_blocks, "news"))
    assert seen[0]["temperature"] == 0.0          # sonnet keeps deterministic sampling
    assert "temperature" not in seen[1]           # opus 4.8 would 400 on it


def test_call_sends_strict_tool_and_falls_back_on_400():
    seen = []

    async def create(**kw):
        seen.append(kw)
        if any(t.get("strict") for t in kw.get("tools", [])):
            raise _bad_request("strict not supported")
        return _tool_resp()

    a = _analyzer(create)
    out = asyncio.run(a._call("claude-sonnet-4-6", [{"type": "text", "text": "s"}], "news"))
    assert out == {"ticker": "BTC"}               # the retry without strict succeeded
    assert seen[0]["tools"][0]["strict"] is True
    assert "strict" not in seen[1]["tools"][0]
    assert a._strict_ok is False                  # disabled for the session
    assert "strict" not in a._tools()[0]          # subsequent calls (and warms) match


def test_genuine_400_still_raises():
    async def create(**_kw):
        raise _bad_request("malformed")

    a = _analyzer(create, strict_tool=False)      # no optional features sent
    with pytest.raises(BadRequestError):
        asyncio.run(a._call("claude-sonnet-4-6", [{"type": "text", "text": "s"}], "news"))


def test_effort_attached_via_extra_body_except_haiku():
    seen = []

    async def create(**kw):
        seen.append(kw)
        return _tool_resp()

    a = _analyzer(create, effort="low")
    asyncio.run(a._call("claude-sonnet-4-6", [{"type": "text", "text": "s"}], "news"))
    assert seen[0]["extra_body"] == {"output_config": {"effort": "low"}}
    assert a._extra_body("claude-haiku-4-5-20251001") is None   # unsupported on haiku
    a._effort_ok = False
    assert a._extra_body("claude-sonnet-4-6") is None           # disabled after a rejection


def _universe(eq=("MRVL",), cr=("BTC",)):
    return types.SimpleNamespace(equity_symbols=lambda: list(eq),
                                 crypto_symbols=lambda: list(cr))


def test_keepwarm_warms_once_then_skips_until_idle_or_prefix_change():
    calls = []

    async def create(**kw):
        calls.append(kw)
        return types.SimpleNamespace(content=[], usage=None)

    a = _analyzer(create)
    u = _universe()
    assert asyncio.run(a.maybe_warm_cache(u, 2700)) is True     # cold start -> warm
    assert asyncio.run(a.maybe_warm_cache(u, 2700)) is False    # fresh -> skip
    a.regime_context = "risk-off"                               # prefix changed -> re-warm now
    assert asyncio.run(a.maybe_warm_cache(u, 2700)) is True
    a._prefix_used_at -= 9999                                   # idle past the interval -> re-warm
    assert asyncio.run(a.maybe_warm_cache(u, 2700)) is True
    a.context_bridge = "- 2025-09: thing"                       # bridge is part of the prefix too
    assert asyncio.run(a.maybe_warm_cache(u, 2700)) is True
    # Warm requests must match the real-call prefix (same tools) but use the prewarm idiom.
    assert calls[0]["max_tokens"] == 0
    assert calls[0]["model"] == "claude-sonnet-4-6"
    assert calls[0]["tools"][0]["strict"] is True
    assert "tool_choice" not in calls[0]                        # forced tool rejects max_tokens=0


def test_keepwarm_falls_back_to_one_token_if_zero_rejected():
    calls = []

    async def create(**kw):
        calls.append(kw)
        if kw["max_tokens"] == 0:
            raise _bad_request("max_tokens must be > 0")
        return types.SimpleNamespace(content=[], usage=None)

    a = _analyzer(create)
    assert asyncio.run(a.maybe_warm_cache(_universe(), 2700)) is True
    assert [c["max_tokens"] for c in calls] == [0, 1]


def test_real_call_counts_as_prefix_use_for_keepwarm():
    calls = []

    async def create(**kw):
        calls.append(kw)
        return _tool_resp()

    a = _analyzer(create)
    u = _universe()
    asyncio.run(a.maybe_warm_cache(u, 2700))                    # initial warm
    asyncio.run(a._call("claude-sonnet-4-6", [{"type": "text", "text": "s"}], "news"))
    n = len(calls)
    assert asyncio.run(a.maybe_warm_cache(u, 2700)) is False    # real call refreshed the TTL
    assert len(calls) == n
