"""Multi-LLM provider adapters: schema translation, response/usage parsing, prefix routing,
and the analyzer's non-Anthropic call path. All offline — fake SDK clients are injected, so no
network and no real keys. The Anthropic/Claude path is unchanged and covered by
test_speed_cost / test_confirm / test_analyzer_retry."""
import asyncio
import json
import types

import pytest

from hlbot.analysis.analyzer import Analyzer
from hlbot.analysis.pricing import response_cost_usd
from hlbot.analysis.prompts import ANALYSIS_TOOL
from hlbot.analysis.providers import (
    CallResult,
    GeminiProvider,
    NormalizedUsage,
    OpenAICompatProvider,
    ProviderConfigError,
    ProviderResponseError,
    ProviderRouter,
    clean_schema,
    is_anthropic,
    split_model,
    to_function_tool,
)
from hlbot.models import NewsItem, now_ms


# ---------------------------------------------------------------- routing ----

def test_split_model_and_is_anthropic():
    assert split_model("openai:gpt-5.4") == ("openai", "gpt-5.4")
    assert split_model("grok:grok-4.3") == ("xai", "grok-4.3")          # alias -> xai
    assert split_model("glm:glm-5.2-max") == ("zhipu", "glm-5.2-max")   # alias -> zhipu
    assert split_model("gemini:gemini-3.5-flash") == ("google", "gemini-3.5-flash")
    assert split_model("claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")
    assert split_model("anthropic:claude-opus-4-8") == ("anthropic", "claude-opus-4-8")
    assert is_anthropic("claude-sonnet-4-6") and is_anthropic("anthropic:claude-opus-4-8")
    assert not is_anthropic("openai:gpt-5.4") and not is_anthropic("xai:grok-4.3")


def _secrets():
    return types.SimpleNamespace(openai_api_key="k", gemini_api_key="k", deepseek_api_key="k",
                                 xai_api_key="k", zhipu_api_key="k")


def _cfg():
    return types.SimpleNamespace(max_retries=4, retry_base_delay=0.0, request_timeout_seconds=30.0,
                                 max_tokens=1024, temperature=0.0)


def test_router_builds_openai_compat_with_per_kind_params():
    r = ProviderRouter(_secrets(), _cfg())
    p_openai, bare = r.provider_for("openai:gpt-5.4")
    assert bare == "gpt-5.4"
    assert isinstance(p_openai, OpenAICompatProvider)
    assert p_openai.token_param == "max_completion_tokens"   # GPT-5/o-series
    assert p_openai.send_temperature is False                # rejects custom temperature
    assert p_openai.supports_temperature("gpt-5.4") is False

    p_ds, _ = r.provider_for("deepseek:deepseek-v4-pro")
    assert p_ds.token_param == "max_tokens" and p_ds.send_temperature is True
    assert p_ds.tool_choice_mode == "json"            # thinking mode rejects forced tool_choice
    assert "deepseek.com" in str(p_ds.client.base_url)

    p_grok, _ = r.provider_for("grok:grok-4.3")
    assert "x.ai" in str(p_grok.client.base_url)

    p_glm, _ = r.provider_for("glm:glm-5.2")
    assert p_glm.tool_choice_mode == "named"
    # GLM thinking DISABLED: z.ai's reasoning path drops tool calls under load (the one asterisk)
    assert p_glm.extra_body == {"thinking": {"type": "disabled"}}


def test_router_builds_gemini_and_caches_per_kind():
    r = ProviderRouter(_secrets(), _cfg())
    p, bare = r.provider_for("google:gemini-3.5-flash")
    assert isinstance(p, GeminiProvider) and bare == "gemini-3.5-flash"
    # one instance per kind, cached for the session
    assert r.provider_for("google:gemini-2.0-flash")[0] is p


def test_router_rejects_anthropic_and_missing_key():
    r = ProviderRouter(_secrets(), _cfg())
    with pytest.raises(ProviderConfigError):
        r.provider_for("claude-sonnet-4-6")          # native path, never the router
    r2 = ProviderRouter(types.SimpleNamespace(), _cfg())   # no keys at all
    with pytest.raises(ProviderConfigError):
        r2.provider_for("openai:gpt-5.4")


# ---------------------------------------------------- schema translation ----

def test_to_function_tool_shape():
    ft = to_function_tool(ANALYSIS_TOOL)
    assert ft["type"] == "function"
    assert ft["function"]["name"] == "submit_analysis"
    assert ft["function"]["parameters"] is ANALYSIS_TOOL["input_schema"]


def test_clean_schema_strips_additional_properties_recursively():
    cleaned = clean_schema(ANALYSIS_TOOL["input_schema"])
    assert "additionalProperties" not in json.dumps(cleaned)   # nested object too
    # enums and required survive (they're valid in Gemini's OpenAPI subset)
    assert cleaned["properties"]["direction"]["enum"] == ["long", "short", "none"]
    assert "ticker" in cleaned["required"]


# --------------------------------------------------- OpenAI-compat calls ----

def _oa_resp(*, tool_calls=None, content=None, prompt=120, completion=30, cached=20):
    msg = types.SimpleNamespace(tool_calls=tool_calls, content=content)
    usage = types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                  prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached))
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def _oa_tool_call(name, args_obj):
    fn = types.SimpleNamespace(name=name, arguments=json.dumps(args_obj))
    return types.SimpleNamespace(function=fn)


def _fake_oa_client(resp, seen):
    async def create(**kw):
        seen.append(kw)
        return resp
    return types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))


def test_openai_compat_parses_tool_call_and_usage_and_kwargs():
    seen = []
    args = {"ticker": "BTC", "direction": "long", "confidence": 0.9}
    client = _fake_oa_client(_oa_resp(tool_calls=[_oa_tool_call("submit_analysis", args)]), seen)
    p = OpenAICompatProvider(api_key="k", token_param="max_tokens", send_temperature=True,
                             client=client)
    res = asyncio.run(p.call_tool(model="grok-4.3", system_blocks=[{"type": "text", "text": "sys"}],
                                  user_text="news", tool=ANALYSIS_TOOL,
                                  tool_name="submit_analysis", max_tokens=512, temperature=0.0))
    assert res.tool_input == args
    # usage: input excludes cached reads (Anthropic semantics), cached -> cache_read
    assert res.usage.input_tokens == 100 and res.usage.cache_read_input_tokens == 20
    assert res.usage.output_tokens == 30
    kw = seen[0]
    assert kw["model"] == "grok-4.3" and kw["max_tokens"] == 512        # token_param honored
    assert kw["tool_choice"] == {"type": "function", "function": {"name": "submit_analysis"}}
    assert kw["tools"][0]["function"]["name"] == "submit_analysis"
    assert kw["messages"][0]["role"] == "system" and kw["messages"][0]["content"] == "sys"
    assert kw["temperature"] == 0.0


def test_openai_compat_token_param_and_temperature_gating():
    seen = []
    client = _fake_oa_client(_oa_resp(tool_calls=[_oa_tool_call("submit_analysis", {"x": 1})]), seen)
    p = OpenAICompatProvider(api_key="k", token_param="max_completion_tokens",
                             send_temperature=False, client=client)
    asyncio.run(p.call_tool(model="gpt-5.4", system_blocks="sys", user_text="n", tool=ANALYSIS_TOOL,
                            tool_name="submit_analysis", max_tokens=777, temperature=0.0))
    kw = seen[0]
    assert kw["max_completion_tokens"] == 777 and "max_tokens" not in kw
    assert "temperature" not in kw                 # send_temperature False -> never sent


def test_openai_compat_sends_extra_body():
    seen = []
    client = _fake_oa_client(_oa_resp(tool_calls=[_oa_tool_call("submit_analysis", {"ticker": "BTC"})]),
                             seen)
    p = OpenAICompatProvider(api_key="k", extra_body={"thinking": {"type": "disabled"}}, client=client)
    asyncio.run(p.call_tool(model="glm-5.2", system_blocks="s", user_text="n", tool=ANALYSIS_TOOL,
                            tool_name="submit_analysis", max_tokens=10))
    assert seen[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_compat_content_json_fallback():
    # A vendor that ignores the forced tool and emits JSON in message.content.
    client = _fake_oa_client(_oa_resp(tool_calls=None, content='{"ticker": "ETH", "direction": "short"}'), [])
    p = OpenAICompatProvider(api_key="k", client=client)
    res = asyncio.run(p.call_tool(model="m", system_blocks="s", user_text="n", tool=ANALYSIS_TOOL,
                                  tool_name="submit_analysis", max_tokens=10))
    assert res.tool_input == {"ticker": "ETH", "direction": "short"}


def test_openai_compat_json_mode_omits_tools_and_parses_content():
    # DeepSeek V4 Pro path: no tools, response_format=json_object, schema appended to the prompt.
    seen = []
    client = _fake_oa_client(
        _oa_resp(tool_calls=None, content='{"ticker": "NVDA", "direction": "long", "confidence": 0.9}'),
        seen)
    p = OpenAICompatProvider(api_key="k", tool_choice_mode="json", client=client)
    res = asyncio.run(p.call_tool(model="deepseek-v4-pro", system_blocks="base sys", user_text="news",
                                  tool=ANALYSIS_TOOL, tool_name="submit_analysis", max_tokens=512,
                                  temperature=0.0))
    assert res.tool_input == {"ticker": "NVDA", "direction": "long", "confidence": 0.9}
    kw = seen[0]
    assert kw["response_format"] == {"type": "json_object"}
    assert "tools" not in kw and "tool_choice" not in kw
    sys_msg = kw["messages"][0]["content"]
    assert "base sys" in sys_msg                       # original system preserved
    assert "json" in sys_msg.lower()                   # required by DeepSeek's json mode
    assert "ticker" in sys_msg and "direction" in sys_msg   # schema fields delivered as text


def test_openai_compat_persistent_empty_raises_not_cached():
    # Always-empty output: after retries it must RAISE (not return {}), so the engine flags it as
    # an error and never caches it as a false 'none' (which would be skipped forever on resume).
    client = _fake_oa_client(_oa_resp(tool_calls=None, content="not json"), [])
    p = OpenAICompatProvider(api_key="k", client=client, max_retries=2, retry_base_delay=0)
    with pytest.raises(ProviderResponseError):
        asyncio.run(p.call_tool(model="m", system_blocks="s", user_text="n", tool=ANALYSIS_TOOL,
                                tool_name="submit_analysis", max_tokens=10))


def test_openai_compat_retries_empty_then_recovers():
    # GLM/z.ai scenario: the tool call is intermittently dropped; a retry succeeds. The empty
    # response must NOT fail the analysis when the next attempt returns a valid verdict.
    calls = {"n": 0}
    good = _oa_resp(tool_calls=[_oa_tool_call("submit_analysis", {"ticker": "BTC", "direction": "long"})])
    empty = _oa_resp(tool_calls=None, content=None)

    async def create(**kw):
        calls["n"] += 1
        return empty if calls["n"] == 1 else good

    client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    p = OpenAICompatProvider(api_key="k", client=client, max_retries=4, retry_base_delay=0)
    res = asyncio.run(p.call_tool(model="glm-5.2", system_blocks="s", user_text="n", tool=ANALYSIS_TOOL,
                                  tool_name="submit_analysis", max_tokens=10))
    assert res.tool_input == {"ticker": "BTC", "direction": "long"} and calls["n"] == 2


def test_openai_compat_retries_on_transient(monkeypatch):
    calls = {"n": 0}

    class Boom(Exception):
        status_code = 503

    async def create(**_kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise Boom()
        return _oa_resp(tool_calls=[_oa_tool_call("submit_analysis", {"ticker": "BTC"})])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    p = OpenAICompatProvider(api_key="k", client=client, max_retries=4, retry_base_delay=0.0)
    res = asyncio.run(p.call_tool(model="m", system_blocks="s", user_text="n", tool=ANALYSIS_TOOL,
                                  tool_name="submit_analysis", max_tokens=10))
    assert res.tool_input == {"ticker": "BTC"} and calls["n"] == 2


# ----------------------------------------------------------- Gemini calls ----

def _fake_gemini_client(resp, seen):
    async def gen(**kw):
        seen.append(kw)
        return resp
    return types.SimpleNamespace(aio=types.SimpleNamespace(
        models=types.SimpleNamespace(generate_content=gen)))


def test_gemini_parses_function_call_and_usage_and_config():
    seen = []
    fc = types.SimpleNamespace(name="submit_analysis",
                               args={"ticker": "BTC", "direction": "long", "confidence": 0.8})
    part = types.SimpleNamespace(function_call=fc)
    cand = types.SimpleNamespace(content=types.SimpleNamespace(parts=[part]))
    um = types.SimpleNamespace(prompt_token_count=200, candidates_token_count=40,
                               cached_content_token_count=50)
    resp = types.SimpleNamespace(candidates=[cand], usage_metadata=um)
    p = GeminiProvider(api_key="k", client=_fake_gemini_client(resp, seen))
    res = asyncio.run(p.call_tool(model="gemini-3.5-flash",
                                  system_blocks=[{"type": "text", "text": "sys"}],
                                  user_text="news", tool=ANALYSIS_TOOL,
                                  tool_name="submit_analysis", max_tokens=512, temperature=0.0))
    assert res.tool_input == {"ticker": "BTC", "direction": "long", "confidence": 0.8}
    assert res.usage.input_tokens == 150 and res.usage.cache_read_input_tokens == 50
    assert res.usage.output_tokens == 40
    cfg = seen[0]["config"]
    assert cfg["system_instruction"] == "sys"
    assert cfg["tool_config"]["function_calling_config"]["mode"] == "ANY"
    assert "additionalProperties" not in json.dumps(cfg["tools"])      # schema cleaned
    assert seen[0]["model"] == "gemini-3.5-flash"


def test_gemini_no_function_call_raises():
    # No function call (e.g. truncated/refused) -> raise so it isn't cached as a false 'none'.
    part = types.SimpleNamespace(function_call=None)
    cand = types.SimpleNamespace(content=types.SimpleNamespace(parts=[part]))
    resp = types.SimpleNamespace(candidates=[cand], usage_metadata=None)
    p = GeminiProvider(api_key="k", client=_fake_gemini_client(resp, []), max_retries=2,
                       retry_base_delay=0)
    with pytest.raises(ProviderResponseError):
        asyncio.run(p.call_tool(model="m", system_blocks="s", user_text="n", tool=ANALYSIS_TOOL,
                                tool_name="submit_analysis", max_tokens=10))


def test_gemini_counts_thinking_tokens_as_output():
    # Gemini bills thoughts_token_count as output but reports it SEPARATELY from
    # candidates_token_count — dropping it undercounts arena cost in Gemini's favor.
    fc = types.SimpleNamespace(name="submit_analysis", args={"direction": "none"})
    cand = types.SimpleNamespace(content=types.SimpleNamespace(
        parts=[types.SimpleNamespace(function_call=fc)]))
    um = types.SimpleNamespace(prompt_token_count=100, candidates_token_count=40,
                               cached_content_token_count=0, thoughts_token_count=500)
    resp = types.SimpleNamespace(candidates=[cand], usage_metadata=um)
    p = GeminiProvider(api_key="k", client=_fake_gemini_client(resp, []))
    res = asyncio.run(p.call_tool(model="m", system_blocks="s", user_text="n",
                                  tool=ANALYSIS_TOOL, tool_name="submit_analysis",
                                  max_tokens=10))
    assert res.usage.output_tokens == 540


def test_gemini_wedged_call_times_out_instead_of_hanging():
    # A hung Gemini call must not hold an arena semaphore slot forever: the per-attempt
    # asyncio.wait_for turns it into a (retryable) TimeoutError that eventually raises.
    async def hang(**kw):
        await asyncio.sleep(3600)

    client = types.SimpleNamespace(aio=types.SimpleNamespace(
        models=types.SimpleNamespace(generate_content=hang)))
    p = GeminiProvider(api_key="k", client=client, timeout=0.01, max_retries=2,
                       retry_base_delay=0)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(p.call_tool(model="m", system_blocks="s", user_text="n",
                                tool=ANALYSIS_TOOL, tool_name="submit_analysis",
                                max_tokens=10))


# ------------------------------------------------- analyzer routing + mapping ----

def _news():
    return NewsItem("n", "t", "b", "X", None, now_ms(), now_ms())


def _analyzer_with_provider(provider, bare="gpt-5.4"):
    a = object.__new__(Analyzer)   # bypass __init__: no real client/keys
    a.cfg = types.SimpleNamespace(max_tokens=1024, temperature=0.0)
    a.router = types.SimpleNamespace(provider_for=lambda model: (provider, bare))
    a._record = lambda model, result: 0.0   # shadow ledger/usage accounting
    return a


def test_analyzer_call_routes_non_anthropic_to_provider():
    captured = {}

    class FakeProvider:
        def supports_temperature(self, model):
            return True

        async def call_tool(self, **kw):
            captured.update(kw)
            return CallResult({"ticker": "BTC", "direction": "long"}, NormalizedUsage(10, 5))

    a = _analyzer_with_provider(FakeProvider())
    out = asyncio.run(a._call("openai:gpt-5.4", [{"type": "text", "text": "sys"}], "news body"))
    assert out == {"ticker": "BTC", "direction": "long"}
    assert captured["model"] == "gpt-5.4"          # prefix stripped for the SDK
    assert captured["tool_name"] == "submit_analysis"
    assert captured["temperature"] == 0.0


def test_analyzer_omits_temperature_when_provider_rejects_it():
    captured = {}

    class FakeProvider:
        def supports_temperature(self, model):
            return False                            # e.g. GPT-5 reasoning

        async def call_tool(self, **kw):
            captured.update(kw)
            return CallResult({}, NormalizedUsage())

    a = _analyzer_with_provider(FakeProvider())
    asyncio.run(a._call("openai:gpt-5.4", "sys", "news"))
    assert captured["temperature"] is None


def test_provider_path_output_is_sanitized_by_mapping():
    # A non-Anthropic model leaks tool-call markup into the ticker; _to_analysis must discard it
    # (strict schema validation isn't required of these providers — the mapper is the backstop).
    class FakeProvider:
        def supports_temperature(self, model):
            return True

        async def call_tool(self, **kw):
            return CallResult({"ticker": '</X>\n<PARAM>NONE', "direction": "long",
                               "asset_class": "equity", "confidence": 0.9}, NormalizedUsage())

    a = _analyzer_with_provider(FakeProvider())
    raw = asyncio.run(a._call_provider("openai:gpt-5.4", [{"type": "text", "text": "s"}], "n"))
    res = Analyzer._to_analysis(a, _news(), raw, "openai:gpt-5.4")
    assert res.ticker is None                       # markup garbage discarded
    assert res.direction == "long"                  # rest of the verdict preserved
    assert res.model == "openai:gpt-5.4"


# --------------------------------------------------------------- pricing ----

def test_response_cost_usd_for_arena_models():
    one_m = NormalizedUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(response_cost_usd("openai:gpt-5.4", one_m) - (2.50 + 15.0)) < 1e-6
    assert abs(response_cost_usd("grok-4.3", one_m) - (1.25 + 2.50)) < 1e-6
    # cached reads price at the cache_read column, not full input
    cached = NormalizedUsage(input_tokens=0, output_tokens=0, cache_read_input_tokens=1_000_000)
    assert abs(response_cost_usd("xai:grok-4.3", cached) - 0.20) < 1e-6


def test_counts_cost_usd_prices_infra_key_by_base_model():
    # Plumbing calls (keep-warm, regime briefs) ledger under "<model>#infra"; they must
    # price at the BASE model's rates, not the unknown-model Sonnet fallback.
    from hlbot.analysis.pricing import counts_cost_usd

    one_m = {"input": 1_000_000, "output": 0, "cache_read": 0, "cache_creation": 0}
    assert abs(counts_cost_usd("deepseek:deepseek-v4-pro#infra", one_m) - 0.28) < 1e-9
    assert (counts_cost_usd("claude-sonnet-4-6#infra", one_m)
            == counts_cost_usd("claude-sonnet-4-6", one_m))
