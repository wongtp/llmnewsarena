"""Connectivity + structured-output smoke test for the arena's non-Anthropic providers.

Makes ONE real, cheap analysis call per provider (GPT / Gemini / DeepSeek / GLM / Grok) using
the production ANALYSIS_TOOL schema, then verifies the result maps to a valid Analysis. Confirms
auth, base_url, the function-calling translation, and usage/cost parsing before we spend on the
training backtests. Needs the 5 keys in .env. No Hyperliquid connection required.

    .venv/bin/python scripts/smoke_providers.py            # all arena models
    .venv/bin/python scripts/smoke_providers.py gpt grok   # a subset (keys = MODELS short names)
"""
from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console chokes on unicode otherwise
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.analyzer import Analyzer  # noqa: E402
from hlbot.analysis.pricing import response_cost_usd  # noqa: E402
from hlbot.analysis.prompts import ANALYSIS_TOOL  # noqa: E402
from hlbot.analysis.providers import ProviderRouter  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.models import NewsItem, now_ms  # noqa: E402

# Arena entrants (matches scripts/backtest.py MODELS, non-Anthropic only).
ARENA = {
    "gpt": "openai:gpt-5.4",
    "gemini": "google:gemini-3.5-flash",
    "deepseek": "deepseek:deepseek-v4-pro",
    "glm": "zhipu:glm-5.2",
    "grok": "xai:grok-4.3",
}

SYSTEM = (
    "You are a news analyst for a trading bot. Given a market headline, decide the tradable "
    "ticker, direction (long/short/none), and a 0..1 confidence. The tradable universe includes "
    "NVDA, AMD, TSLA, BTC, ETH. Return ONLY via the submit_analysis tool."
)
HEADLINE = ("NVIDIA reports Q3 revenue of $35.1B, up 94% YoY, beating consensus of $33.2B; "
            "guides Q4 revenue to ~$37.5B vs ~$35.9B expected.")


async def one(router: ProviderRouter, label: str, model: str) -> bool:
    provider, bare = router.provider_for(model)
    t0 = time.perf_counter()
    result = await provider.call_tool(
        model=bare, system_blocks=SYSTEM, user_text=HEADLINE,
        tool=ANALYSIS_TOOL, tool_name="submit_analysis",
        max_tokens=2048,   # generous: GPT-5 reasoning tokens count toward the budget
        temperature=(0.0 if provider.supports_temperature(bare) else None))
    dt_ms = int((time.perf_counter() - t0) * 1000)
    # Map through the exact production path so we know the verdict is usable downstream.
    item = NewsItem("smoke", "NVDA earnings", HEADLINE, "smoke", None, now_ms(), now_ms())
    a = Analyzer._to_analysis(object.__new__(Analyzer), item, result.tool_input, model)
    u = result.usage
    cost = response_cost_usd(model, u)
    ok = a.direction in ("long", "short", "none") and bool(result.tool_input)
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label:9s} {model:28s} -> {a.direction:5s} {a.ticker or '-':6s} "
          f"conf={a.confidence:.2f}  {dt_ms:5d}ms  in/out/cache={u.input_tokens}/"
          f"{u.output_tokens}/{u.cache_read_input_tokens}  ${cost:.5f}")
    if not ok:
        print(f"         raw tool_input={result.tool_input!r}")
    return ok


async def main() -> None:
    config = Config()
    router = ProviderRouter(config.secrets, config.app.analyzer)
    wanted = sys.argv[1:] or list(ARENA)
    print(f"Smoke-testing {len(wanted)} provider(s): {', '.join(wanted)}\n")
    results = {}
    for label in wanted:
        model = ARENA.get(label)
        if model is None:
            print(f"  [SKIP] {label}: unknown (choices: {', '.join(ARENA)})")
            continue
        try:
            results[label] = await one(router, label, model)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next provider
            results[label] = False
            print(f"  [FAIL] {label:9s} {model:28s} -> {type(exc).__name__}: {exc}")
    npass = sum(1 for v in results.values() if v)
    print(f"\n{npass}/{len(results)} provider(s) passed.")
    sys.exit(0 if npass == len(results) and results else 1)


if __name__ == "__main__":
    asyncio.run(main())
