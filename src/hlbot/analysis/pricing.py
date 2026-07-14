"""Claude price sheet shared by live per-analysis cost tracking, the backtest
report and scripts/token_report.py — keep it in ONE place so a price change
can't drift between the dashboards.

Approx Anthropic list prices, USD per million tokens: (input, output, cache_write,
cache_read). cache_write uses the 1h-TTL rate (2x input): the analyzer caches with
cache_ttl "1h" and the usage counters don't split 5m (1.25x) vs 1h (2x) writes, so
price at what the bot actually pays. Unknown models fall back to Sonnet pricing.
"""
from __future__ import annotations

PRICING = {
    "claude-sonnet-4-6": (3.0, 15.0, 6.0, 0.30),
    "claude-haiku-4-5-20251001": (1.0, 5.0, 2.0, 0.10),
    "claude-haiku-4-5": (1.0, 5.0, 2.0, 0.10),
    "claude-opus-4-8": (5.0, 25.0, 10.0, 0.50),
    # --- Arena entrants (non-Anthropic). Keyed by the prefixed routing string AND a bare
    # alias so cost resolves whichever form is logged. ESTIMATES — verify against each
    # vendor's live price sheet (CONFIRM). These providers have no separate cache-WRITE rate
    # and the adapters never report cache_creation tokens, so cache_write := input; cache_read
    # is the vendor's cached-input rate (the analyzer's NormalizedUsage splits it out).
    "openai:gpt-5.4": (2.50, 15.0, 2.50, 0.25),
    "gpt-5.4": (2.50, 15.0, 2.50, 0.25),
    "google:gemini-3.5-flash": (1.50, 9.0, 1.50, 0.375),
    "gemini-3.5-flash": (1.50, 9.0, 1.50, 0.375),
    "deepseek:deepseek-v4-pro": (0.28, 0.42, 0.28, 0.028),
    "deepseek-v4-pro": (0.28, 0.42, 0.28, 0.028),
    "xai:grok-4.3": (1.25, 2.50, 1.25, 0.20),
    "grok-4.3": (1.25, 2.50, 1.25, 0.20),
    "zhipu:glm-5.2": (1.0, 3.20, 1.0, 0.20),
    "glm-5.2": (1.0, 3.20, 1.0, 0.20),
}
_DEFAULT_PRICE = (3.0, 15.0, 6.0, 0.30)


def counts_cost_usd(model: str, counts: dict) -> float:
    """Estimated $ cost from a usage-ledger counts dict
    ({input, output, cache_creation, cache_read} token totals). Ledger keys may carry a
    "#infra" suffix (plumbing calls split out of per-model spend) — price by base model."""
    pin, pout, pcw, pcr = PRICING.get(model.split("#", 1)[0], _DEFAULT_PRICE)
    return ((counts.get("input", 0) or 0) * pin
            + (counts.get("output", 0) or 0) * pout
            + (counts.get("cache_creation", 0) or 0) * pcw
            + (counts.get("cache_read", 0) or 0) * pcr) / 1e6


def response_cost_usd(model: str, usage) -> float:
    """Estimated $ cost of a single API response from its usage block (0.0 if absent)."""
    if not usage:
        return 0.0
    pin, pout, pcw, pcr = PRICING.get(model, _DEFAULT_PRICE)
    return ((getattr(usage, "input_tokens", 0) or 0) * pin
            + (getattr(usage, "output_tokens", 0) or 0) * pout
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * pcw
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * pcr) / 1e6
