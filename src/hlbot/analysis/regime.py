"""Macro-regime brief: condense recent headlines into a short market-climate summary
that primes the analyzer when it judges future breaking news.

Built once at startup (scripts/build_regime.py -> data/regime.md) and then refreshed
periodically while the bot runs (main.regime_refresh_loop), so big events that arrive
mid-session are folded into the prevailing-climate context for SUBSEQUENT news.
"""
from __future__ import annotations

import time

from anthropic import AsyncAnthropic

from ..config import Config
from . import usage_ledger
from .prompts import supports_temperature


def format_recent_catalysts(rows: list[dict]) -> str:
    """Render already-traded catalysts (from store.recent_entered_catalysts) into a
    compact anti-re-trade memory block for the analyzer prompt."""
    if not rows:
        return ""
    now = time.time() * 1000
    lines = []
    for r in rows:
        age_h = max(0.0, (now - r["ts"]) / 3_600_000)
        ago = f"{age_h / 24:.0f}d ago" if age_h >= 24 else f"{age_h:.0f}h ago"
        reason = (r.get("reason") or "").strip().replace("\n", " ")[:120]
        lines.append(f"- {r['symbol']} {r['side']} · \"{reason}\" · {ago}")
    return "\n".join(lines)

REGIME_PROMPT = """You are a macro strategist. From the news headlines below (most recent \
last), write a concise market-regime brief (<= 250 words) that a trading bot will use as \
context when judging FUTURE breaking news. Cover: overall risk tone; crypto regime \
(trend/themes); equities/AI regime; any ONGOING geopolitical conflicts and whether they are \
an ambiguous back-and-forth stalemate (low edge) vs a decisive turning point; commodity \
drivers. Describe the prevailing CLIMATE/REGIME — do not just list individual events. Output \
ONLY the brief (markdown bullets), no preamble."""


async def summarize_headlines(cfg: Config, lines: list[str],
                              model: str = "claude-sonnet-4-6", max_items: int = 1200,
                              ledger_path: str | None = None) -> str:
    """Summarize raw headline strings (most recent last) into a regime brief via Claude.
    Pass ledger_path (live) to record the call's tokens; omit it (backtest) to skip."""
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    # Bounded timeout (house policy — the SDK default is 10 minutes) and a context manager so
    # the httpx pool is closed instead of leaking one per 12h refresh / backtest window.
    async with AsyncAnthropic(api_key=cfg.secrets.anthropic_api_key, timeout=60.0) as client:
        kwargs: dict = {"temperature": 0.0} if supports_temperature(model) else {}
        resp = await client.messages.create(
            model=model, max_tokens=700,
            messages=[{"role": "user",
                       "content": REGIME_PROMPT + "\n\nHEADLINES:\n"
                                  + "\n".join(lines[-max_items:])}],
            **kwargs,
        )
    # "#infra" key: shared plumbing, not the model's own analysis spend (see Analyzer._record).
    usage_ledger.record(ledger_path, f"{model}#infra", resp)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
