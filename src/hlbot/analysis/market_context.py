"""Compact tape summary for the entry-confirmation call: what has this asset's price
actually done lately? Reconstructed from candles ending at a given timestamp, so the SAME
function serves live (at_ms = now) and the backtest replay (at_ms = the news time) with
no live/backtest divergence.

The first-pass analyzer deliberately sees NO market data (rules-only, replayable); the
skeptic confirmer gets this block so its second opinion is decorrelated — it can argue
"this is already up 12% on the day, the catalyst is priced in" with the tape in hand.

Lookahead safety: only bars whose CLOSE time is <= at_ms are used — the bar in progress
at the news instant would otherwise leak post-news price into "current". "price" is
therefore up to one bar stale; the precise post-news move arrives separately as the
deterministic pre_move_pct measured by the risk engine's already-moved guard.
"""
from __future__ import annotations

import logging

log = logging.getLogger("hlbot.market_context")

_INTERVAL = "15m"
_INTERVAL_MS = 15 * 60_000


def format_context(market, candles: list[dict], at_ms: int) -> str:
    """Render the tape block from raw candles (pure; unit-tested). '' when there isn't
    enough completed history to say anything honest (caller omits the block)."""
    bars = []
    for c in candles or []:
        try:
            if int(c["T"]) <= at_ms:
                bars.append((float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(bars) < 4:   # under an hour of completed bars: no meaningful tape
        return ""
    o0 = bars[0][0]
    last_o, _, _, price = bars[-1]
    hi24 = max(b[1] for b in bars)
    lo24 = min(b[2] for b in bars)
    lines = [f"MARKET TAPE for {market.name} (24h of completed {_INTERVAL} bars up to the news):"]
    parts = [f"price {price:g}"]
    if o0 > 0:
        parts.append(f"24h change {(price - o0) / o0:+.1%}")
    if last_o > 0:
        parts.append(f"last {_INTERVAL} {(price - last_o) / last_o:+.1%}")
    lines.append(" · ".join(parts))
    if hi24 > 0 and lo24 > 0:
        lines.append(f"24h range {lo24:g}-{hi24:g} "
                     f"(now {max(0.0, (hi24 - price) / hi24):.1%} below the high, "
                     f"{max(0.0, (price - lo24) / lo24):+.1%} above the low)")
    vol = getattr(market, "day_volume_usd", 0.0) or 0.0
    if vol > 0:
        lines.append(f"24h volume ~${vol / 1e6:.0f}M")
    return "\n".join(lines)


async def build_market_context(hl, market, at_ms: int) -> str:
    """Fetch ~24h of bars ending at at_ms and render the block. Fails soft to '' on any
    fetch error — the confirmer then judges without tape rather than blocking the trade."""
    try:
        candles = await hl.candles(market.name, _INTERVAL,
                                   int(at_ms - 24 * 3600_000 - _INTERVAL_MS), int(at_ms))
    except Exception:  # noqa: BLE001
        log.warning("market-context candle fetch failed for %s", market.name)
        return ""
    return format_context(market, candles, at_ms)
