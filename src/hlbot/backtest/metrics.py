"""Pure portfolio metrics over a list of BTTrade: equity curve, drawdown, profit
factor, holding-time and exposure stats. Total PnL alone can't rank two runs — a
config that makes $900 with a $1,400 trough is worse for a $10k account with a
$1.5k daily kill switch than one that makes $700 smoothly. No I/O; unit-tested.

Drawdown here is REALIZED (PnL booked at trade close): it understates the
mark-to-market trough of an open 72h position. Treat it as a lower bound; an
open-PnL curve from candles is a possible later upgrade.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable


def _day(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


def equity_curve(trades: Iterable) -> list[tuple[str, float]]:
    """(utc_day, cumulative realized PnL) per day with at least one close, by exit time."""
    daily: dict[str, float] = {}
    for t in trades:
        daily[_day(t.exit_ms)] = daily.get(_day(t.exit_ms), 0.0) + t.pnl
    out, run = [], 0.0
    for day in sorted(daily):
        run += daily[day]
        out.append((day, run))
    return out


def max_drawdown(curve: list[tuple[str, float]]) -> float:
    """Largest peak-to-trough fall ($) of the cumulative realized curve (>= 0).
    The peak starts at 0 (flat account), so an immediate losing streak counts."""
    peak, dd = 0.0, 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dd


def profit_factor(trades: Iterable) -> float:
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl < 0)
    if gross_loss <= 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def expectancy(trades: list) -> float:
    return sum(t.pnl for t in trades) / len(trades) if trades else 0.0


def holding_stats(trades: list) -> dict:
    """Median hold (hours) overall and per exit reason, with each reason's win rate —
    answers 'is the time exit a winner or a loser bucket?'."""
    def med(vals: list[float]) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    hold_h = lambda t: (t.exit_ms - t.time_ms) / 3_600_000
    by_reason: dict[str, list] = {}
    for t in trades:
        by_reason.setdefault(t.reason, []).append(t)
    return {
        "median_hold_h": med([hold_h(t) for t in trades]),
        "by_reason": {
            reason: {"n": len(ts), "median_hold_h": med([hold_h(t) for t in ts]),
                     "win_rate": sum(1 for t in ts if t.pnl > 0) / len(ts),
                     "total": sum(t.pnl for t in ts)}
            for reason, ts in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))
        },
    }


def exposure_curve(trades: list) -> dict:
    """Peak concurrent notional and position count from [entry, exit] intervals — the
    capital the strategy actually demanded, without any candle fetches."""
    events: list[tuple[int, float, int]] = []
    for t in trades:
        events.append((t.time_ms, t.notional, 1))
        events.append((t.exit_ms, -t.notional, -1))
    events.sort()
    notional = count = 0.0
    peak_notional, peak_count = 0.0, 0
    for _, dn, dc in events:
        notional += dn
        count += dc
        peak_notional = max(peak_notional, notional)
        peak_count = max(peak_count, int(count))
    return {"peak_notional": peak_notional, "peak_concurrent": peak_count}


def summary_metrics(trades: list, account_size_usd: float = 0.0) -> dict:
    """The headline risk metrics for a run, JSON-friendly (inf -> None for pf)."""
    curve = equity_curve(trades)
    dd = max_drawdown(curve)
    pf = profit_factor(trades)
    hold = holding_stats(trades)
    expo = exposure_curve(trades)
    return {
        "profit_factor": None if pf == float("inf") else round(pf, 3),
        "max_dd_usd": round(dd, 2),
        "max_dd_pct_of_account": (round(dd / account_size_usd * 100, 2)
                                  if account_size_usd else None),
        "expectancy_usd": round(expectancy(trades), 2),
        "median_hold_h": round(hold["median_hold_h"], 2),
        "hold_by_reason": hold["by_reason"],
        "peak_notional": round(expo["peak_notional"], 2),
        "peak_concurrent": expo["peak_concurrent"],
        "equity_curve": [(d, round(v, 2)) for d, v in curve],
    }
