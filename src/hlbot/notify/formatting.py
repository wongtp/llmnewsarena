"""HTML message templates for Telegram alerts."""
from __future__ import annotations

import html
import time

from ..models import Position


def _e(text) -> str:
    return html.escape(str(text or ""))


def _num(x: float) -> str:
    return f"{x:,.4f}".rstrip("0").rstrip(".") if x else "0"


def _tag(dry_run: bool) -> str:
    return "🧪 DRY-RUN" if dry_run else "🔴 LIVE"


def format_entry(pos: Position) -> str:
    side = "🟢 LONG" if pos.is_long else "🔴 SHORT"
    lines = [
        f"<b>{_tag(pos.dry_run)} ENTRY</b>",
        f"{side} <b>{_e(pos.symbol)}</b> <code>{_e(pos.market)}</code>",
        f"Confidence: <b>{pos.confidence:.0%}</b>",
        f"Size: {_num(pos.size)} (${pos.notional_usd:,.0f}) @ {_num(pos.entry_px)} · x{pos.leverage}",
        f"🛑 SL {_num(pos.stop_loss)}   🎯 TP {_num(pos.take_profit)}",
        f"💡 {_e(pos.rationale)}",
    ]
    if pos.news_source:
        lines.append(f"📰 {_e(pos.news_source)}")
    if pos.news_title:
        lines.append(f"<i>{_e(pos.news_title)[:200]}</i>")
    if pos.link:
        lines.append(f'<a href="{_e(pos.link)}">→ news link</a>')
    return "\n".join(lines)


def format_exit(pos: Position) -> str:
    sign = "✅" if pos.pnl_usd >= 0 else "❌"
    return "\n".join([
        f"<b>{_tag(pos.dry_run)} EXIT</b> {sign}",
        f"<b>{_e(pos.symbol)}</b> {pos.side.upper()} closed @ {_num(pos.exit_px)}",
        f"PnL: <b>${pos.pnl_usd:,.2f}</b>",
        f"Reason: {_e(pos.exit_reason)}",
    ])


def format_error(payload: dict) -> str:
    return f"⚠️ <b>Order error</b> {_e(payload.get('market'))}: {_e(payload.get('error'))}"


def _dur(ms: float) -> str:
    """Compact duration: 90061000 -> '25h 1m'; negatives (overdue time exit) -> 'now'."""
    s = ms / 1000
    if s <= 0:
        return "now"
    if s < 3600:
        return f"{s / 60:.0f}m"
    if s < 48 * 3600:
        return f"{int(s // 3600)}h {int(s % 3600 // 60)}m"
    return f"{s / 86400:.1f}d"


def _usd(v: float) -> str:
    return f"{'+' if v >= 0 else '−'}${abs(v):,.2f}"


SEP = "━━━━━━━━━━━━━━━"


def format_position_card(pos: Position, snap: dict, idx: int, total: int, now_ms: int) -> str:
    """One open position, same fields as the dashboard's open-position card, laid out for
    a phone glance: header / price+PnL / exit-plan / news blocks separated by blank lines,
    icons per line, key numbers bold. `snap` is PositionManager.snapshot()
    (mark/upnl/roe/eff_stop/stop_label); mark may be None."""
    side = "🟢 LONG" if pos.is_long else "🔴 SHORT"
    mark, upnl = snap.get("mark"), snap.get("upnl")
    unit = "shares" if pos.dex else _e(pos.symbol)

    head = [
        f"<b>{_tag(pos.dry_run)}</b> · 📊 position <b>{idx + 1}/{total}</b>",
        SEP,
        f"{side} <b>{_e(pos.symbol)} ×{pos.leverage}</b> · <code>{_e(pos.market)}</code>",
        f"🤖 confidence <b>{pos.confidence:.0%}</b>",
    ]

    money = [
        f"💵 Size <b>${pos.notional_usd:,.0f}</b> · {_num(pos.size)} {unit}",
        f"📍 Entry <b>{_num(pos.entry_px)}</b>  →  Mark <b>{_num(mark) if mark else '?'}</b>",
    ]
    if upnl is not None:
        pct = upnl / pos.notional_usd * 100 if pos.notional_usd else 0.0
        money.append(f"{'🟢' if upnl >= 0 else '🔴'} uPnL <b>{_usd(upnl)}</b> · "
                     f"{pct:+.2f}% · ROE {snap.get('roe') or 0:+.1f}%")
    else:
        money.append("⚪ uPnL <b>?</b> (no live price)")
    if pos.funding_usd:
        money.append(f"💸 Funding <b>{_usd(-pos.funding_usd)}</b> "
                     f"({'paid' if pos.funding_usd >= 0 else 'received'})")

    plan = []
    eff = snap.get("eff_stop") or pos.stop_loss
    if eff and pos.entry_px:
        dist = (eff / pos.entry_px - 1) * 100
        label = snap.get("stop_label") or "stop loss"
        plan.append(f"🛑 Stop <b>{_num(eff)}</b> · {dist:+.1f}%"
                    + (f" · <i>{_e(label)}</i>" if label != "stop loss" else ""))
    if pos.trail_pct > 0:
        plan.append(f"📐 Trail <b>{pos.trail_pct * 100:.1f}%</b>")
    elif pos.take_profit > 0:
        plan.append(f"🎯 TP <b>{_num(pos.take_profit)}</b>")
    plan.append(f"⏳ Exits in <b>{_dur(pos.time_exit_ms - now_ms)}</b> · "
                f"held {_dur(now_ms - pos.opened_ms)}")

    tail = []
    if pos.news_source or pos.news_title:
        tail.append(f"📰 <b>{_e(pos.news_source)}</b>: <i>{_e(pos.news_title)[:160]}</i>")
    if pos.rationale:
        tail.append(f"💡 <i>{_e(pos.rationale)[:300]}</i>")

    blocks = ["\n".join(head), "\n".join(money), "\n".join(plan)]
    if tail:
        blocks.append("\n".join(tail))
    stamp = time.strftime("%H:%M:%S", time.gmtime(now_ms / 1000))
    blocks.append(f"🕒 <i>updated {stamp} UTC</i>")
    return "\n\n".join(blocks)
