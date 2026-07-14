"""Order execution. In dry-run it simulates a fill at the decision price and records
a paper position; live it sets leverage and sends an aggressive market (IOC) order
via the SDK. One runtime flag flips between the two."""
from __future__ import annotations

import logging
import uuid

from ..bus import EventBus
from ..config import Config
from ..models import Analysis, Decision, Market, NewsItem, Position, now_ms
from .hl_client import HLClient

log = logging.getLogger("hlbot.executor")

# Taker fee per side modeled in dry-run/backtest PnL. The bot is a taker on BOTH legs
# (IOC open, market close); Hyperliquid's base-tier perp taker fee is 0.045%. Volume
# tiers / HYPE staking lower it; HIP-3 builder dexes (xyz) may add a builder fee on top —
# if real fills show higher all-in fees, raise this so paper PnL stays honest.
FEE_RATE_PER_SIDE = 0.00045


class Executor:
    def __init__(self, config: Config, hl: HLClient, store, bus: EventBus):
        self.config = config
        self.hl = hl
        self.store = store
        self.bus = bus
        self._leverage_set: dict[str, int] = {}   # market -> last leverage set (skip redundant RTT)
        # Arena: per-lane paper/live override. None -> use the global runtime flag (production).
        # True forces paper, False forces live — lets one lane go live while others stay paper.
        self.dry_run_override: bool | None = None

    async def open(self, decision: Decision, item: NewsItem, analysis: Analysis) -> Position | None:
        market = decision.market
        assert market is not None
        dry = (self.dry_run_override if self.dry_run_override is not None
               else self.config.runtime.dry_run)
        is_buy = decision.side == "long"
        opened = now_ms()
        pos = Position(
            id=uuid.uuid4().hex[:12],
            news_id=item.id,
            market=market.name,
            symbol=market.symbol,
            dex=market.dex,
            side=decision.side,
            size=decision.size,
            entry_px=decision.entry_px,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            leverage=decision.leverage,
            notional_usd=decision.notional_usd,
            opened_ms=opened,
            time_exit_ms=opened + decision.time_exit_seconds * 1000,
            dry_run=dry,
            trail_pct=decision.trail_pct,
            peak_px=decision.entry_px,
            link=item.link,
            confidence=analysis.confidence,
            rationale=decision.reason,
            news_title=item.title,
            news_source=item.source or "Twitter",
            model_id=analysis.model,   # arena lane partition (== the lane's model)
        )

        if dry:
            slip = self.config.app.risk.dry_run_slippage_pct
            pos.entry_px = decision.entry_px * ((1 + slip) if is_buy else (1 - slip))
            pos.peak_px = pos.entry_px
            self._rescale_exits(pos, decision)
            log.info("[DRY-RUN] %s %s size=%s @ %.4f (mid %.4f, slip %.3f%%)",
                     pos.side.upper(), pos.market, pos.size, pos.entry_px,
                     decision.entry_px, slip * 100)
        else:
            if self._leverage_set.get(market.name) != decision.leverage:
                try:
                    await self.hl.set_leverage(market, decision.leverage)
                    self._leverage_set[market.name] = decision.leverage
                except Exception:  # noqa: BLE001 - leverage may already be set / unsupported
                    log.warning("Could not set leverage for %s", market.name)
            try:
                result = await self.hl.market_open(
                    market, is_buy, decision.size, self.config.app.risk.slippage_pct)
                avg_px, total_sz, err = HLClient.parse_fill(result)
            except Exception as exc:  # noqa: BLE001
                # The order may have reached the exchange even though the call errored (e.g.
                # timeout while reading the response). Check before declaring it dead — a
                # filled order that goes untracked has NO exits at all, bot-side or exchange.
                log.exception("market_open errored for %s — checking exchange for a fill",
                              market.name)
                adopted_sz = await self._find_untracked_fill(market, is_buy)
                if adopted_sz is None:
                    avg_px, total_sz, err = None, None, f"order errored: {exc}"
                else:
                    # Adopt it: entry estimated at the decision mid (the true avg fill px was
                    # lost with the response); exits re-anchor to that estimate.
                    avg_px, total_sz, err = decision.entry_px, adopted_sz, None
                    log.warning("Adopted exchange position for %s (size %s) after order error "
                                "— entry estimated at decision mid %.4f",
                                market.name, adopted_sz, decision.entry_px)
                    await self.bus.publish("trade.error", {
                        "market": market.name, "news_id": item.id,
                        "error": "order response lost — adopted the exchange position "
                                 "(entry price estimated)"})
            if err or not avg_px:
                log.error("Order failed for %s: %s", market.name, err)
                await self.bus.publish("trade.error", {"market": market.name, "error": err,
                                                       "news_id": item.id})
                return None
            pos.entry_px = avg_px
            pos.size = total_sz or decision.size
            pos.notional_usd = pos.entry_px * pos.size
            pos.peak_px = avg_px
            self._rescale_exits(pos, decision)
            log.info("LIVE FILL %s %s size=%s @ %.4f", pos.side.upper(), pos.market,
                     pos.size, pos.entry_px)
            await self._place_exchange_stop(pos, market, is_buy)

        try:
            await self.store.upsert_position(pos)
        except Exception:  # noqa: BLE001 - a DB hiccup must not orphan a live fill
            log.exception("Failed to persist new position %s — still tracked in memory", pos.id)
        await self.bus.publish("trade.open", pos)
        return pos

    async def _find_untracked_fill(self, market: Market, is_buy: bool) -> float | None:
        """After a market_open call errored, ask the exchange whether the order actually
        filled. Returns the position size to adopt, or None if the exchange is flat on this
        market (the order really failed). The risk gate blocks entries on markets with an
        open position, so a matching-side position here is ours."""
        try:
            state = await self.hl.position_state(market.dex)
        except Exception:  # noqa: BLE001 - can't confirm; treat as failed (alert already sent)
            log.exception("Could not verify exchange state for %s after order error",
                          market.name)
            return None
        for coin, st in state.items():
            if coin in (market.name, market.symbol) or coin.split(":")[-1] == market.symbol:
                szi = st.get("szi", 0.0)
                if szi and (szi > 0) == is_buy:
                    return abs(szi)
        return None

    @staticmethod
    def _rescale_exits(pos: Position, decision: Decision) -> None:
        """Re-anchor SL/TP to the ACTUAL fill price. The decision computed them from the
        pre-trade mid; a slipped fill would otherwise silently tighten (or widen) the stop
        distance — and the exchange backstop stop would rest at the wrong level. Scaling by
        fill/decision preserves the configured percentage distances exactly."""
        if decision.entry_px <= 0 or pos.entry_px == decision.entry_px:
            return
        ratio = pos.entry_px / decision.entry_px
        if pos.stop_loss > 0:
            pos.stop_loss *= ratio
        if pos.take_profit > 0:
            pos.take_profit *= ratio

    async def _place_exchange_stop(self, pos: Position, market: Market, is_buy: bool) -> None:
        """Rest a reduce-only stop on the exchange so the position is protected even if the bot
        goes down (the bot-side monitor only enforces stops while it's running). Best-effort:
        a failure does NOT abort the entry — the bot-side stop still applies while it runs.
        Records the oid on the position so a normal close (or a restart) can cancel it."""
        if not (pos.stop_loss and pos.stop_loss > 0):
            return
        try:
            result = await self.hl.place_stop(
                market, not is_buy, pos.size, pos.stop_loss, self.config.app.risk.slippage_pct)
            oid = HLClient.parse_resting_oid(result)
            if oid:
                pos.stop_order_id = oid
                log.info("Placed exchange stop for %s @ %.4f (oid %s)",
                         pos.market, pos.stop_loss, oid)
            else:
                log.warning("Exchange stop for %s did not rest (%s) — bot-side stop only",
                            pos.market, result)
                await self.bus.publish("trade.error", {
                    "market": pos.market, "news_id": pos.news_id,
                    "error": "exchange stop not placed — bot-side stop only (active only while "
                             "the bot is running)"})
        except Exception as exc:  # noqa: BLE001 - never let stop placement abort a filled entry
            log.exception("Failed to place exchange stop for %s", pos.market)
            await self.bus.publish("trade.error", {
                "market": pos.market, "news_id": pos.news_id,
                "error": f"exchange stop failed ({exc}) — bot-side stop only"})

    async def close(self, pos: Position, reason: str, exit_px: float | None = None) -> Position | None:
        """Close a position. Returns the closed Position on success, or None if a LIVE
        close did NOT fill (the position is still open on the exchange) — the caller must
        then keep managing/retrying it rather than dropping it from tracking. Dry-run
        always succeeds.

        Close in the mode the position was OPENED in (pos.dry_run), NOT the current runtime
        mode: a paper position carried into a live session must close on paper (no real order
        for a position that never existed on the exchange), and a real live position must keep
        closing live even if the runtime is toggled to dry-run mid-flight (or its real exposure
        would be left unmanaged)."""
        market = Market(name=pos.market, symbol=pos.symbol, dex=pos.dex,
                        asset_class="", sz_decimals=0, max_leverage=0)
        if pos.dry_run:
            if exit_px is None:
                exit_px = await self.hl.mid(market) or pos.entry_px
            pos.exit_decision_px = exit_px   # pre-slip trigger mid (slippage attribution)
            slip = self.config.app.risk.dry_run_slippage_pct
            exit_px *= (1 - slip) if pos.is_long else (1 + slip)   # adverse fill on close
        else:
            pos.exit_decision_px = exit_px or 0.0   # trigger mid; the fill overwrites exit_px
            try:
                result = await self.hl.market_close(market, self.config.app.risk.slippage_pct)
                avg_px, fill_sz, err = HLClient.parse_fill(result)
                if not avg_px:
                    # The close did NOT fill -> the position is STILL OPEN. Never mark it
                    # closed or PnL/exposure/cooldown state diverges from the exchange and
                    # the position is left with no stop enforcement. Alert and let the
                    # monitor retry on the next poll.
                    log.error("Close did NOT fill for %s: %s — STILL OPEN, will retry",
                              pos.market, err)
                    await self.bus.publish("trade.error", {
                        "market": pos.market, "news_id": pos.news_id,
                        "error": f"close did not fill ({err}) — position still open, retrying"})
                    return None
                if fill_sz and fill_sz < pos.size * (1 - 1e-6):
                    # Partial IOC fill: the exchange is NOT flat. Realize the filled slice,
                    # shrink the tracked position, and keep managing the remainder (the
                    # monitor retries the close next poll). The reduce-only exchange stop
                    # stays — oversized reduce-only on the remainder is harmless.
                    await self._record_partial_close(pos, fill_sz, avg_px, reason)
                    return None
                exit_px = avg_px
            except Exception as exc:  # noqa: BLE001
                log.exception("Close errored for %s — may still be open, will retry", pos.market)
                await self.bus.publish("trade.error", {
                    "market": pos.market, "news_id": pos.news_id,
                    "error": f"close errored ({exc}) — position may still be open, retrying"})
                return None
            # Filled: cancel the resting exchange stop so it can't fire on a now-flat position.
            await self._cancel_exchange_stop(pos, market)

        return await self._record_close(pos, reason, exit_px)

    async def _cancel_exchange_stop(self, pos: Position, market: Market) -> None:
        """Cancel the resting reduce-only stop after we've closed the position ourselves, so a
        stale trigger can't linger. Best-effort — a reduce-only order on a flat position is
        harmless, and HL auto-cancels it when the position closes."""
        if not pos.stop_order_id:
            return
        try:
            await self.hl.cancel_order(market, pos.stop_order_id)
        except Exception:  # noqa: BLE001 - may already be gone (triggered/auto-cancelled)
            log.debug("Could not cancel exchange stop %s for %s", pos.stop_order_id, pos.market)
        pos.stop_order_id = 0

    async def _record_partial_close(self, pos: Position, filled_sz: float, exit_px: float,
                                    reason: str) -> None:
        """A close order filled only part of the position. Realize the slice into
        partial_pnl_usd (folded into pnl_usd at the final close), shrink the live size, and
        alert — the position stays tracked and managed."""
        direction = 1 if pos.is_long else -1
        gross = (exit_px - pos.entry_px) * filled_sz * direction
        fees = (pos.entry_px + exit_px) * filled_sz * FEE_RATE_PER_SIDE
        pos.partial_pnl_usd += gross - fees
        pos.size -= filled_sz
        pos.notional_usd = pos.entry_px * pos.size
        log.warning("PARTIAL close for %s: %s closed @ %.4f, %s still open (%s)",
                    pos.market, filled_sz, exit_px, pos.size, reason)
        try:
            await self.store.upsert_position(pos)
        except Exception:  # noqa: BLE001
            log.exception("Could not persist partial close for %s", pos.id)
        await self.bus.publish("trade.error", {
            "market": pos.market, "news_id": pos.news_id,
            "error": f"close filled partially ({filled_sz} of {filled_sz + pos.size}) — "
                     "remainder still open, retrying"})

    async def _record_close(self, pos: Position, reason: str, exit_px: float) -> Position:
        """Finalize a closed position: compute PnL (incl. round-trip fees), persist, alert.
        Idempotent: concurrent closers (monitor loop, manual close, reconciler) can race to
        record the same position — the first result wins, a second call is a no-op."""
        if pos.status == "closed":
            return pos
        direction = 1 if pos.is_long else -1
        gross = (exit_px - pos.entry_px) * pos.size * direction
        fees = (pos.entry_px + exit_px) * pos.size * FEE_RATE_PER_SIDE
        pos.exit_px = exit_px
        pos.pnl_usd = gross - fees + pos.partial_pnl_usd
        pos.exit_reason = reason
        pos.status = "closed"
        pos.closed_ms = now_ms()

        await self.store.upsert_position(pos)
        await self.bus.publish("trade.close", pos)
        log.info("CLOSED %s %s @ %.4f pnl=%.2f (%s)", pos.side.upper(), pos.market,
                 exit_px, pos.pnl_usd, reason)
        return pos

    async def mark_closed(self, pos: Position, reason: str, exit_px: float) -> Position:
        """Record a position as closed WITHOUT sending a close order — it was already closed on
        the exchange (a resting stop fired, or a manual/liquidation close). The position
        manager calls this during reconciliation so PnL/exposure/slot state rejoins reality
        instead of looping forever trying to close a position that no longer exists."""
        return await self._record_close(pos, reason, exit_px)
