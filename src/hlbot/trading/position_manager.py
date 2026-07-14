"""Tracks open positions and enforces exits: stop-loss, take-profit, and a
time-based exit (news edge decays). Polls live mid prices on a fixed cadence."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import fields as dataclass_fields
from typing import Optional

from ..config import Config
from ..models import Market, Position, now_ms
from .executor import Executor
from .hl_client import HLClient

log = logging.getLogger("hlbot.positions")

MISSING_PRICE_ALERT = 10   # consecutive no-price polls before warning a position is unmonitored


class PositionManager:
    def __init__(self, config: Config, hl: HLClient, executor: Executor):
        self.config = config
        self.hl = hl
        self.executor = executor
        self._open: dict[str, Position] = {}   # position id -> Position
        self._no_price: dict[str, int] = {}    # position id -> consecutive polls with no price
        self._last_upnl: dict[str, float] = {} # position id -> last marked unrealized PnL (USD)
        self._funding_t: dict[str, float] = {} # position id -> last PAPER funding accrual (monotonic)
        self._closing: set[str] = set()        # ids with a close in flight (monitor or manual)
        self._absent: dict[str, int] = {}      # id -> consecutive reconciler polls gone from exchange
        # Grace before reconciling a LIVE position as exchange-closed, so a just-opened fill the
        # exchange hasn't surfaced yet isn't mistaken for one whose stop already fired.
        self._reconcile_grace_ms = 15_000
        self._stop = asyncio.Event()

    # ---- registry used by RiskEngine -------------------------------------
    def track(self, pos: Position) -> None:
        self._open[pos.id] = pos

    def open_count(self) -> int:
        return len(self._open)

    def total_exposure(self) -> float:
        return sum(p.notional_usd for p in self._open.values())

    def unrealized_pnl(self, dry_run: bool) -> float:
        """Sum of the last-marked unrealized PnL across open positions for the given mode.
        Feeds the daily-loss kill switch so open drawdown counts, not just realized losses.
        Uses the most recent monitor-tick mark (<= poll_interval old); a position not yet
        marked contributes 0 (conservative — it won't trip the halt prematurely)."""
        return sum(self._last_upnl.get(p.id, 0.0)
                   for p in self._open.values() if p.dry_run == dry_run)

    def has_open(self, market_name: str) -> bool:
        return any(p.market == market_name for p in self._open.values())

    def open_positions(self) -> list[Position]:
        return list(self._open.values())

    def position_for(self, market_name: str) -> "Position | None":
        return next((p for p in self._open.values() if p.market == market_name), None)

    def position_by_id(self, pos_id: str) -> "Position | None":
        return self._open.get(pos_id)

    async def snapshot(self, pos: Position) -> dict:
        """On-demand live mark/uPnL/effective-stop view of one position (Telegram browser,
        manual-close confirmations). Read-only: must NOT advance pos.peak_px — that's the
        monitor loop's job, and a display fetch moving the trail would tighten real exits."""
        market = Market(name=pos.market, symbol=pos.symbol, dex=pos.dex,
                        asset_class="", sz_decimals=0, max_leverage=0)
        try:
            price = await self.hl.mid(market)
        except Exception:  # noqa: BLE001 - display-only; the card shows unknowns as "?"
            price = None
        eff_stop, stop_label = self._eff_stop_label(pos)
        upnl = roe = None
        if price:
            upnl = ((price - pos.entry_px) if pos.is_long else (pos.entry_px - price)) * pos.size
            margin = (pos.notional_usd / pos.leverage) if pos.leverage else pos.notional_usd
            roe = (upnl / margin * 100.0) if margin else 0.0
        return {"mark": price, "upnl": upnl, "roe": roe,
                "eff_stop": eff_stop, "stop_label": stop_label}

    def _drop(self, pos_id: str) -> None:
        """Forget a position's per-id monitor state after it closed."""
        self._open.pop(pos_id, None)
        self._no_price.pop(pos_id, None)
        self._last_upnl.pop(pos_id, None)
        self._funding_t.pop(pos_id, None)
        self._absent.pop(pos_id, None)

    async def force_close(self, pos: Position, reason: str) -> bool:
        """Close a position immediately from outside the monitor loop (e.g. the
        contrary-news safeguard, UI/Telegram manual close). Returns True if it actually
        closed; on a failed LIVE close it stays tracked (the monitor will keep enforcing
        its stop and retry). The _closing guard serializes against the monitor loop and
        the reconciler so two closers can't both send orders / record the close."""
        if pos.id in self._closing or pos.id not in self._open:
            return False
        self._closing.add(pos.id)
        try:
            closed = await self.executor.close(pos, reason)
            if closed:
                self._drop(pos.id)
                return True
            return False
        finally:
            self._closing.discard(pos.id)

    async def restore(self, store, model_id: Optional[str] = None) -> None:
        """Reload open positions from the DB after a restart so exits resume. Filters to
        known fields (tolerates schema drift) and logs LOUDLY if a position can't be
        rebuilt — we must never silently drop a live position from management. `model_id`
        restricts to one arena lane's positions (None = all, the production behavior)."""
        valid = {f.name for f in dataclass_fields(Position)}
        for d in await store.open_positions(model_id=model_id):
            pid = d.get("id")
            try:
                unknown = set(d) - valid
                if unknown:
                    log.warning("Position %s: ignoring unknown stored fields %s", pid, unknown)
                self._open[pid] = Position(**{k: v for k, v in d.items() if k in valid})
            except Exception as exc:  # noqa: BLE001
                log.error("FAILED to restore position %s: %s — it will NOT be managed! data=%s",
                          pid, exc, d)
        if self._open:
            log.info("Restored %d open positions from store", len(self._open))

    # ---- monitor loop -----------------------------------------------------
    async def run(self) -> None:
        interval = self.config.app.poll_interval_seconds
        while not self._stop.is_set():
            try:
                await self._check_all()
            except Exception:  # noqa: BLE001
                log.exception("Position monitor error")
            await asyncio.sleep(interval)

    @staticmethod
    def _exchange_has(open_coins: set, pos: Position) -> bool:
        """Is this tracked position still open on the exchange? Match defensively across the
        possible coin spellings (full name 'xyz:MRVL', bare 'MRVL', dex-prefixed)."""
        cand = {pos.market, pos.symbol, pos.market.split(":")[-1]}
        return bool(open_coins & cand) or any(c.split(":")[-1] == pos.symbol for c in open_coins)

    async def _position_states(self) -> dict[str, "dict | None"]:
        """One user_state fetch per dex that has LIVE positions: {dex: {coin: {szi, funding}}},
        with None marking a failed fetch (callers must fail SAFE on it). Feeds both the
        exchange-close reconciler and the live funding-paid tracking."""
        out: dict[str, dict | None] = {}
        for dex in {p.dex for p in self._open.values() if not p.dry_run}:
            try:
                out[dex] = await self.hl.position_state(dex)
            except Exception:  # noqa: BLE001 - can't confirm anything for this dex
                out[dex] = None
        return out

    async def _reconcile_exchange_closes(self, states: dict) -> None:
        """Detect LIVE positions that the exchange closed out from under us — almost always our
        own resting reduce-only stop firing (also a manual close or liquidation). Without this,
        such a position lingers in tracking forever: it blocks re-entry, miscounts exposure, and
        its eventual time-exit loops trying to market_close a position that no longer exists.

        Dry-run positions never exist on the exchange, so they're skipped (otherwise every paper
        position would be 'reconciled' away). Fails SAFE: any fetch error -> keep managing, since
        we can't confirm the position is actually gone."""
        now = now_ms()
        for pos in [p for p in self._open.values() if not p.dry_run]:
            state = states.get(pos.dex)
            if state is None:                            # this dex's fetch failed -> skip
                continue
            if pos.id in self._closing:                  # a close is mid-flight; it owns the record
                continue
            open_coins = {c for c, st in state.items() if abs(st.get("szi", 0)) > 0}
            if now - pos.opened_ms < self._reconcile_grace_ms:
                continue                                 # just opened; let the exchange catch up
            if self._exchange_has(open_coins, pos):
                self._absent.pop(pos.id, None)
                continue                                 # still open -> nothing to do
            # Debounce: a single well-formed-but-empty user_state (API glitch) must not read
            # as "position gone" — that would record a close for real, live exposure. Require
            # two consecutive absent polls (~3s extra) before trusting it.
            misses = self._absent.get(pos.id, 0) + 1
            self._absent[pos.id] = misses
            if misses < 2:
                continue
            # Gone from the exchange. The resting stop fired at ~stop_loss (best estimate); a
            # manual/liq close lands near there too. Record it closed without sending an order.
            log.warning("Position %s (%s) gone from exchange — recording exchange-stop close",
                        pos.id, pos.market)
            exit_px = pos.stop_loss or pos.entry_px
            try:
                await self.executor.mark_closed(pos, "stop loss (exchange)", exit_px)
            except Exception:  # noqa: BLE001
                log.exception("Failed to record exchange-stop close for %s", pos.id)
                continue
            self._drop(pos.id)

    def _update_live_funding(self, states: dict) -> None:
        """Copy the exchange's cumulative funding-since-open onto each LIVE position (matched
        across the possible coin spellings). Persisted opportunistically with the next upsert."""
        for pos in self._open.values():
            if pos.dry_run:
                continue
            state = states.get(pos.dex)
            if not state:
                continue
            for coin, st in state.items():
                if coin in (pos.market, pos.symbol) or coin.split(":")[-1] == pos.symbol:
                    pos.funding_usd = st.get("funding", pos.funding_usd)
                    break

    async def _accrue_paper_funding(self, pos: Position, market: Market) -> None:
        """PAPER positions never pay real funding, so accrue an estimate from the current
        hourly rate: rate * notional * hours, sign-flipped for shorts (a short RECEIVES when
        the rate is positive). Keeps the paper PnL display honest about carry cost."""
        now = time.monotonic()
        last = self._funding_t.get(pos.id)
        self._funding_t[pos.id] = now
        if last is None:
            return
        try:
            rate = await self.hl.funding_rate(market)
        except Exception:  # noqa: BLE001 - estimate only
            return
        if not rate:
            return
        sign = 1.0 if pos.is_long else -1.0
        pos.funding_usd += rate * pos.notional_usd * ((now - last) / 3600.0) * sign

    async def _check_all(self) -> None:
        states = await self._position_states()
        await self._reconcile_exchange_closes(states)   # positions an exchange stop already closed
        self._update_live_funding(states)
        ticks = []
        for pos in list(self._open.values()):
            if pos.id in self._closing:   # manual/contrary close in flight — don't double-send
                continue
            market = Market(name=pos.market, symbol=pos.symbol, dex=pos.dex,
                            asset_class="", sz_decimals=0, max_leverage=0)
            if pos.dry_run:
                await self._accrue_paper_funding(pos, market)
            price = await self.hl.mid(market)
            prev_peak = pos.peak_px
            reason = self._exit_reason(pos, price)   # also advances pos.peak_px
            if reason:
                self._closing.add(pos.id)
                try:
                    closed = await self.executor.close(pos, reason, exit_px=price)
                    if closed:                      # None => live close didn't fill; keep & retry
                        self._drop(pos.id)
                finally:
                    self._closing.discard(pos.id)
                continue
            if price is None:
                # No price -> SL/TP/trail cannot be enforced; only time-exit can fire. Surface it.
                n = self._no_price.get(pos.id, 0) + 1
                self._no_price[pos.id] = n
                if n == MISSING_PRICE_ALERT:
                    log.warning("No live price for %s for %d polls — stop/TP/trail NOT enforced",
                                pos.market, n)
                    await self.executor.bus.publish("trade.error", {
                        "market": pos.market, "news_id": pos.news_id,
                        "error": f"no live price for {n} polls — stop/TP/trail not enforced"})
                continue
            self._no_price.pop(pos.id, None)
            if pos.peak_px != prev_peak:   # new high-water mark -> persist so a restart keeps the trail
                try:
                    await self.executor.store.upsert_position(pos)
                except Exception:  # noqa: BLE001
                    log.debug("Could not persist peak for %s", pos.id)
            upnl = ((price - pos.entry_px) if pos.is_long else (pos.entry_px - price)) * pos.size
            self._last_upnl[pos.id] = upnl   # feeds the daily-loss kill switch (open drawdown)
            margin = (pos.notional_usd / pos.leverage) if pos.leverage else pos.notional_usd
            roe = (upnl / margin * 100.0) if margin else 0.0
            ticks.append({"id": pos.id, "mark": price, "upnl": upnl, "roe": roe,
                          "eff_stop": self._eff_stop(pos), "peak_px": pos.peak_px,
                          "funding": pos.funding_usd})
        if ticks:
            await self.executor.bus.publish("positions.tick", {"ticks": ticks})

    def _eff_stop_label(self, pos: Position) -> tuple[float, str]:
        """Binding stop floor (initial / trailing / breakeven) and its exit-reason label,
        from the current peak. Mirrors the backtest's walk_candles.eff_stop exactly.
        Breakeven reads config at poll time (like trailing, it's bot-side only — the
        exchange backstop stays at the initial stop)."""
        r = self.config.app.risk
        arm = getattr(r, "breakeven_arm_pct", 0.0)
        off = getattr(r, "breakeven_offset_pct", 0.0)
        eff, label = pos.stop_loss, "stop loss"
        peak = pos.peak_px or pos.entry_px
        if pos.is_long:
            if pos.trail_pct > 0 and peak > pos.entry_px:   # trail only once in profit
                t = peak * (1 - pos.trail_pct)
                if t > eff:
                    eff, label = t, "trailing stop"
            if arm > 0 and peak >= pos.entry_px * (1 + arm):
                b = pos.entry_px * (1 + off)
                if b > eff:
                    eff, label = b, "breakeven stop"
        else:
            if pos.trail_pct > 0 and 0 < peak < pos.entry_px:
                t = peak * (1 + pos.trail_pct)
                if t < eff:
                    eff, label = t, "trailing stop"
            if arm > 0 and peak <= pos.entry_px * (1 - arm):
                b = pos.entry_px * (1 - off)
                if b < eff:
                    eff, label = b, "breakeven stop"
        return eff, label

    def _eff_stop(self, pos: Position) -> float:
        """Current effective stop incl. trailing/breakeven (read-only, for UI ticks)."""
        return self._eff_stop_label(pos)[0]

    def _exit_reason(self, pos: Position, price: float | None) -> str | None:
        if now_ms() >= pos.time_exit_ms:
            return "time exit"
        if not price:
            return None
        if pos.is_long:
            pos.peak_px = max(pos.peak_px or pos.entry_px, price)
            eff_stop, label = self._eff_stop_label(pos)
            if price <= eff_stop:
                return label
            if pos.take_profit > 0 and price >= pos.take_profit:
                return "take profit"
        else:
            pos.peak_px = min(pos.peak_px or pos.entry_px, price)
            eff_stop, label = self._eff_stop_label(pos)
            if price >= eff_stop:
                return label
            if pos.take_profit > 0 and price <= pos.take_profit:
                return "take profit"
        return None

    def stop(self) -> None:
        self._stop.set()
