"""Risk & decision engine. Turns an Analysis + resolved Market into a Decision,
applying every guardrail that keeps the bot net-positive: confidence gate, stale
filter, confidence-scaled sizing, leverage/exposure/concurrency caps, per-ticker
cooldown, and a daily-loss kill switch.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from ..config import Config
from ..models import Analysis, Decision, Market, NewsItem
from .hl_client import HLClient

log = logging.getLogger("hlbot.risk")

MIN_ORDER_USD = 10.0  # Hyperliquid minimum order value

_LISTING_RE = re.compile(
    r"\b(listing|will\s+list|to\s+list|now\s+list|lists|delist|delisting|"
    r"spot\s+trading|launchpool|launchpad|pre-?market)\b", re.IGNORECASE)
_EXCHANGE_RE = re.compile(
    r"\b(binance|upbit|bithumb|coinbase|okx|bybit|kraken|kucoin|gate\.io|bitget)\b",
    re.IGNORECASE)


def is_listing_news(text: str) -> bool:
    """Exchange listing/delisting news: real, but heavily front-run/competitive."""
    return bool(_LISTING_RE.search(text) and _EXCHANGE_RE.search(text))


_DUP_TOKEN_RE = re.compile(r"[a-z0-9]+")


def dup_fingerprint(text: str) -> frozenset:
    """Token set for duplicate-suppression similarity — a lexical proxy for 'same story'.
    Drops 1-char tokens (punctuation-split noise). Two distinct catalysts that merely resolve
    to the same ticker+direction (e.g. a supplier's earnings vs the company's own) share few
    tokens; a reworded repost of the SAME wire shares most."""
    return frozenset(t for t in _DUP_TOKEN_RE.findall((text or "").lower()) if len(t) > 1)


def dup_similarity(a: frozenset, b: frozenset) -> float:
    """Jaccard overlap of two fingerprints, 0..1. Two EMPTY fingerprints (token-less / degenerate
    text) count as identical (1.0) so we fall back to suppress-on-(ticker,direction) rather than
    letting token-less news slip the dedup; one empty + one not counts as different (0.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def liquidity_penalty(market: Optional[Market], r) -> float:
    """Confidence haircut for thin crypto alts (by 24h notional volume)."""
    if not market or market.dex != "":   # only crypto perps; equities handled by trade.xyz
        return 0.0
    vol = market.day_volume_usd or 0.0
    if vol <= 0 or vol >= r.liquidity_high_usd:
        return 0.0
    return r.liquidity_penalty_med if vol >= r.liquidity_med_usd else r.liquidity_penalty_low


def premarket_factor(market: Optional[Market], r) -> float:
    """Size multiplier: pre-IPO/premarket synthetic perps are traded smaller."""
    if market and market.symbol.upper() in {s.upper() for s in r.premarket_symbols}:
        return r.premarket_size_factor
    return 1.0


def new_listing_penalty(market: Optional[Market], r) -> float:
    """Confidence haircut for markets Hyperliquid listed very recently (thin liquidity).
    first_seen_ms == 0 means baseline/old (no penalty)."""
    if not market or not getattr(market, "first_seen_ms", 0.0):
        return 0.0
    age_h = (time.time() * 1000 - market.first_seen_ms) / 3_600_000
    return r.new_listing_penalty if 0 <= age_h < r.new_listing_age_hours else 0.0


def adjust_confidence(base: float, market: Optional[Market], text: str, r,
                      mentioned: bool = True) -> tuple[float, list[str]]:
    """Apply deterministic post-analysis haircuts; returns (effective_conf, notes).
    `mentioned` False (the resolved ticker isn't directly named in the news) is a 2nd-order
    inference -> haircut by indirect_mention_penalty."""
    conf, notes = base, []
    lp = liquidity_penalty(market, r)
    if lp:
        conf -= lp
        notes.append(f"-{lp:.2f} illiquid")
    nlp = new_listing_penalty(market, r)
    if nlp:
        conf -= nlp
        notes.append(f"-{nlp:.2f} new-listing(<{r.new_listing_age_hours:g}h)")
    if is_listing_news(text):
        conf -= r.listing_penalty
        notes.append(f"-{r.listing_penalty:.2f} listing")
    imp = getattr(r, "indirect_mention_penalty", 0.0)
    if not mentioned and imp:
        conf -= imp
        notes.append(f"-{imp:.2f} indirect")
    sp = getattr(r, "symbol_penalties", None) or {}
    if market and sp:
        sym = market.symbol.upper()
        pen = next((v for k, v in sp.items() if k.upper() == sym), 0.0)
        if pen:
            conf -= pen
            notes.append(f"-{pen:.2f} {sym}")
    return max(0.0, conf), notes


def ref_price_from_candles(candles: list[dict], news_ms: int) -> Optional[float]:
    """Pre-news reference price from 1m candles: close of the last bar ENDING at/before the
    news; falls back to the open of the bar containing the news (that open predates the
    headline). None when no bar predates the news (fail open)."""
    ref = None
    for c in candles or []:
        try:
            if int(c["T"]) <= news_ms:
                ref = float(c["c"])
            elif int(c["t"]) <= news_ms and ref is None:
                ref = float(c["o"])
        except (KeyError, TypeError, ValueError):
            continue
    return ref


def signed_move_pct(ref_px: float, cur_px: float, direction: str) -> float:
    """Fractional move from the pre-news reference to the entry price, positive when the
    market already moved IN the trade's direction (the repricing happened without us)."""
    if not ref_px or ref_px <= 0 or not cur_px or cur_px <= 0:
        return 0.0
    raw = (cur_px - ref_px) / ref_px
    return raw if direction == "long" else -raw


def apply_move_guard(eff_conf: float, move_pct: float, time_sensitivity: str,
                     r) -> tuple[float, list[str], Optional[str]]:
    """Already-moved entry guard: we knowingly enter seconds-to-minutes behind HFT, so if
    the repricing ALREADY happened the edge is gone — haircut at pre_move_haircut_pct,
    reject at pre_move_reject_pct. 'days' (regime-change) catalysts are haircut-only: a
    one-stop-sized pop doesn't exhaust a multi-session thesis, and those are the trades
    this strategy exists for. A move AGAINST the signal is a better entry — no adjustment.
    Returns (new_conf, notes, reject_reason); both thresholds 0 = disabled."""
    notes: list[str] = []
    haircut_at = getattr(r, "pre_move_haircut_pct", 0.0)
    reject_at = getattr(r, "pre_move_reject_pct", 0.0)
    if move_pct <= 0:
        return eff_conf, notes, None
    if reject_at > 0 and move_pct >= reject_at and time_sensitivity != "days":
        return eff_conf, notes, (f"already moved {move_pct:+.1%} in-direction since the news "
                                 f"(>= {reject_at:.1%}); edge likely captured")
    if haircut_at > 0 and move_pct >= haircut_at:
        pen = getattr(r, "pre_move_penalty", 0.0)
        if pen:
            eff_conf = max(0.0, eff_conf - pen)
            notes.append(f"-{pen:.2f} already-moved({move_pct:+.1%})")
    return eff_conf, notes, None


def scale_notional(confidence: float, r) -> float:
    """Confidence-tiered notional: pick the highest tier whose confidence <= the signal."""
    notional = r.base_notional_usd
    for thr, ntl in sorted(r.size_tiers):
        if confidence >= thr:
            notional = ntl
    return min(notional, r.max_notional_usd)


def exit_params(time_sensitivity: str, r) -> tuple[int, float, float, float]:
    """Adaptive exits from the analyzer's time_sensitivity.
    Returns (horizon_seconds, stop_pct, take_profit_pct, trail_pct).
    trail_pct > 0 => trailing stop active and no fixed take-profit (let it run)."""
    ts = time_sensitivity if time_sensitivity in r.exit_horizons else "immediate"
    horizon = r.exit_horizons.get(ts, r.time_exit_seconds)
    stop_pct = r.stop_loss_by_sensitivity.get(ts, r.stop_loss_pct)
    trail_pct = r.trail_pct_by_sensitivity.get(ts, 0.0)
    tp_pct = 0.0 if trail_pct > 0 else r.take_profit_pct
    return horizon, stop_pct, tp_pct, trail_pct


def compute_sl_tp(side: str, price: float, stop_pct: float, tp_pct: float) -> tuple[float, float]:
    """Return (stop_loss, take_profit) prices. take_profit is 0.0 when tp_pct is 0."""
    if side == "long":
        return price * (1 - stop_pct), (price * (1 + tp_pct) if tp_pct > 0 else 0.0)
    return price * (1 + stop_pct), (price * (1 - tp_pct) if tp_pct > 0 else 0.0)


class RiskEngine:
    def __init__(self, config: Config, hl: HLClient, store, position_manager, universe=None,
                 confirmer=None):
        self.config = config
        self.r = config.app.risk
        self.hl = hl
        self.store = store
        self.pm = position_manager
        self.universe = universe   # for the direct-mention check (None -> check skipped)
        # Optional skeptic entry-confirmation callback (Analyzer.confirm, injected by main.py
        # so trading/ never imports analysis/): async (item, analysis, market_context,
        # pre_move_pct) -> {agree_direction, confidence, risk} | None (None = fail OPEN).
        self.confirmer = confirmer
        self._last_entry_ms: dict[str, int] = {}
        # (symbol, direction) -> (last-seen ms, token fingerprint) of the most recent such signal
        # (incl. faded). The fingerprint lets us suppress only a TRUE repeat of the SAME story.
        self._recent_signal: dict[tuple, tuple] = {}
        self._blacklist = {b.upper() for b in config.app.filters.market_blacklist}
        # Arena per-lane overrides (None/None -> global/all, the production behavior). The lane
        # sets dry_run_override (paper/live for its daily-loss accounting) and model_id, which
        # scopes BOTH the daily-loss accounting (only THIS lane's realized PnL) and the daily-loss
        # HALT (rt.halt_daily(model_id=...) pauses only this wallet — other lanes keep trading).
        # The manual kill switch (rt.halt()) remains global by design.
        self.dry_run_override: bool | None = None
        self.model_id: str | None = None

    def note_entry(self, symbol: str) -> None:
        self._last_entry_ms[symbol.upper()] = int(time.time() * 1000)

    async def restore(self) -> None:
        """Rebuild per-ticker cooldowns from the DB so a restart can't immediately
        re-enter a ticker traded within the cooldown window. Arena lanes (model_id set)
        restore only their OWN entries — another lane's trade must not cool this lane down."""
        since = int(time.time() * 1000 - self.r.per_ticker_cooldown_seconds * 1000)
        try:
            rows = await self.store.last_entries(since, model_id=self.model_id)
            self._last_entry_ms = {s.upper(): ms for s, ms in rows.items()}
            if self._last_entry_ms:
                log.info("Restored cooldowns for %d tickers", len(self._last_entry_ms))
        except Exception:  # noqa: BLE001
            log.warning("Could not restore cooldowns")
        # Rebuild the duplicate-signal window (validated: closes the rebroadcast-after-
        # stop-out hole) — a restart inside the window must not re-open it. Rebuilt from
        # directional analyses (keyed by analyzer ticker ≈ market symbol); chronological,
        # so the newest signal per (ticker, direction) wins, matching live recording.
        if self.r.duplicate_window_seconds:
            try:
                since_dup = int(time.time() * 1000 - self.r.duplicate_window_seconds * 1000)
                for row in await self.store.recent_directional_analyses(
                        since_dup, model=self.model_id):
                    fp = dup_fingerprint(f"{row['title']}\n{row['body']}".strip())
                    self._recent_signal[(row["ticker"].upper(), row["direction"])] = \
                        (int(row["ts"]), fp)
                if self._recent_signal:
                    log.info("Restored %d duplicate-window signals", len(self._recent_signal))
            except Exception:  # noqa: BLE001
                log.warning("Could not restore the duplicate-signal window")

    def _reject(self, item: NewsItem, reason: str, analysis: Analysis,
                market: Optional[Market]) -> Decision:
        return Decision(news_id=item.id, action="reject", reason=reason,
                        market=market, confidence=analysis.confidence)

    async def _pre_news_ref(self, market: Market, news_ms: int) -> Optional[float]:
        """Pre-news reference price (candles ending just before the news). Falls back to
        coarser intervals — HIP-3 equities often have sparse 1m bars, which otherwise made
        the guard fail open on exactly the markets that dominate our flow. Fails OPEN
        (None) on any fetch error — a transient API blip must not block a good signal."""
        for iv, iv_s in (("1m", 60), ("5m", 300), ("15m", 900)):
            lb_ms = (max(iv_s, self.r.pre_move_lookback_seconds) + iv_s) * 1000
            try:
                candles = await self.hl.candles(market.name, iv,
                                                int(news_ms - lb_ms), int(news_ms + 60_000))
            except Exception:  # noqa: BLE001
                log.warning("Pre-move candle fetch failed for %s; skipping move guard",
                            market.name)
                return None
            ref = ref_price_from_candles(candles, news_ms)
            if ref:
                return ref
        return None

    async def liquidity_guard(self, market: Market) -> Optional[str]:
        """Reject entries into a too-wide / too-thin live order book (thin off-hours markets,
        especially trade.xyz equities). Returns a reject reason, or None to allow. LIVE/dry-run
        only — needs the live book, so it is NOT applied in the backtest. Fails OPEN (returns
        None) on any book-fetch error or an empty/one-sided book, so a transient API blip or an
        unsupported market can't block a good signal."""
        r = self.r
        if market.dex not in r.spread_guard_dexes:
            return None
        if r.max_spread_pct <= 0 and r.min_top_depth_usd <= 0:
            return None
        fetch = getattr(self.hl, "l2_book", None)
        if fetch is None:
            return None
        try:
            book = await fetch(market)
        except Exception:  # noqa: BLE001 - fail open; never block a signal on a book-fetch error
            log.warning("Book fetch failed for %s; skipping spread guard", market.name)
            return None
        sp = HLClient.book_spread(book)
        if sp is None:
            return None
        _, _, _, spread_pct = sp
        if r.max_spread_pct > 0 and spread_pct > r.max_spread_pct:
            return f"spread {spread_pct:.2%} > {r.max_spread_pct:.2%} (thin book)"
        if r.min_top_depth_usd > 0:
            depth = HLClient.top_depth_usd(book)
            if 0 < depth < r.min_top_depth_usd:
                return f"book depth ${depth:,.0f} < ${r.min_top_depth_usd:,.0f} (thin book)"
        return None

    async def evaluate(self, item: NewsItem, analysis: Analysis,
                       market: Optional[Market]) -> Decision:
        rt = self.config.runtime
        rt.maybe_auto_resume()   # a daily-loss halt clears itself once the UTC day rolls
        if rt.trading_halted:    # manual kill switch (or, production, the global daily halt) -> all
            return self._reject(item, f"trading halted: {rt.halt_reason}", analysis, market)
        # Per-lane daily-loss halt (arena): this model is paused for the day, others keep trading.
        if self.model_id:
            halted, why = rt.is_daily_halted(self.model_id)
            if halted:
                return self._reject(item, f"daily loss halt: {why}", analysis, market)
        if analysis.error:
            return self._reject(item, f"analyzer error: {analysis.error}", analysis, market)
        if analysis.direction == "none" or not analysis.ticker:
            return self._reject(item, "no actionable signal", analysis, market)
        if analysis.is_stale:
            return self._reject(item, "stale / already priced-in", analysis, market)
        if market is None:
            return self._reject(item, f"'{analysis.ticker}' not tradable on enabled dexes",
                                analysis, market)
        if market.symbol.upper() in self._blacklist:
            return self._reject(item, f"{market.symbol} is blacklisted", analysis, market)

        # Duplicate-event suppression: the same (ticker, direction) seen recently — even if
        # the earlier one faded below the gate — is one event, not a fresh signal. Recorded
        # for every directional read so two near-identical headlines get a consistent outcome.
        now_ms = int(time.time() * 1000)
        dkey = (market.symbol.upper(), analysis.direction)
        prev = self._recent_signal.get(dkey)
        fp = dup_fingerprint(item.text)
        self._recent_signal[dkey] = (now_ms, fp)
        # Suppress only a TRUE repeat: same (ticker, direction) AND the same STORY. Two distinct
        # catalysts that both resolve to the same ticker+direction (e.g. Western Digital earnings
        # -> SNDK long, then SanDisk's OWN earnings -> SNDK long a minute later) are different
        # events and must not block each other.
        if (self.r.duplicate_window_seconds and prev
                and (now_ms - prev[0]) < self.r.duplicate_window_seconds * 1000
                and dup_similarity(fp, prev[1]) >= self.r.duplicate_similarity_min):
            return self._reject(item, "duplicate of a recent signal (same ticker+direction)",
                                analysis, market)

        # Deterministic confidence haircuts (illiquid alts, competitive listings, 2nd-order).
        mentioned = True
        if self.universe is not None and self.r.indirect_mention_penalty:
            # The analyzer's own read that this asset is the news SUBJECT (not a 2nd-order
            # inference) clears the haircut even when the ticker isn't named verbatim (e.g.
            # "SpaceX" -> SPCX); the regex check is the deterministic backstop.
            mentioned = analysis.subject_relation == "direct" or self.universe.mentions(
                market.symbol, item.text,
                hints=[item.coin_hint or "", *item.symbol_hints])
        eff_conf, notes = adjust_confidence(analysis.confidence, market, item.text, self.r,
                                            mentioned=mentioned)
        if eff_conf < self.r.confidence_threshold:
            extra = f" [{', '.join(notes)}]" if notes else ""
            return self._reject(
                item, f"confidence {eff_conf:.2f} < {self.r.confidence_threshold:.2f}{extra}",
                analysis, market)

        # Daily loss kill switch: realized PnL today PLUS current open-position drawdown, so a
        # book that's deep underwater on OPEN positions halts new entries too (not just after
        # losses are realized). Both are scoped to the current mode.
        dry = self.dry_run_override if self.dry_run_override is not None else rt.dry_run
        realized = await self.store.realized_pnl_today(dry, model_id=self.model_id)
        unrealized = self.pm.unrealized_pnl(dry)
        if realized + unrealized <= -abs(self.r.daily_loss_limit_usd):
            # PER-LANE in the arena (model_id set): halts only this wallet, not the others.
            rt.halt_daily(f"daily loss limit hit (realized {realized:.2f} + "
                          f"unrealized {unrealized:.2f})", model_id=self.model_id)  # auto-resumes next UTC day
            return self._reject(item, "daily loss limit hit", analysis, market)

        if self.pm.has_open(market.name):
            return self._reject(item, "position already open on this market", analysis, market)
        if self.pm.open_count() >= self.r.max_concurrent_positions:
            return self._reject(item, "max concurrent positions reached", analysis, market)

        last = self._last_entry_ms.get(market.symbol.upper())
        if last and (time.time() * 1000 - last) < self.r.per_ticker_cooldown_seconds * 1000:
            return self._reject(item, "ticker cooldown active", analysis, market)

        price = await self.hl.mid(market)
        if not price or price <= 0:
            return self._reject(item, "no live price for market", analysis, market)

        # Pre-news move: measured when the already-moved guard OR the skeptic confirmation
        # needs it (one candle fetch, ~100-300ms, only for would-be entries). Fails OPEN.
        ca = self.config.app.analyzer
        confirm_on = self.confirmer is not None and getattr(ca, "confirm_entries", False)
        guard_on = self.r.pre_move_haircut_pct > 0 or self.r.pre_move_reject_pct > 0
        pre_move = None
        if guard_on or confirm_on:
            ref = await self._pre_news_ref(market, item.time_ms)
            if ref:
                pre_move = signed_move_pct(ref, price, analysis.direction)

        # Already-moved guard: how much of the repricing already happened without us? Note
        # this measures to REAL-TIME (incl. the post-news seconds where HFT lives), so live
        # is strictly tighter than the backtest's next-1m-open version.
        if guard_on and pre_move is not None:
            eff_conf, mg_notes, reject = apply_move_guard(
                eff_conf, pre_move, analysis.time_sensitivity, self.r)
            notes += mg_notes
            if reject:
                return self._reject(item, reject, analysis, market)
            if eff_conf < self.r.confidence_threshold:
                return self._reject(
                    item, f"confidence {eff_conf:.2f} < {self.r.confidence_threshold:.2f} "
                          f"after move guard [{', '.join(notes)}]", analysis, market)

        # Live-book liquidity guard (thin off-hours equities): reject a too-wide/too-thin book.
        thin = await self.liquidity_guard(market)
        if thin:
            return self._reject(item, thin, analysis, market)

        # Skeptic entry confirmation (post-gate, pre-sizing): a stronger model argues against
        # the trade WITH the tape + measured pre-move in hand. Direction veto rejects; rule
        # "min" sizes off min(first-pass, skeptic) and re-gates at confirm_gate. Fails OPEN
        # (verdict None) so an API blip can't kill a validated catalyst. Runs inside the
        # serialized trade section — the duration is logged because it holds the trade lock.
        if confirm_on:
            from ..analysis.market_context import build_market_context
            t0 = time.monotonic()
            ctx = await build_market_context(self.hl, market, now_ms)
            verdict = await self.confirmer(item, analysis, market_context=ctx,
                                           pre_move_pct=pre_move)
            took = time.monotonic() - t0
            if verdict is None:
                log.warning("Entry confirmation unavailable for %s after %.1fs — failing OPEN",
                            market.name, took)
            else:
                log.info("Entry confirmation for %s: agree=%s conf=%.2f in %.1fs (%s)",
                         market.name, verdict["agree_direction"], verdict["confidence"],
                         took, verdict["risk"][:120])
                if not verdict["agree_direction"]:
                    return self._reject(
                        item, f"confirmation veto: {verdict['risk'][:160]}", analysis, market)
                if ca.confirm_rule != "veto_only":
                    eff_conf = min(eff_conf, verdict["confidence"])
                    if eff_conf < ca.confirm_gate:
                        return self._reject(
                            item, f"confirmation confidence {verdict['confidence']:.2f} -> "
                                  f"min {eff_conf:.2f} < confirm gate {ca.confirm_gate:.2f}",
                            analysis, market)

        notional = scale_notional(eff_conf, self.r) * premarket_factor(market, self.r)

        # Clamp to remaining exposure budget.
        remaining = self.r.max_total_exposure_usd - self.pm.total_exposure()
        notional = min(notional, remaining)
        if notional < MIN_ORDER_USD:
            return self._reject(item, f"exposure budget exhausted (room ${remaining:.0f})",
                                analysis, market)

        size = round(notional / price, market.sz_decimals)
        if size <= 0:
            return self._reject(item, "size rounds to zero", analysis, market)
        notional = size * price  # recompute after rounding
        if notional < MIN_ORDER_USD:
            # Rounding can drop the effective notional below the exchange minimum — the
            # live order would be rejected ("order value too small") while paper fills.
            return self._reject(item, f"rounded notional ${notional:.2f} below exchange minimum",
                                analysis, market)

        leverage = max(1, min(self.r.max_leverage, market.max_leverage or self.r.max_leverage))
        side = "long" if analysis.direction == "long" else "short"
        horizon_s, stop_pct, tp_pct, trail_pct = exit_params(analysis.time_sensitivity, self.r)
        stop_loss, take_profit = compute_sl_tp(side, price, stop_pct, tp_pct)

        return Decision(
            news_id=item.id, action="enter", reason=analysis.rationale, market=market,
            side=side, notional_usd=notional, size=size, leverage=leverage, entry_px=price,
            stop_loss=stop_loss, take_profit=take_profit, trail_pct=trail_pct,
            time_exit_seconds=horizon_s, confidence=eff_conf,
        )
