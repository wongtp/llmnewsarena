"""Thin async wrapper around the Hyperliquid SDK (Info + Exchange).

The SDK is synchronous (requests-based), so every network call is dispatched to a
worker thread via asyncio.to_thread to keep the event loop responsive. Builder
(HIP-3) dexes like `xyz` are enabled by passing perp_dexs; the SDK then resolves
names such as "xyz:MRVL" to the correct asset id automatically.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from ..config import Config
from ..models import Market

log = logging.getLogger("hlbot.hl")


class HLClient:
    REQUEST_TIMEOUT_S = 15.0   # bounds every SDK HTTP call (see _connect_sync)

    def __init__(self, config: Config, address: str | None = None,
                 secret_key: str | None = None):
        # address/secret default to the single production account; the arena passes a per-wallet
        # pair so each model trades its own funded wallet. Read-only use (prices/candles) works
        # with any valid pair, so paper lanes can share one client.
        self.config = config
        self.address = address or config.secrets.hl_account_address
        self._secret_key = secret_key or config.secrets.hl_secret_key
        self.info = None
        self.exchange = None
        # "" (crypto) + any builder dexes (e.g. "xyz") the user enabled.
        self._dexes = list(config.app.filters.allowed_dexes)
        self._mids: dict[str, dict] = {}        # dex -> {coin: mid}
        self._mids_ts: dict[str, float] = {}    # dex -> last refresh time
        # Cached mids stay valid longer than the refresh cadence (mid_cache_loop, 2s), so
        # mid() never falls back to a blocking live fetch just because the loop is
        # mid-refresh — the TTL must exceed interval + fetch time.
        self._mid_ttl = 5.0
        self._frates: dict[str, dict[str, float]] = {}   # dex -> {coin: hourly funding rate}
        self._frates_ts: dict[str, float] = {}

    # ---- connection -------------------------------------------------------
    def _connect_sync(self) -> str:
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        wallet = Account.from_key(self._secret_key)
        # timeout is mandatory: requests with timeout=None waits FOREVER on a dead TCP
        # connection, which would wedge the position monitor (no stop/trail/time exits) or
        # hold the pipeline's trade lock indefinitely. supervised() only restarts on
        # exceptions, never on hangs — so a bounded timeout is the hang-recovery mechanism.
        self.info = Info(self.config.base_url, skip_ws=True, perp_dexs=self._dexes,
                         timeout=self.REQUEST_TIMEOUT_S)
        self.exchange = Exchange(
            wallet, self.config.base_url, account_address=self.address, perp_dexs=self._dexes,
            timeout=self.REQUEST_TIMEOUT_S,
        )
        return wallet.address

    async def connect(self) -> str:
        agent = await asyncio.to_thread(self._connect_sync)
        log.info("Hyperliquid connected (agent %s, account %s, dexes=%s)",
                 agent, self.address, self._dexes)
        return agent

    # ---- reads ------------------------------------------------------------
    async def meta(self, dex: str = "") -> dict:
        return await asyncio.to_thread(self.info.meta, dex)

    async def meta_and_asset_ctxs(self):
        return await asyncio.to_thread(self.info.meta_and_asset_ctxs)

    async def all_mids(self, dex: str = "") -> dict:
        return await asyncio.to_thread(self.info.all_mids, dex)

    async def position_state(self, dex: str = "") -> dict[str, dict]:
        """coin -> {"szi": signed size, "funding": cumulative funding since open (USD, + = paid)}
        for every open exchange position on the dex. ONE user_state call feeds both the
        closed-position reconciler and the live funding column in the UI."""
        st = await asyncio.to_thread(self.info.user_state, self.address, dex)

        def fnum(v) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        out: dict[str, dict] = {}
        for ap in st.get("assetPositions", []):
            pos = ap.get("position") or {}
            coin = pos.get("coin")
            if coin is None:
                continue
            out[coin] = {"szi": fnum(pos.get("szi")),
                         "funding": fnum((pos.get("cumFunding") or {}).get("sinceOpen"))}
        return out

    async def positions(self, dex: str = "") -> dict[str, float]:
        """Map coin -> signed open-position size currently on the exchange (per dex).
        An empty/zero entry means flat."""
        return {c: v["szi"] for c, v in (await self.position_state(dex)).items()}

    async def funding_rate(self, market: Market) -> float:
        """Current HOURLY funding rate for a market, cached ~60s per dex. Used to accrue an
        estimated funding cost on PAPER positions; live positions read the real cumFunding
        from user_state instead. 0.0 when unknown."""
        dex = market.dex
        if time.time() - self._frates_ts.get(dex, 0) > 60:
            self._frates_ts[dex] = time.time()   # stamp first: a failing dex retries in 60s, not per call
            try:
                meta, ctxs = await self.meta_and_ctxs_raw(dex)
                rates: dict[str, float] = {}
                for a, ctx in zip(meta.get("universe", []), ctxs):
                    try:
                        rates[a["name"]] = float(ctx.get("funding") or 0)
                    except (TypeError, ValueError):
                        pass
                self._frates[dex] = rates
            except Exception:  # noqa: BLE001 - estimate-only; keep the stale table
                log.debug("funding-rate refresh failed for dex=%r", dex)
        rates = self._frates.get(dex, {})
        return rates.get(market.name, rates.get(market.symbol, 0.0))

    async def account_value(self) -> Optional[float]:
        """Total account equity under Hyperliquid's UNIFIED margin: spot USDC + perp account
        value (which already includes open-position PnL) summed across the enabled dexes. The
        funds usually sit in spot until a position opens, so reading perp marginSummary alone
        reports $0 on a funded account — this sums both. None if every read failed."""
        total, ok = 0.0, False
        try:
            sp = await asyncio.to_thread(self.info.spot_user_state, self.address)
            for b in (sp or {}).get("balances", []):
                if b.get("coin") == "USDC":
                    total += float(b.get("total", 0) or 0)
                    ok = True
        except Exception:  # noqa: BLE001
            log.debug("spot balance fetch failed")
        for dex in self._dexes:
            try:
                st = await asyncio.to_thread(self.info.user_state, self.address, dex)
                total += float((st or {}).get("marginSummary", {}).get("accountValue", 0) or 0)
                ok = True
            except Exception:  # noqa: BLE001
                log.debug("perp state fetch failed for dex=%r", dex)
        return total if ok else None

    async def l2_book(self, market: Market) -> dict:
        """Live L2 order book for a market: {levels: [[bids], [asks]], ...} with string px/sz."""
        return await asyncio.to_thread(self.info.l2_snapshot, market.name)

    async def meta_and_ctxs_raw(self, dex: str = "") -> list:
        """metaAndAssetCtxs for ONE dex -> [meta, ctxs] (ctxs aligned with meta.universe).
        Raw POST because the SDK helper has no dex param; builder dexes (xyz) need it."""
        return await asyncio.to_thread(
            self.info.post, "/info", {"type": "metaAndAssetCtxs", "dex": dex})

    async def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list:
        """OHLCV candle snapshot for a coin name ("BTC" or "xyz:MRVL"); list of dicts with
        t/T (open/close ms), o/h/l/c (string prices), v (base volume), n (trades)."""
        req = {"coin": coin, "interval": interval,
               "startTime": int(start_ms), "endTime": int(end_ms)}
        out = await asyncio.to_thread(self.info.post, "/info",
                                      {"type": "candleSnapshot", "req": req})
        return out if isinstance(out, list) else []

    @staticmethod
    def book_spread(book: dict) -> Optional[tuple[float, float, float, float]]:
        """From an L2 snapshot return (best_bid, best_ask, mid, spread_pct), or None if the
        book is empty / one-sided / crossed (can't judge -> caller fails open)."""
        try:
            bids, asks = book["levels"][0], book["levels"][1]
            bid, ask = float(bids[0]["px"]), float(asks[0]["px"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        return bid, ask, mid, (ask - bid) / mid

    @staticmethod
    def top_depth_usd(book: dict, levels: int = 10) -> float:
        """Cumulative near-touch notional (USD) of the THINNER side across the top `levels`
        price levels — a shallow-book proxy for the liquidity guard. Summing several levels (not
        just the single best quote) avoids a transient tiny top-of-book quote reading as 'no
        liquidity' when there's real depth right behind it. 0 if unknown / one-sided."""
        try:
            bids, asks = book["levels"][0], book["levels"][1]
            bid_usd = sum(float(lv["px"]) * float(lv["sz"]) for lv in bids[:levels])
            ask_usd = sum(float(lv["px"]) * float(lv["sz"]) for lv in asks[:levels])
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.0
        if bid_usd <= 0 or ask_usd <= 0:
            return 0.0
        return min(bid_usd, ask_usd)

    @staticmethod
    def _pick_mid(mids: dict, market: Market) -> Optional[float]:
        for key in (market.name, market.symbol):
            if key in mids:
                try:
                    return float(mids[key])
                except (TypeError, ValueError):
                    pass
        return None

    async def mid(self, market: Market) -> Optional[float]:
        """Cached mid (kept fresh by mid_cache_loop); falls back to a live fetch."""
        cached = self._mids.get(market.dex)
        if cached and (time.time() - self._mids_ts.get(market.dex, 0)) < self._mid_ttl:
            # Fresh cache that lacks the symbol means the market has no mid right now
            # (delisted/renamed) — a live refetch would return the same answer, so don't
            # burn an RTT per call on it (the monitor polls such positions every 3s).
            return self._pick_mid(cached, market)
        mids = await self.all_mids(market.dex)
        self._mids[market.dex] = mids
        self._mids_ts[market.dex] = time.time()
        return self._pick_mid(mids, market)

    async def mid_cache_loop(self, dexes: list[str], interval: float = 2.0) -> None:
        """Background task: keep the mid cache warm so mid() is hot-path free. Dexes are
        refreshed concurrently — sequential fetches add one RTT per dex of staleness."""
        async def refresh(dex: str) -> None:
            try:
                self._mids[dex] = await self.all_mids(dex)
                self._mids_ts[dex] = time.time()
            except Exception:  # noqa: BLE001 - best effort; mid() falls back to live
                pass

        while True:
            await asyncio.gather(*(refresh(d) for d in dexes))
            await asyncio.sleep(interval)

    # ---- writes -----------------------------------------------------------
    async def set_leverage(self, market: Market, leverage: int) -> dict:
        return await asyncio.to_thread(self.exchange.update_leverage, int(leverage), market.name, True)

    async def market_open(self, market: Market, is_buy: bool, size: float, slippage: float) -> dict:
        return await asyncio.to_thread(
            self.exchange.market_open, market.name, is_buy, size, None, slippage
        )

    async def market_close(self, market: Market, slippage: float) -> dict:
        return await asyncio.to_thread(
            self.exchange.market_close, market.name, None, None, slippage
        )

    def _place_stop_sync(self, name: str, is_buy_to_close: bool, size: float,
                         trigger_px: float, slippage: float) -> dict:
        # A reduce-only stop-MARKET trigger: rests on the book and converts to an aggressive
        # market order when the trigger is breached. _slippage_price rounds both the trigger
        # and the (aggressive) limit to the asset's valid tick (the same rounding market_open
        # relies on). slippage=0 yields the trigger price itself, just rounded.
        trig = self.exchange._slippage_price(name, is_buy_to_close, 0.0, trigger_px)
        limit_px = self.exchange._slippage_price(name, is_buy_to_close, slippage, trigger_px)
        order_type = {"trigger": {"triggerPx": trig, "isMarket": True, "tpsl": "sl"}}
        return self.exchange.order(name, is_buy_to_close, float(size), limit_px, order_type, True)

    async def place_stop(self, market: Market, is_buy_to_close: bool, size: float,
                         trigger_px: float, slippage: float) -> dict:
        """Place a reduce-only stop-market trigger that protects the position on the exchange
        even if the bot is offline. is_buy_to_close: True closes a SHORT (buy), False closes a
        LONG (sell). Backstop only — fixed at the entry stop; the bot-side monitor handles the
        trailing/TP/time exits and cancels this on a normal close."""
        return await asyncio.to_thread(
            self._place_stop_sync, market.name, is_buy_to_close, size, trigger_px, slippage)

    async def cancel_order(self, market: Market, oid: int) -> dict:
        return await asyncio.to_thread(self.exchange.cancel, market.name, int(oid))

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def parse_resting_oid(result: dict) -> Optional[int]:
        """Extract the resting order id from an order response (a stop trigger rests rather
        than filling). Returns None if it didn't rest (error / immediate fill / unparsable)."""
        try:
            if result.get("status") != "ok":
                return None
            for st in result["response"]["data"]["statuses"]:
                if "resting" in st:
                    return int(st["resting"]["oid"])
                if "error" in st:
                    return None
        except (KeyError, TypeError, ValueError):
            return None
        return None

    @staticmethod
    def parse_fill(result: dict) -> tuple[Optional[float], Optional[float], Optional[str]]:
        """Extract (avg_px, total_sz, error) from an order/market_open response."""
        if not isinstance(result, dict):
            # SDK market_close returns None when no matching position exists on the exchange.
            return None, None, "no position / empty order response"
        try:
            if result.get("status") != "ok":
                return None, None, str(result)
            statuses = result["response"]["data"]["statuses"]
            for st in statuses:
                if "error" in st:
                    return None, None, st["error"]
                fill = st.get("filled")
                if fill:
                    return float(fill["avgPx"]), float(fill["totalSz"]), None
                if "resting" in st:
                    return None, None, "order resting (not filled)"
            return None, None, "no fill in response"
        except (KeyError, TypeError, ValueError) as exc:
            return None, None, f"unparsable response: {exc} :: {result}"
