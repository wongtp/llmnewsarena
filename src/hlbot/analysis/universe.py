"""Tradable market universe: maps a ticker/symbol to a Hyperliquid market.

Built from `info.meta(dex="")` (crypto perps) and `info.meta(dex="xyz")` (trade.xyz
equities/indices/commodities). The bare symbol (e.g. MRVL from "xyz:MRVL") is the
lookup key. When the same symbol exists on multiple dexes, the analyzer's
asset_class hint disambiguates.
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
import time

from ..models import Market
from ..trading.hl_client import HLClient

log = logging.getLogger("hlbot.universe")

LISTING_SEEN_FILE = "data/listing_seen.json"   # persisted {market name -> first-seen ms}

# Small fallback alias map (company/asset name -> ticker). The analyzer is also
# given the exact symbol list and asked to return a listed symbol, so this only
# backstops the rare case where it returns a name instead of a ticker.
ALIASES = {
    "nvidia": "NVDA", "marvell": "MRVL", "tesla": "TSLA", "apple": "AAPL",
    "amazon": "AMZN", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "meta platforms": "META", "facebook": "META", "netflix": "NFLX",
    "advanced micro devices": "AMD", "broadcom": "AVGO", "palantir": "PLTR",
    "coinbase": "COIN", "microstrategy": "MSTR", "strategy": "MSTR",
    # Ticker variants fold to one canonical so GOOG == GOOGL == "Google"/"Alphabet".
    "goog": "GOOGL",
    # Common name!=ticker cases (so the direct-mention check isn't fooled by name-only
    # headlines lacking a cashtag). Extend as you see false "-indirect" haircuts in backtests.
    "intel": "INTC", "micron": "MU", "qualcomm": "QCOM", "oracle": "ORCL",
    "taiwan semiconductor": "TSM", "tsmc": "TSM", "circle": "CRCL", "lumentum": "LITE",
    "super micro": "SMCI", "supermicro": "SMCI", "alibaba": "BABA",
    # Pre-IPO / premarket names whose company name != ticker (else a name-only headline like
    # "SpaceX IPO..." reads as an INDIRECT mention of SPCX and gets the haircut).
    "spacex": "SPCX", "space x": "SPCX",
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
    "dogecoin": "DOGE",
}


class Universe:
    def __init__(self, hl: HLClient, allowed_dexes: list[str]):
        self.hl = hl
        self.allowed_dexes = allowed_dexes
        self.by_symbol: dict[str, list[Market]] = {}
        self._pattern: re.Pattern | None = None
        self._first_seen: dict[str, float] = self._load_first_seen()

    @staticmethod
    def _load_first_seen() -> dict[str, float]:
        p = pathlib.Path(LISTING_SEEN_FILE)
        if p.exists():
            try:
                return {k: float(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _save_first_seen(self) -> None:
        try:
            p = pathlib.Path(LISTING_SEEN_FILE)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._first_seen), encoding="utf-8")
        except Exception:  # noqa: BLE001
            log.debug("Could not persist listing-seen map")

    def _build_pattern(self) -> re.Pattern:
        tokens = set(self.by_symbol) | {a.upper() for a in ALIASES} | set(ALIASES)
        parts = sorted((re.escape(t) for t in tokens if t), key=len, reverse=True)
        return re.compile(r"(?<![A-Za-z0-9$])(" + "|".join(parts) + r")(?![A-Za-z0-9])",
                          re.IGNORECASE)

    def matches(self, text: str) -> bool:
        """Fast, free pre-filter: does the text mention a tradable symbol or known alias?"""
        return bool(self._pattern and self._pattern.search(text or ""))

    @staticmethod
    def mentions(symbol: str | None, text: str, hints=()) -> bool:
        """Is THIS specific ticker directly referenced in the news — by symbol, cashtag
        ($MRVL), a ticker variant, or a known company/asset name (e.g. 'Nvidia' for NVDA)?
        Ticker variants + the name are folded into ONE equivalence group via ALIASES, so
        GOOG == GOOGL == 'Google' (or 'Alphabet') all count, regardless of which class the
        universe lists. Also true if a feed auto-detected it (Tree coin/symbol hints). Used to
        flag 2nd-order calls where the resolved asset is only an INFERENCE (e.g. 'Alphabet
        capex up' -> NVDA), which we de-prioritize. Returns True when the symbol is unknown so
        we never penalize blindly."""
        sym = (symbol or "").strip().upper()
        if not sym:
            return True
        canon = ALIASES.get(sym.lower(), sym)   # fold variants/names to one canonical ticker
        names = {sym, canon} | {a.upper() for a, s in ALIASES.items() if s == canon}
        if any((h or "").strip().upper() in names for h in hints):
            return True
        for n in names:
            if re.search(r"(?<![A-Za-z0-9])\$?" + re.escape(n) + r"(?![A-Za-z0-9])",
                         text or "", re.IGNORECASE):
                return True
        return False

    async def refresh(self) -> int:
        # 24h notional volume per crypto coin (liquidity proxy) from asset contexts.
        volume: dict[str, float] = {}
        try:
            meta_ctx = await self.hl.meta_and_asset_ctxs()
            cmeta, ctxs = meta_ctx[0], meta_ctx[1]
            for a, ctx in zip(cmeta.get("universe", []), ctxs):
                try:
                    volume[a["name"]] = float(ctx.get("dayNtlVlm", 0) or 0)
                except (TypeError, ValueError):
                    pass
        except Exception:  # noqa: BLE001 - volume is best-effort
            log.debug("Could not load asset contexts for volume")

        # First-ever run (no persisted history): treat the whole existing universe as
        # baseline/old (first_seen=0) so we don't penalize all 300+ markets as "new".
        # Anything appearing in a LATER refresh is genuinely new and gets stamped now.
        baseline = not self._first_seen
        now_ms = time.time() * 1000
        changed = False
        dex_failed = False

        new: dict[str, list[Market]] = {}
        loaded_dexes: set[str] = set()
        for dex in self.allowed_dexes:
            try:
                meta = await self.hl.meta(dex)
            except Exception:  # noqa: BLE001
                log.exception("Failed to load meta for dex=%r", dex)
                dex_failed = True
                continue
            loaded_dexes.add(dex)
            for a in meta.get("universe", []):
                if a.get("isDelisted"):
                    continue
                name = a["name"]                       # "BTC" or "xyz:MRVL"
                symbol = name.split(":")[-1].upper()
                if name not in self._first_seen:
                    self._first_seen[name] = 0.0 if baseline else now_ms
                    changed = True
                    if not baseline:
                        log.info("New market listed: %s (penalized for %s)", name, "first hours")
                market = Market(
                    name=name,
                    symbol=symbol,
                    dex=dex,
                    asset_class="crypto" if dex == "" else "equity",
                    sz_decimals=int(a.get("szDecimals", 2)),
                    max_leverage=int(a.get("maxLeverage", 1)),
                    day_volume_usd=volume.get(name, 0.0) if dex == "" else 0.0,
                    first_seen_ms=self._first_seen[name],
                )
                new.setdefault(symbol, []).append(market)

        # Carry over markets for any dex that FAILED to load this round, so its symbols
        # don't vanish (become untradable) until the next successful refresh. (On the very
        # first run there's nothing to carry over; the baseline guard below handles that.)
        failed_dexes = set(self.allowed_dexes) - loaded_dexes
        if failed_dexes and self.by_symbol:
            carried = 0
            for sym, mkts in self.by_symbol.items():
                for m in mkts:
                    if m.dex in failed_dexes:
                        new.setdefault(sym, []).append(m)
                        carried += 1
            if carried:
                log.warning("Carried over %d markets for failed dex(es) %s", carried, failed_dexes)

        if new:
            self.by_symbol = new
            self._pattern = self._build_pattern()
            if baseline and dex_failed:
                # A dex failed to load on the very first run: don't lock in a partial
                # baseline (it would later mis-flag the missing dex's markets as "new").
                # The loaded markets keep first_seen=0 (no penalty); re-seed next refresh.
                self._first_seen = {}
            elif changed:
                self._save_first_seen()
        log.info("Universe refreshed: %d symbols across dexes %s",
                 len(self.by_symbol), self.allowed_dexes)
        return len(self.by_symbol)

    def resolve(self, ticker: str | None, asset_class: str | None = None) -> Market | None:
        if not ticker:
            return None
        t = ticker.strip().lstrip("$").upper()
        t = ALIASES.get(t.lower(), t)
        candidates = self.by_symbol.get(t)
        if not candidates:
            return None
        # Do NOT cross asset classes on a ticker collision (e.g. Quantinuum the stock
        # vs QNT the crypto token). If the analyzer named a class, require a market of
        # that class; otherwise no trade.
        if asset_class == "crypto":
            pref = [m for m in candidates if m.dex == ""]
            return pref[0] if pref else None
        if asset_class in ("equity", "index", "commodity"):
            pref = [m for m in candidates if m.dex != ""]
            return pref[0] if pref else None
        return candidates[0]  # unknown class -> best effort

    def crypto_symbols(self) -> list[str]:
        return sorted({s for s, ms in self.by_symbol.items() if any(m.dex == "" for m in ms)})

    def equity_symbols(self) -> list[str]:
        return sorted({s for s, ms in self.by_symbol.items() if any(m.dex != "" for m in ms)})
