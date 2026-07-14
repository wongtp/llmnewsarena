"""Backtest engine: replay historical Tree-of-Alpha news through the live analyzer
and simulate trade outcomes against Hyperliquid historical candles.

Reuses the exact production logic: NewsItem parsing, the Claude analyzer, the
signal gate (confidence threshold + stale filter), confidence-scaled sizing
(`scale_notional`) and `compute_sl_tp`. Only the price source differs — historical
candles instead of live mids — and exits are evaluated bar-by-bar.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import pathlib
import re
import time
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Optional

import httpx

from ..analysis.analyzer import Analyzer
from ..analysis.burst import BurstBuffer, build_burst_item
from ..analysis.prompts import supports_temperature
from ..analysis.universe import ALIASES, Universe
from ..config import Config
from ..models import Analysis, NewsItem
from ..trading.executor import FEE_RATE_PER_SIDE
from ..trading.hl_client import HLClient
from ..trading.risk import (
    MIN_ORDER_USD,
    adjust_confidence,
    apply_move_guard,
    compute_sl_tp,
    dup_fingerprint,
    dup_similarity,
    exit_params,
    premarket_factor,
    ref_price_from_candles,
    scale_notional,
    signed_move_pct,
)

log = logging.getLogger("hlbot.backtest")

HISTORY_URL = "https://news.treeofalpha.com/api/news"

# In a historical replay, present each item to the analyzer as if it just arrived
# (a realistic detection latency), so age isn't mistaken for staleness.
BACKTEST_FRESH_AGE_S = 10.0
# If the nearest candle at/after the news is more than this late, there was no live price
# at news time (e.g. the asset wasn't listed yet) -> reject the entry (prevents phantom
# trades entered at a much-later listing price, which inflates PnL).
BACKTEST_MAX_ENTRY_DELAY_S = 7200   # 2h


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def pick_entry(candles: list[dict], news_ms: int,
               max_delay_ms: Optional[int] = None) -> tuple[Optional[int], Optional[float]]:
    """Entry at the OPEN of the first candle starting at/after the news time (a realistic
    ~1-bar delay, avoids using the post-news bar's own movement). If max_delay_ms is set
    and the nearest such candle is more than that after the news, there was no tradable
    price at news time (e.g. the asset wasn't listed yet) -> (None, None), no entry."""
    for i, c in enumerate(candles):
        if int(c["t"]) >= news_ms:
            if max_delay_ms is not None and int(c["t"]) - news_ms > max_delay_ms:
                return None, None
            return i, float(c["o"])
    return None, None


@dataclass
class WalkResult:
    """Outcome of walking a position through candles. Excursions are upper bounds on
    coarse candles (intra-bar ordering is unknowable), measured through the exit bar."""
    exit_px: float
    reason: str
    exit_ms: int
    mae_pct: float = 0.0       # max adverse excursion, fraction of entry (>= 0)
    mfe_pct: float = 0.0       # max favorable excursion, fraction of entry (>= 0)
    time_to_peak_ms: int = 0   # entry bar -> the bar that set the MFE high-water mark


def walk_candles(side: str, entry_px: float, stop_px: float, tp_px: float, trail_pct: float,
                 candles: list[dict], start_index: int, horizon_end_ms: int,
                 breakeven_arm_pct: float = 0.0,
                 breakeven_offset_pct: float = 0.0) -> WalkResult:
    """Walk bars forward from entry; return a WalkResult.

    Supports a trailing stop (trail_pct > 0): once in profit, the stop trails the
    best price by trail_pct. tp_px == 0 means no fixed take-profit (let it run).
    A breakeven stop (breakeven_arm_pct > 0) floors the effective stop at
    entry*(1 +/- breakeven_offset_pct) once the peak has moved breakeven_arm_pct in
    our favor; the binding floor (initial / trailing / breakeven) decides the reason.
    Stop is checked before TP within a bar (pessimistic); the peak is updated after
    the checks so a bar's own extreme can't pre-emptively tighten OR arm the stop. A
    bar that OPENS through the stop fills at its open (a gap can't fill at the stop
    price); TP fills stay at tp_px (filling a gap-up at the open would be better
    than tp, so this is already pessimistic). MAE/MFE are tracked from per-bar
    extremes independently of the trailing peak."""
    peak = entry_px
    adverse = favor = entry_px           # excursion extremes (price space)
    entry_ms = int(candles[start_index]["t"])
    favor_ms = entry_ms
    last = candles[-1]

    def res(px: float, reason: str, ct: int) -> WalkResult:
        if side == "long":
            mae = max(0.0, (entry_px - adverse) / entry_px)
            mfe = max(0.0, (favor - entry_px) / entry_px)
        else:
            mae = max(0.0, (adverse - entry_px) / entry_px)
            mfe = max(0.0, (entry_px - favor) / entry_px)
        return WalkResult(px, reason, ct, mae, mfe, max(0, favor_ms - entry_ms))

    def eff_stop() -> tuple[float, str]:
        """Binding stop floor and its exit-reason label, from the current peak."""
        eff, label = stop_px, "stop loss"
        if side == "long":
            if trail_pct > 0 and peak > entry_px:
                t = peak * (1 - trail_pct)
                if t > eff:
                    eff, label = t, "trailing stop"
            if breakeven_arm_pct > 0 and peak >= entry_px * (1 + breakeven_arm_pct):
                b = entry_px * (1 + breakeven_offset_pct)
                if b > eff:
                    eff, label = b, "breakeven stop"
        else:
            if trail_pct > 0 and peak < entry_px:
                t = peak * (1 + trail_pct)
                if t < eff:
                    eff, label = t, "trailing stop"
            if breakeven_arm_pct > 0 and peak <= entry_px * (1 - breakeven_arm_pct):
                b = entry_px * (1 - breakeven_offset_pct)
                if b < eff:
                    eff, label = b, "breakeven stop"
        return eff, label

    for c in candles[start_index:]:
        o, hi, lo = float(c["o"]), float(c["h"]), float(c["l"])
        close, ct = float(c["c"]), int(c["T"])
        eff, label = eff_stop()
        if side == "long":
            if o <= eff:   # bar gapped through the stop: the fill is the open, not the stop
                adverse = min(adverse, o)
                return res(o, label + " (gap)" if label != "stop loss" else "stop loss (gap)", ct)
            adverse = min(adverse, lo)
            if hi > favor:
                favor, favor_ms = hi, ct
            if lo <= eff:
                return res(eff, label, ct)
            if tp_px > 0 and hi >= tp_px:
                return res(tp_px, "take profit", ct)
            peak = max(peak, hi)
        else:
            if o >= eff:
                adverse = max(adverse, o)
                return res(o, label + " (gap)" if label != "stop loss" else "stop loss (gap)", ct)
            adverse = max(adverse, hi)
            if lo < favor:
                favor, favor_ms = lo, ct
            if hi >= eff:
                return res(eff, label, ct)
            if tp_px > 0 and lo <= tp_px:
                return res(tp_px, "take profit", ct)
            peak = min(peak, lo)
        if ct >= horizon_end_ms:
            return res(close, "time exit", ct)
    return res(float(last["c"]), "time exit (data end)", int(last["T"]))


def bt_slippage_pct(r) -> float:
    """Backtest fill slippage per side. Defaults to the dry-run model so paper and
    backtest PnL share one realism assumption; override with risk.backtest_slippage_pct."""
    v = getattr(r, "backtest_slippage_pct", None)
    return r.dry_run_slippage_pct if v is None else v


def slip_entry(side: str, px: float, r) -> float:
    """Adverse entry fill: the IOC pays up (long) / down (short) from the printed price."""
    s = bt_slippage_pct(r)
    return px * (1 + s) if side == "long" else px * (1 - s)


def slip_exit(side: str, px: float, reason: str, r) -> float:
    """Adverse exit fill. Stop exits (incl. gaps/trails) optionally pay extra
    (backtest_stop_slippage_pct): a stop-market converts to an aggressive IOC into a
    falling/rising market — calibrate from scripts/slippage_report.py once live data exists."""
    s = bt_slippage_pct(r)
    if "stop" in reason:
        s += getattr(r, "backtest_stop_slippage_pct", 0.0)
    return px * (1 - s) if side == "long" else px * (1 + s)


def pnl_usd(side: str, entry_px: float, exit_px: float, size: float) -> float:
    direction = 1 if side == "long" else -1
    gross = (exit_px - entry_px) * size * direction
    fees = (entry_px + exit_px) * size * FEE_RATE_PER_SIDE
    return gross - fees


def utc_day(ms: int) -> str:
    """UTC calendar day for a ms timestamp (matches the live daily-loss window)."""
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Funding (carry cost on multi-hour holds)
# --------------------------------------------------------------------------- #
# Below this hold, funding is <= ~2 hourly accruals — noise next to taker fees. Skip
# the fetch so short-horizon sweeps stay fast.
FUNDING_MIN_HOLD_MS = 2 * 3600_000


class FundingHistory:
    """Disk-cached hourly funding rates per market (re-runs and the optimizer are free).
    Tracks which [start, end] ranges were fetched, so a missing hour inside a covered
    range means 'no funding event' (new market) rather than 'not fetched'. Fails soft:
    a fetch error returns None and is counted in stats, costing 0 funding for that trade."""

    def __init__(self, path: str = "data/bt_funding_cache.json"):
        self.path = pathlib.Path(path)
        self.data: dict[str, dict] = {}   # market -> {"covered": [[s,e],..], "rates": {ms: rate}}
        self.stats = {"applied": 0, "failed": 0}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.data = {}

    def _covered(self, m: dict, start_ms: int, end_ms: int) -> bool:
        return any(s <= start_ms and end_ms <= e for s, e in m.get("covered", []))

    async def rates(self, hl: HLClient, market_name: str,
                    start_ms: int, end_ms: int) -> Optional[list[tuple[int, float]]]:
        """Hourly (event_ms, rate) pairs covering (start_ms, end_ms]. None on fetch failure."""
        m = self.data.setdefault(market_name, {"covered": [], "rates": {}})
        if not self._covered(m, start_ms, end_ms):
            for attempt in range(1, 4):
                try:
                    hist = await _paced(
                        hl.info.funding_history, market_name, int(start_ms), int(end_ms)) or []
                    for h in hist:
                        m["rates"][str(int(h["time"]))] = float(h["fundingRate"])
                    m["covered"].append([int(start_ms), int(end_ms)])
                    self.save()
                    break
                except Exception as exc:  # noqa: BLE001 - fail soft; trade just skips funding
                    # AttributeError/TypeError = a client without funding_history (test fakes,
                    # misconfiguration) — retrying with real sleeps can't fix those.
                    if attempt < 3 and not isinstance(exc, (AttributeError, TypeError)):
                        await asyncio.sleep(8.0 if "429" in str(exc) else 0.5 * attempt)
                        continue
                    self.stats["failed"] += 1
                    log.debug("funding history failed for %s: %s", market_name, exc)
                    return None
        self.stats["applied"] += 1
        return sorted((int(t), r) for t, r in m["rates"].items()
                      if start_ms < int(t) <= end_ms)

    def save(self) -> None:
        try:
            _atomic_write_json(self.path, self.data)   # atomic: safe under parallel arena runs
        except Exception:  # noqa: BLE001
            pass


_FUNDING: FundingHistory | None = None


def funding_cache() -> FundingHistory:
    global _FUNDING
    if _FUNDING is None:
        _FUNDING = FundingHistory()
    return _FUNDING


def funding_usd_from_rates(rates: list[tuple[int, float]], side: str, notional: float,
                           start_ms: int, end_ms: int) -> float:
    """Signed funding over the hold (+ = cost): longs pay positive hourly rates, shorts
    receive them. Notional approximated as constant (entry notional)."""
    sign = 1.0 if side == "long" else -1.0
    return sign * notional * sum(r for t, r in rates if start_ms < t <= end_ms)


async def funding_cost(hl: HLClient, market_name: str, side: str, notional: float,
                       entry_ms: int, exit_ms: int) -> float:
    """Funding cost in USD for a simulated hold; 0.0 for short holds or on fetch failure."""
    if exit_ms - entry_ms <= FUNDING_MIN_HOLD_MS:
        return 0.0
    rates = await funding_cache().rates(hl, market_name, entry_ms, exit_ms)
    if not rates:
        return 0.0
    return funding_usd_from_rates(rates, side, notional, entry_ms, exit_ms)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class BTTrade:
    news_id: str
    time_ms: int
    symbol: str
    market: str
    side: str
    confidence: float
    notional: float
    size: float
    leverage: int
    entry_px: float
    exit_px: float
    pnl: float
    reason: str
    exit_ms: int
    stop_px: float = 0.0
    tp_px: float = 0.0
    trail_pct: float = 0.0
    title: str = ""
    link: Optional[str] = None
    mae_pct: float = 0.0           # max adverse excursion over the hold (fraction of entry)
    mfe_pct: float = 0.0           # max favorable excursion (fraction of entry)
    time_to_peak_ms: int = 0
    candle_interval: str = ""      # which fallback interval simulated the exits (1m best)
    funding_usd: float = 0.0       # carry over the hold (+ = cost), already inside pnl
    pre_move_pct: Optional[float] = None   # in-direction move from pre-news px to entry px


# --------------------------------------------------------------------------- #
# Fetch / filter / analyze
# --------------------------------------------------------------------------- #
async def fetch_history(limit: int = 5000) -> list[NewsItem]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(HISTORY_URL, params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
    arr = data if isinstance(data, list) else data.get("data", [])
    items = [it for it in (NewsItem.from_tree(x) for x in arr) if it]
    items.sort(key=lambda i: i.time_ms)
    return items


from ..analysis.regime import REGIME_PROMPT, summarize_headlines  # noqa: E402,F401


async def summarize_regime(cfg: Config, items: list[NewsItem],
                           model: str = "claude-sonnet-4-6", max_items: int = 1200) -> str:
    """Summarize a set of news items into a market-regime brief via Claude."""
    lines = [f"{it.title} {it.body}".strip()[:200] for it in items[-max_items:]]
    return await summarize_headlines(cfg, lines, model=model, max_items=max_items)


async def load_or_build_regime(cfg: Config, pre_items: list[NewsItem], key: str,
                               rebuild: bool = False) -> str:
    """Reuse a saved regime brief (no token cost) unless rebuild is requested."""
    path = pathlib.Path(f"data/bt_regime_{key}.md")
    if path.exists() and not rebuild:
        return path.read_text(encoding="utf-8")
    brief = await summarize_regime(cfg, pre_items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brief, encoding="utf-8")
    return brief


async def fetch_telegram_history(cfg: Config, channels: list[str], days: float) -> list[NewsItem]:
    """Fetch recent message history from the given Telegram channels via Telethon
    (uses the existing session from scripts/telegram_login.py)."""
    from telethon import TelegramClient

    from ..news.telegram_source import SESSION_PATH

    api_id = int(cfg.secrets.telegram_api_id) if cfg.secrets.telegram_api_id else 0
    api_hash = cfg.secrets.telegram_api_hash
    if not (api_id and api_hash and channels):
        log.error("Telegram source not configured (api id/hash + telegram_channels).")
        return []
    cutoff = time.time() - days * 86400
    items: list[NewsItem] = []
    client = TelegramClient(SESSION_PATH, api_id, api_hash)
    await client.connect()
    try:
        if not await client.is_user_authorized():
            log.error("Telegram session not authorized. Run: python scripts/telegram_login.py")
            return []
        for ch in channels:
            try:
                ent = await client.get_entity(ch)
            except Exception:  # noqa: BLE001
                log.warning("Cannot access channel %r (joined? correct username?)", ch)
                continue
            uname = getattr(ent, "username", None)
            title = getattr(ent, "title", None)
            n = 0
            async for msg in client.iter_messages(ent):
                if msg.date.timestamp() < cutoff:
                    break
                if not msg.message:
                    continue
                it = NewsItem.from_telegram(
                    channel=uname or str(getattr(ent, "id", "")), channel_title=title,
                    msg_id=msg.id, text=msg.message,
                    date_ms=int(msg.date.timestamp() * 1000), chat_id=getattr(ent, "id", ""))
                if it:
                    items.append(it)
                    n += 1
            log.info("  %s: %d messages in last %.0fd", ch, n, days)
    finally:
        await client.disconnect()
    items.sort(key=lambda i: i.time_ms)
    return items


def build_patterns(universe: Universe) -> re.Pattern:
    """One regex matching any tradable symbol (word-boundary) or known alias name."""
    tokens = set(universe.by_symbol.keys()) | {a.upper() for a in ALIASES} | set(ALIASES.keys())
    # longest first so multi-word aliases win; escape for safety
    parts = sorted((re.escape(t) for t in tokens if t), key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(parts) + r")(?![A-Za-z0-9])",
                      re.IGNORECASE)


def passes_prefilter(item: NewsItem, pattern: re.Pattern) -> bool:
    return bool(pattern.search(item.text))


def keep_news(item: NewsItem, cfg: Config, seen: set) -> bool:
    """Backtest dedup/filter: drops dupes (by unique _id only — NOT by author),
    retweets/replies per config, and applies source whitelist/blacklist. Does NOT
    apply the live staleness filter (all historical news is 'old')."""
    if item.id in seen:
        return False
    seen.add(item.id)
    f = cfg.app.filters
    if f.skip_retweets and item.is_retweet:
        return False
    if f.skip_replies and item.is_reply:
        return False
    if f.skip_quotes and item.is_quote:
        return False
    hay = f"{item.source or ''} {item.title or ''}".lower()
    if f.source_blacklist and any(b.lower() in hay for b in f.source_blacklist):
        return False
    if f.source_whitelist and not any(w.lower() in hay for w in f.source_whitelist):
        return False
    return True


# Cached analysis dicts outlive Analysis schema changes (a cache written by another
# checkout may carry fields this one lacks, and vice versa) — load tolerantly: drop
# unknown keys, let dataclass defaults fill missing ones.
_ANALYSIS_FIELDS = {f.name for f in dataclass_fields(Analysis)}


# Abort a run after this many CONSECUTIVE failed analyses (out of credits / bad key / outage):
# failures aren't cached, so churning the rest of the window into errors wastes time/spend with
# nothing to show. Stop loudly + resumably instead. An isolated failure resets the counter.
_MAX_CONSECUTIVE_FAILURES = 25


def _atomic_write_json(path, obj) -> None:
    """Write JSON via a pid-unique temp file + atomic rename. A crash/kill mid-write then can't
    truncate the target (protects the resume-safe caches), and concurrent writers from parallel
    arena runs can't interleave into a corrupt file (each rename is atomic). Same idiom as
    usage_ledger.record."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    tmp.replace(p)


class AnalysisCache:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.data = {}

    def get(self, nid: str) -> Optional[Analysis]:
        d = self.data.get(nid)
        return Analysis(**{k: v for k, v in d.items() if k in _ANALYSIS_FIELDS}) if d else None

    def put(self, a: Analysis) -> None:
        self.data[a.news_id] = a.to_dict()

    def save(self) -> None:
        _atomic_write_json(self.path, self.data)   # atomic: a kill mid-save can't lose progress


class ConfirmCache:
    """Disk cache for skeptic-confirmation verdicts (same shape as AnalysisCache, keyed by
    news id). The confirm prompt embeds the tape + pre-move, so like the analysis cache it
    must be pointed at a FRESH path when the confirm prompt or its inputs change."""

    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.data = {}

    def get(self, nid: str) -> Optional[dict]:
        return self.data.get(nid)

    def put(self, nid: str, verdict: dict) -> None:
        self.data[nid] = verdict

    def save(self) -> None:
        _atomic_write_json(self.path, self.data)


async def analyze_all(analyzer: Analyzer, universe: Universe, items: list[NewsItem],
                      cache: AnalysisCache, concurrency: int = 8,
                      model: str | None = None) -> dict[str, Analysis]:
    out: dict[str, Analysis] = {}
    todo = []
    for it in items:
        cached = cache.get(it.id)
        if cached:
            out[it.id] = cached
        else:
            todo.append(it)
    log.info("Analyzing %d items (%d cached)…", len(todo), len(items) - len(todo))
    sem = asyncio.Semaphore(concurrency)
    done = 0
    fails = 0

    async def one(it: NewsItem):
        nonlocal done, fails
        async with sem:
            a = await analyzer.analyze(it, universe, age_seconds=BACKTEST_FRESH_AGE_S, model=model)
        out[it.id] = a
        if getattr(a, "error", None):
            # Failed call (out-of-credits / rate-limit-exhausted / dropped conn / truncated):
            # leave it UNCACHED so a resume retries it — caching its error 'none' would skip it
            # forever. Abort if failures pile up so we stop early and resumably.
            fails += 1
            if fails >= _MAX_CONSECUTIVE_FAILURES:
                cache.save()
                raise RuntimeError(
                    f"Aborting: {fails} consecutive analysis failures (last: {a.error!r}). "
                    f"Likely out of credits / bad key / API outage. Progress saved to "
                    f"{cache.path}; fix and re-run to resume.")
            return
        fails = 0
        cache.put(a)
        done += 1
        if done % 25 == 0:
            log.info("  analyzed %d/%d", done, len(todo))
            cache.save()

    if todo:
        # Warm the prompt cache before fanning out: a cache entry only becomes readable
        # once the first response has begun, so `concurrency` simultaneous first calls
        # would ALL pay the full uncached/write price on the same shared prefix.
        await one(todo[0])
        await asyncio.gather(*(one(it) for it in todo[1:]))
    cache.save()
    return out


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
# Hyperliquid's /info endpoint rate-limits per IP; a 180d replay fires thousands of candle
# requests, and an unpaced burst gets every one of them 429'd (each then silently dropping a
# trade as "no data"). All backtest /info calls go through _paced(); candle windows that lie
# fully in the past are cached to data/bt_candles/ so A/B re-runs cost zero API calls.
HL_INFO_MIN_GAP_S = 0.35
CANDLE_CACHE_DIR = pathlib.Path("data/bt_candles")
_hl_gap_lock = asyncio.Lock()
_hl_last_call = 0.0


async def _paced(fn, *args):
    """Run a blocking HL info call with a global minimum gap between call starts."""
    global _hl_last_call
    async with _hl_gap_lock:
        wait = _hl_last_call + HL_INFO_MIN_GAP_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _hl_last_call = time.monotonic()
    return await asyncio.to_thread(fn, *args)


def _candle_cache_path(market_name: str, interval: str,
                       start_ms: int, end_ms: int) -> pathlib.Path:
    safe = market_name.replace(":", "_").replace("/", "_")
    return CANDLE_CACHE_DIR / f"{safe}_{interval}_{int(start_ms)}_{int(end_ms)}.json"


async def fetch_candles(hl: HLClient, market_name: str, start_ms: int, end_ms: int,
                        interval: str = "1m", retries: int = 6) -> list[dict]:
    """Candles for a market/interval, disk-cached and rate-limit aware. Retries transient
    fetch errors with backoff — a 429 gets a long exponential wait (the IP quota resets on
    a ~minute window), anything else a short one — because a failed fetch returns [] and
    gets silently misreported as 'no data', dropping a real trade from the backtest. Empty
    (genuine no-data, e.g. not-yet-listed) is NOT retried — only exceptions are. Only
    fully-past windows are cached (a still-open window would freeze partial data)."""
    cpath = _candle_cache_path(market_name, interval, start_ms, end_ms)
    if cpath.exists():
        try:
            return json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt cache entry; refetch
            pass
    for attempt in range(1, retries + 1):
        try:
            out = await _paced(hl.info.candles_snapshot,
                               market_name, interval, start_ms, end_ms) or []
            # Never disk-cache an EMPTY result: a transient empty response (non-exception)
            # would otherwise poison the cache and that trade reads n_no_data forever.
            # Genuine no-data (pre-listing) just re-checks next run — one paced call.
            if out and end_ms < (time.time() - 2 * 3600) * 1000:
                try:
                    CANDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cpath.write_text(json.dumps(out), encoding="utf-8")
                except Exception:  # noqa: BLE001 - cache write is best-effort
                    pass
            return out
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                log.warning("candles failed for %s [%s] after %d tries: %s",
                            market_name, interval, retries, exc)
                return []
            if "429" in str(exc):
                await asyncio.sleep(min(2.0 * 2 ** (attempt - 1), 45.0))
            else:
                await asyncio.sleep(0.4 * attempt)
    return []


async def price_at(hl: HLClient, market_name: str, ms: int) -> Optional[float]:
    """Price at/just after `ms` (open of the first candle >= ms) — used to fill an early
    contrary-news exit at the news time."""
    for iv in ("1m", "5m", "15m"):
        candles = await fetch_candles(hl, market_name, ms - 60_000, ms + 2 * 3600_000, iv)
        if candles:
            _, px = pick_entry(candles, ms, max_delay_ms=BACKTEST_MAX_ENTRY_DELAY_S * 1000)
            if px:
                return px
    return None


async def measure_pre_move(hl: HLClient, market_name: str, news_ms: int, direction: str,
                           lookback_s: int) -> Optional[float]:
    """In-direction move from the pre-news close to the backtest's entry price (open of
    the first bar at/after the news). Falls back to coarser intervals like simulate() —
    HIP-3 equities have sparse 1m bars, which otherwise left pre_move unrecorded on most
    xyz rows (the guard's main audience). None when no interval covers the news — the
    guard then fails open, matching live. NOTE: live measures to real-time mid instead,
    so the live guard is strictly tighter than this."""
    for iv, iv_s in (("1m", 60), ("5m", 300), ("15m", 900)):
        candles = await fetch_candles(
            hl, market_name,
            news_ms - (max(iv_s, lookback_s) + iv_s) * 1000,
            news_ms + BACKTEST_MAX_ENTRY_DELAY_S * 1000, iv)
        ref = ref_price_from_candles(candles, news_ms)
        _, cur = pick_entry(candles, news_ms, max_delay_ms=BACKTEST_MAX_ENTRY_DELAY_S * 1000)
        if ref and cur:
            return signed_move_pct(ref, cur, direction)
    return None


async def simulate(hl: HLClient, market, side: str, news_ms: int, confidence: float,
                   time_sensitivity: str, r, news_id: str = "",
                   notional_cap: Optional[float] = None) -> Optional[BTTrade]:
    horizon_s, stop_pct, tp_pct, trail_pct = exit_params(time_sensitivity, r)
    horizon_ms = horizon_s * 1000
    start, end = news_ms - 60_000, news_ms + horizon_ms + 120_000
    # Prefer fine candles, but HIP-3 equity perps have sparse/short 1m history -
    # fall back to coarser intervals so the trade still simulates (lower SL/TP precision).
    intervals = ["1m", "5m", "15m"] if horizon_s <= 6 * 3600 else ["5m", "15m", "1h"]
    candles: list[dict] = []
    interval = ""
    for iv in intervals:
        candles = await fetch_candles(hl, market.name, start, end, iv)
        if candles:
            interval = iv
            break
    if not candles:
        return None
    idx, raw_entry = pick_entry(candles, news_ms,
                                max_delay_ms=BACKTEST_MAX_ENTRY_DELAY_S * 1000)
    if idx is None or not raw_entry:
        return None   # no live price at news time (asset not listed yet / data gap)
    # Adverse fill slippage on the entry; SL/TP anchored to the FILL (mirrors live
    # Executor._rescale_exits, which re-anchors the configured % distances to the fill).
    entry_px = slip_entry(side, raw_entry, r)
    sl, tp = compute_sl_tp(side, entry_px, stop_pct, tp_pct)
    walk = walk_candles(side, entry_px, sl, tp, trail_pct, candles, idx, news_ms + horizon_ms,
                        breakeven_arm_pct=getattr(r, "breakeven_arm_pct", 0.0),
                        breakeven_offset_pct=getattr(r, "breakeven_offset_pct", 0.0))
    exit_px = slip_exit(side, walk.exit_px, walk.reason, r)
    notional = scale_notional(confidence, r) * premarket_factor(market, r)
    if notional_cap is not None:
        notional = min(notional, notional_cap)   # remaining-exposure clamp (mirrors live)
    # Mirror live sizing exactly (risk.py): round to the market's size precision, reject a
    # zero rounding, recompute notional from the rounded size.
    size = round(notional / entry_px, market.sz_decimals)
    if size <= 0:
        return None
    notional = size * entry_px
    leverage = max(1, min(r.max_leverage, market.max_leverage or r.max_leverage))
    # Funding accrues from the ENTRY bar, not the news time — sparse xyz markets can put
    # the first bar up to 2h after the news, and the position doesn't exist until then.
    entry_ms = int(candles[idx].get("t") or news_ms)
    funding = await funding_cost(hl, market.name, side, notional, max(news_ms, entry_ms),
                                 walk.exit_ms)
    return BTTrade(news_id=news_id, time_ms=news_ms, symbol=market.symbol, market=market.name,
                   side=side, confidence=confidence, notional=notional, size=size,
                   leverage=leverage, entry_px=entry_px, exit_px=exit_px,
                   pnl=pnl_usd(side, entry_px, exit_px, size) - funding,
                   reason=walk.reason, exit_ms=walk.exit_ms, stop_px=sl, tp_px=tp,
                   trail_pct=trail_pct, mae_pct=walk.mae_pct, mfe_pct=walk.mfe_pct,
                   time_to_peak_ms=walk.time_to_peak_ms, candle_interval=interval,
                   funding_usd=funding)


async def truncate_excursions(hl: HLClient, tr: BTTrade, exit_ms: int) -> None:
    """Re-walk a contrary-truncated trade's bars only up to the truncation time, so its
    recorded MAE/MFE/time-to-peak cover the REAL hold — the originals were measured over the
    full simulated horizon and would overstate the excursion percentiles edge_report feeds
    into stop/trail placement. Candles are disk-cached, so the re-walk is free. Best-effort:
    on any data gap the stale values are kept (same as before this existed)."""
    candles = await fetch_candles(hl, tr.market, tr.time_ms - 60_000, exit_ms + 120_000,
                                  tr.candle_interval or "1m")
    if not candles:
        return
    idx, _ = pick_entry(candles, tr.time_ms, max_delay_ms=BACKTEST_MAX_ENTRY_DELAY_S * 1000)
    if idx is None:
        return
    walk = walk_candles(tr.side, tr.entry_px, tr.stop_px, tr.tp_px, tr.trail_pct,
                        candles, idx, exit_ms)
    tr.mae_pct, tr.mfe_pct, tr.time_to_peak_ms = walk.mae_pct, walk.mfe_pct, walk.time_to_peak_ms


async def forward_returns(hl: HLClient, market_name: str, news_ms: int,
                          hours: tuple[int, ...] = (1, 6, 24, 72)) -> dict:
    """Price at news time and forward % moves (for context, ignores SL/TP)."""
    end = news_ms + max(hours) * 3600_000 + 3600_000
    candles = await fetch_candles(hl, market_name, news_ms - 60_000, end, interval="1h")
    if not candles:
        candles = await fetch_candles(hl, market_name, news_ms - 60_000, end, interval="1m")
    if not candles:
        return {}
    idx, base = pick_entry(candles, news_ms)
    if base is None:
        return {}
    res = {"base_px": base}
    for h in hours:
        target = news_ms + h * 3600_000
        px = None
        for c in candles:
            if int(c["T"]) >= target:
                px = float(c["c"])
                break
        if px is not None:
            res[f"+{h}h"] = (px, (px - base) / base * 100)
    return res


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run_backtest(cfg: Config, hl: HLClient, universe: Universe, *, limit: int = 5000,
                       days: float = 7.0, prefilter: bool = True, concurrency: int = 8,
                       cache_path: str = "data/bt_analysis_cache.json",
                       model: str | None = None,
                       items: list[NewsItem] | None = None,
                       regime_context: str | None = None) -> dict:
    pinned = items is not None   # a supplied set (--history-file) must replay identically forever
    if items is None:
        items = await fetch_history(limit)
    if not items:
        return {"error": "no history returned"}
    # Anchor the window to the newest ITEM when the set is pinned, not the wall clock —
    # otherwise re-running the identical command N days later silently drops the oldest
    # N days and the two A/B arms are no longer the same replay.
    anchor_ms = max(i.time_ms for i in items) if pinned else time.time() * 1000
    cutoff = int(anchor_ms - days * 86400 * 1000)
    items = [i for i in items if i.time_ms >= cutoff]
    window = (items[0].time_ms, items[-1].time_ms)

    seen: set = set()
    items = [i for i in items if keep_news(i, cfg, seen)]

    candidates = items
    if prefilter:
        pat = build_patterns(universe)
        candidates = [i for i in items if passes_prefilter(i, pat)]
    log.info("News in window: %d | after filter: %d | candidates to analyze: %d",
             len(seen), len(items), len(candidates))

    analyzer = Analyzer(cfg)
    if regime_context is not None:   # point-in-time regime for the backtest (no lookahead)
        analyzer.regime_context = regime_context
    cache = AnalysisCache(cache_path)
    analyses = await analyze_all(analyzer, universe, candidates, cache, concurrency, model=model)

    r = cfg.app.risk
    blacklist = {b.upper() for b in cfg.app.filters.market_blacklist}
    items_by_id = {i.id: i for i in items}
    trades: list[BTTrade] = []
    rows: list[dict] = []   # per-item review rows (only items with a directional read)
    n_signals = 0
    n_no_data, no_data_drops = 0, []   # gate-passing signals lost to missing candle data
    last_entry_ms: dict[str, int] = {}

    for it in candidates:  # chronological (candidates preserve order)
        a = analyses.get(it.id)
        if not a or a.direction == "none" or not a.ticker:
            continue  # no directional read -> not interesting for review
        row = {
            "id": it.id, "time_ms": it.time_ms, "source": it.source or "Twitter",
            "title": it.title, "body": it.body, "link": it.link, "ticker": a.ticker,
            "direction": a.direction, "confidence": a.confidence,
            "time_sensitivity": a.time_sensitivity, "is_stale": a.is_stale,
            "rationale": a.rationale, "model": a.model,
            "status": "rejected", "reason": "", "passed_gate": False,
            "symbol": None, "market": None, "dex": None,
            "side": None, "entry_px": None, "exit_px": None, "pnl": None, "exit_reason": None,
            "notional": None, "size": None, "leverage": None, "exit_ms": None,
            "stop_px": None, "tp_px": None, "trail_pct": None,
            "mae_pct": None, "mfe_pct": None, "funding_usd": None, "candle_interval": None,
            "pre_move_pct": None,
            "confirm_agree": None, "confirm_confidence": None, "confirm_risk": None,
        }

        def finish(reason: str):
            row["reason"] = reason
            rows.append(row)

        if a.is_stale:
            finish("stale / already priced-in"); continue
        market = universe.resolve(a.ticker, a.asset_class)
        if not market:
            finish(f"{a.ticker} not tradable on enabled dexes"); continue
        row["market"], row["dex"] = market.name, market.dex
        if market.symbol.upper() in blacklist:
            finish(f"{market.symbol} is blacklisted"); continue
        # Deterministic haircuts (illiquid alts, competitive listings).
        mentioned = a.subject_relation == "direct" or universe.mentions(
            market.symbol, it.text, hints=[it.coin_hint or "", *it.symbol_hints])
        eff_conf, notes = adjust_confidence(a.confidence, market, it.text, r, mentioned=mentioned)
        row["confidence"] = eff_conf
        if notes:
            row["rationale"] = f"[{', '.join(notes)}] {row['rationale']}"
        if eff_conf < r.confidence_threshold:
            extra = f" {notes}" if notes else ""
            finish(f"below gate ({eff_conf:.0%} < {r.confidence_threshold:.0%}){extra}"); continue
        n_signals += 1
        row["passed_gate"] = True   # report filter: gate-passers incl. later portfolio rejects
        # per-ticker cooldown (mirror live behavior so one story isn't traded N times)
        prev = last_entry_ms.get(market.symbol)
        if prev and (it.time_ms - prev) < r.per_ticker_cooldown_seconds * 1000:
            finish("ticker cooldown (already in this trade)"); continue
        # Pre-move: always RECORDED for gate-passers (edge_report slices it to validate the
        # guard before enabling); the haircut/reject APPLIES only when thresholds are set.
        pre_move = await measure_pre_move(hl, market.name, it.time_ms, a.direction,
                                          r.pre_move_lookback_seconds)
        if pre_move is not None:
            row["pre_move_pct"] = pre_move
            eff_conf, mg_notes, mg_reject = apply_move_guard(eff_conf, pre_move,
                                                             a.time_sensitivity, r)
            if mg_notes:
                row["confidence"] = eff_conf
                row["rationale"] = f"[{', '.join(mg_notes)}] {row['rationale']}"
            if mg_reject:
                finish(mg_reject); continue
            if eff_conf < r.confidence_threshold:
                finish(f"below gate after move guard ({eff_conf:.0%} < "
                       f"{r.confidence_threshold:.0%})"); continue
        trade = await simulate(hl, market, a.direction, it.time_ms, eff_conf,
                               a.time_sensitivity, r, news_id=it.id)
        if trade:
            trade.pre_move_pct = pre_move
        if not trade:
            n_no_data += 1
            no_data_drops.append((market.name, it.time_ms))
            finish("no backtest candle data at news time (not listed yet or data gap; "
                   "not a live tradability issue)"); continue
        last_entry_ms[market.symbol] = it.time_ms   # set only on a real fill (not on no-data)
        trade.title, trade.link = it.title, it.link
        trades.append(trade)
        row.update(status="traded", reason=trade.reason, symbol=trade.symbol, side=trade.side,
                   entry_px=trade.entry_px, exit_px=trade.exit_px, pnl=trade.pnl,
                   exit_reason=trade.reason, notional=trade.notional, size=trade.size,
                   leverage=trade.leverage, exit_ms=trade.exit_ms, stop_px=trade.stop_px,
                   tp_px=trade.tp_px, trail_pct=trade.trail_pct,
                   mae_pct=trade.mae_pct, mfe_pct=trade.mfe_pct,
                   funding_usd=trade.funding_usd, candle_interval=trade.candle_interval,
                   pre_move_pct=trade.pre_move_pct)
        rows.append(row)

    if n_no_data:
        log.warning("%d gate-passing signal(s) dropped for missing candle data: %s",
                    n_no_data, ", ".join(f"{m}@{utc_day(t)}" for m, t in no_data_drops))
    return {
        "window": window,
        "n_news": len(seen),
        "n_candidates": len(candidates),
        "n_analyzed": len(analyses),
        "n_signals": n_signals,
        "n_no_data": n_no_data,
        "funding_stats": dict(funding_cache().stats),
        "model": model or cfg.app.analyzer.model_fast,
        "threshold": r.confidence_threshold,
        "account_size_usd": r.account_size_usd,
        "usage": analyzer.usage_summary(),
        "exit_config": {
            ts: {
                "hold_h": round(r.exit_horizons.get(ts, r.time_exit_seconds) / 3600, 1),
                "stop_pct": r.stop_loss_by_sensitivity.get(ts, r.stop_loss_pct),
                "trail_pct": r.trail_pct_by_sensitivity.get(ts, 0.0),
                "tp_pct": (0.0 if r.trail_pct_by_sensitivity.get(ts, 0) > 0
                           else r.take_profit_pct),
            }
            for ts in ("immediate", "hours", "days")
        },
        "trades": trades,
        "rows": rows,
        "analyses": analyses,
        "items_by_id": items_by_id,
        "all_items": items,
    }


async def build_regime_timeline(analyzer: Analyzer, cfg: Config, raw_items: list[NewsItem],
                                window_start: int, window_end: int, refresh_seconds: float,
                                lookback_days: float, key: str, model: str,
                                concurrency: int = 6) -> tuple[dict, list]:
    """Regime briefs at every `refresh_seconds` boundary across the window, each built from
    the trailing `lookback_days` of raw news up to that boundary (mirrors the live 12h/14d
    auto-refresh). Briefs are cached to disk so re-runs are free; tokens recorded on the
    analyzer so the report's cost panel includes them. A long window can need hundreds of
    boundaries, so calls are capped at `concurrency` (firing them all at once exhausts the
    HTTP pool -> connect timeouts) and routed through the analyzer's retry wrapper."""
    path = pathlib.Path(f"data/bt_regime_timeline_{key}.json")
    cached: dict[int, str] = {}
    if path.exists():
        try:
            cached = {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
        except Exception:  # noqa: BLE001
            cached = {}
    refresh_ms, lookback_ms = int(refresh_seconds * 1000), int(lookback_days * 86400 * 1000)
    # Anchor boundaries to a FIXED grid (multiples of refresh_ms from the epoch) rather than
    # to window_start, so the same calendar instants recur across days. A re-run on a later
    # day then reuses the cached briefs for the overlapping boundaries (only the new leading
    # edge costs tokens) instead of rebuilding the whole timeline.
    boundaries, b = [], (int(window_start) // refresh_ms) * refresh_ms
    while b <= window_end:
        boundaries.append(b)
        b += refresh_ms
    import bisect as _bisect
    raw_sorted = sorted(raw_items, key=lambda x: x.time_ms)
    times = [it.time_ms for it in raw_sorted]
    sem = asyncio.Semaphore(max(1, concurrency))

    async def build(bnd: int):
        if bnd in cached:
            return bnd, cached[bnd]
        # bisect the window instead of scanning (and formatting) every item per boundary —
        # a 250d replay has ~500 boundaries x tens of thousands of items.
        lo = _bisect.bisect_left(times, bnd - lookback_ms)
        hi = _bisect.bisect_left(times, bnd)
        lines = [f"{it.title} {it.body}".strip()[:200]
                 for it in raw_sorted[max(lo, hi - 1200):hi]]
        if not lines:
            return bnd, ""
        async with sem:
            kw: dict = {"temperature": 0.0} if supports_temperature(model) else {}
            resp = await analyzer._create(
                model=model, max_tokens=700,
                messages=[{"role": "user",
                           "content": REGIME_PROMPT + "\n\nHEADLINES:\n" + "\n".join(lines)}],
                **kw)
        analyzer._record(model, resp)
        brief = "".join(bl.text for bl in resp.content if getattr(bl, "type", "") == "text").strip()
        return bnd, brief

    n_missing = sum(1 for bd in boundaries if bd not in cached)
    results = await asyncio.gather(*(build(bd) for bd in boundaries))
    timeline = dict(results)
    # Only persist when we built something new. When the timeline is already complete (the arena
    # case — all entrants share one pre-built Sonnet regime), this is a no-op, so parallel runs
    # never race on the shared file. The write is atomic regardless — and MERGED with the
    # previously cached boundaries: a shorter/later run must not truncate a shared timeline
    # to just its own window (re-validating an older window would re-pay every brief).
    if n_missing:
        merged = {**{str(k): v for k, v in cached.items()},
                  **{str(k): v for k, v in timeline.items()}}
        _atomic_write_json(path, merged)
    return timeline, sorted(boundaries)


async def run_live_replay(cfg: Config, hl: HLClient, universe: Universe, *,
                          raw_items: list[NewsItem], days: float = 30.0, model: str | None = None,
                          cache_path: str = "data/bt_cache_replay.json",
                          regime_refresh_seconds: float = 43200, regime_lookback_days: float = 14.0,
                          catalyst_memory_days: float = 7.0, regime_key: str = "replay",
                          regime_model: str | None = None,
                          analysis_transform=None, offline_only: bool = False) -> dict:
    """Faithful replay of a live run: walk the window chronologically, judging each headline
    under the regime that was current at its time (progressively rebuilt) AND the rolling
    anti-re-trade catalyst memory built from entries so far. Deterministic + cached."""
    import bisect

    from ..analysis.regime import format_recent_catalysts

    if not raw_items:
        return {"error": "no history returned"}
    raw_items = sorted(raw_items, key=lambda x: x.time_ms)
    # Window anchored to the newest item, not the wall clock: the item set is always pinned
    # here (caller passes it), so the same command must select the same window forever —
    # re-running an A/B days later must not silently drop the oldest days.
    cutoff = int(raw_items[-1].time_ms - days * 86400 * 1000)
    window_raw = [i for i in raw_items if i.time_ms >= cutoff]
    if not window_raw:
        return {"error": "no news in window"}
    window = (window_raw[0].time_ms, window_raw[-1].time_ms)
    seen: set = set()
    candidates = [i for i in window_raw if keep_news(i, cfg, seen)]
    log.info("Replay window: %d raw items | %d after filter", len(window_raw), len(candidates))

    model_full = model or cfg.app.analyzer.model_fast
    analyzer = Analyzer(cfg)   # ledger off (backtest)
    # Regime is shared CONTEXT, not part of the model under test: build/load it under a single
    # key + a fixed regime_model (e.g. sonnet) so every arena entrant sees the IDENTICAL brief
    # (and we don't pay a per-model regime rebuild). Defaults to the analysis model when unset.
    timeline, boundaries = await build_regime_timeline(
        analyzer, cfg, raw_items, window[0], window[1], regime_refresh_seconds,
        regime_lookback_days, key=regime_key, model=(regime_model or model_full))
    log.info("Regime timeline: %d briefs (every %.0fh, %.0fd lookback)",
             len(boundaries), regime_refresh_seconds / 3600, regime_lookback_days)

    def regime_at(t: int) -> str:
        i = bisect.bisect_right(boundaries, t) - 1
        return timeline.get(boundaries[i], "") if i >= 0 else ""

    cache = AnalysisCache(cache_path)
    r = cfg.app.risk
    ca = cfg.app.analyzer
    # Skeptic entry-confirmation A/B: enabled via analyzer.confirm_entries (same flag as
    # live). Verdicts are disk-cached per news id, so re-running an A/B is free.
    confirm_cache = (ConfirmCache(f"data/bt_cache_confirm_{ca.confirm_model}.json")
                     if ca.confirm_entries else None)
    blacklist = {b.upper() for b in cfg.app.filters.market_blacklist}
    mem_ms = int(catalyst_memory_days * 86400 * 1000)
    trades: list[BTTrade] = []
    rows: list[dict] = []
    analyses: dict[str, Analysis] = {}
    entered: list[dict] = []        # rolling catalyst memory (mirrors live note_entry)
    last_entry_ms: dict[str, int] = {}
    recent_signal: dict[tuple, tuple] = {}   # (symbol, direction) -> (last-seen ms, fingerprint)
    halted_days: set[str] = set()   # UTC days whose daily-loss halt latched (mirrors rt.halt_daily)
    open_positions: dict[str, int] = {}    # market name -> exit_ms (for has_open / max-concurrent)
    open_trade: dict[str, BTTrade] = {}    # market name -> currently-open trade (contrary-exit/exposure)
    open_row: dict[str, dict] = {}         # market name -> its review row (mutated on early close)
    n_signals = done = fails = 0
    n_no_data, no_data_drops = 0, []   # gate-passing signals lost to missing candle data
    # Burst aggregation (mirrors live pipeline; analyzer.burst_window_seconds, 0 = off).
    # Combined verdicts cache under a deterministic "<id>+b<n>" key -> A/B re-runs free.
    burst = BurstBuffer(ca.burst_window_seconds)
    n_burst = 0

    def open_for(name: str, t: int) -> Optional[BTTrade]:
        op = open_trade.get(name)
        return op if (op and op.exit_ms > t) else None

    for it in candidates:           # chronological
        analyzer.regime_context = regime_at(it.time_ms)
        recent = [e for e in entered if it.time_ms - e["ts"] < mem_ms][-25:]
        analyzer.recent_catalysts = format_recent_catalysts(recent)
        a = cache.get(it.id)
        if a is None and offline_only:
            # Frozen-cache replay (control_arms): an unanalyzed item never produced a signal —
            # skip it. NEVER call the live API (that would spend money + re-roll the cache).
            continue
        if a is None:
            a = await analyzer.analyze(it, universe, age_seconds=BACKTEST_FRESH_AGE_S, model=model_full)
            if getattr(a, "error", None):
                # Don't cache a FAILED call — caching its error 'none' would skip it on resume,
                # poisoning the window. Leave it uncached (re-run on resume); trip a breaker if
                # failures pile up (out of credits / bad key / outage) so we stop resumably.
                fails += 1
                if fails >= _MAX_CONSECUTIVE_FAILURES:
                    cache.save()
                    raise RuntimeError(
                        f"Aborting replay: {fails} consecutive analysis failures "
                        f"(last: {a.error!r}). Likely out of credits / bad key / API outage. "
                        f"Progress saved to {cache_path}; fix and re-run to resume.")
            else:
                fails = 0
                cache.put(a)
                done += 1
                if done % 25 == 0:
                    log.info("  analyzed %d…", done)
                    cache.save()
        analyses[it.id] = a
        burst_n = 0
        if burst.enabled and a.ticker:
            prior = burst.prior(a.ticker, it.time_ms)
            burst.add(a.ticker, it.time_ms, it.body.strip() or it.title)
            if prior:
                bit = build_burst_item(it, prior)
                ab = cache.get(bit.id)
                if ab is None:
                    ab = await analyzer.analyze(bit, universe,
                                                age_seconds=BACKTEST_FRESH_AGE_S,
                                                model=model_full)
                    if not getattr(ab, "error", None):   # never cache a failed call (resume-safe)
                        cache.put(ab)
                        done += 1
                analyses[bit.id] = ab
                burst_n = len(prior) + 1
                n_burst += 1
                a = ab   # the holistic verdict decides this piece's action
        # Counterfactual hook (default None = no-op; production replays never pass this).
        # control_arms.py uses it to swap the cached verdict's direction for a random one,
        # so the full risk/exit/portfolio machinery runs identically against a NULL strategy.
        # Must return a NEW Analysis (e.g. dataclasses.replace) — the cached object is reused.
        if analysis_transform is not None:
            a = analysis_transform(a)
        if a.direction == "none" or not a.ticker:
            continue
        row = {
            "id": it.id, "time_ms": it.time_ms, "source": it.source or "Twitter",
            "title": it.title, "body": it.body, "link": it.link, "ticker": a.ticker,
            "direction": a.direction, "confidence": a.confidence,
            "time_sensitivity": a.time_sensitivity, "is_stale": a.is_stale,
            "rationale": a.rationale, "model": a.model,
            "status": "rejected", "reason": "", "passed_gate": False,
            "symbol": None, "market": None, "dex": None,
            "side": None, "entry_px": None, "exit_px": None, "pnl": None, "exit_reason": None,
            "notional": None, "size": None, "leverage": None, "exit_ms": None,
            "stop_px": None, "tp_px": None, "trail_pct": None,
            "mae_pct": None, "mfe_pct": None, "funding_usd": None, "candle_interval": None,
            "pre_move_pct": None, "burst_pieces": burst_n or None,
            "confirm_agree": None, "confirm_confidence": None, "confirm_risk": None,
        }

        def finish(reason: str):
            row["reason"] = reason
            rows.append(row)

        if a.is_stale:
            finish("stale / already priced-in"); continue
        market = universe.resolve(a.ticker, a.asset_class)
        if not market:
            finish(f"{a.ticker} not tradable on enabled dexes"); continue
        row["market"], row["dex"] = market.name, market.dex
        if market.symbol.upper() in blacklist:
            finish(f"{market.symbol} is blacklisted"); continue
        # Contrary-exit safeguard (mirror live): contrary news on an OPEN position closes it
        # early — even below the entry gate, on RAW confidence — and we do NOT enter the
        # contrary side. Truncates the open trade's exit at this news time.
        op = open_for(market.name, it.time_ms)
        if (op and not a.is_stale and a.direction in ("long", "short")
                and a.direction != op.side and r.contrary_exit_min_confidence > 0
                and a.confidence >= r.contrary_exit_min_confidence):
            px = await price_at(hl, market.name, it.time_ms)
            if px:
                px = slip_exit(op.side, px, "contrary news exit", r)
                op.exit_px, op.exit_ms = px, it.time_ms
                op.reason = "contrary news exit"
                # Funding re-accrues over the now-truncated hold (the original covered the
                # full simulated hold); rates are disk-cached so this is free.
                op.funding_usd = await funding_cost(hl, market.name, op.side, op.notional,
                                                    op.time_ms, it.time_ms)
                op.pnl = pnl_usd(op.side, op.entry_px, px, op.size) - op.funding_usd
                # Excursions were measured over the FULL simulated hold — re-walk them over
                # the truncated one or MAE/MFE stats overstate this trade.
                await truncate_excursions(hl, op, it.time_ms)
                orow = open_row.pop(market.name, None)
                open_trade.pop(market.name, None)
                open_positions[market.name] = it.time_ms   # slot freed now
                if orow:
                    orow.update(exit_px=op.exit_px, exit_ms=op.exit_ms, pnl=op.pnl,
                                reason=op.reason, exit_reason=op.reason,
                                funding_usd=op.funding_usd, mae_pct=op.mae_pct,
                                mfe_pct=op.mfe_pct)
                finish("contrary news exit (closed open position early)"); continue
            # couldn't price the exit -> fall through; has_open will reject re-entry anyway
        # Duplicate-event suppression (same as live): a repeat of the same (ticker, direction)
        # within the window — even if the earlier one faded — is one event, not a new signal.
        dkey = (market.symbol.upper(), a.direction)
        prev_sig = recent_signal.get(dkey)
        fp = dup_fingerprint(it.text)
        recent_signal[dkey] = (it.time_ms, fp)
        if (r.duplicate_window_seconds and prev_sig
                and (it.time_ms - prev_sig[0]) < r.duplicate_window_seconds * 1000
                and dup_similarity(fp, prev_sig[1]) >= r.duplicate_similarity_min):
            finish("duplicate of a recent signal (same ticker+direction)"); continue
        mentioned = a.subject_relation == "direct" or universe.mentions(
            market.symbol, it.text, hints=[it.coin_hint or "", *it.symbol_hints])
        eff_conf, notes = adjust_confidence(a.confidence, market, it.text, r, mentioned=mentioned)
        row["confidence"] = eff_conf
        if notes:
            row["rationale"] = f"[{', '.join(notes)}] {row['rationale']}"
        if eff_conf < r.confidence_threshold:
            finish(f"below gate ({eff_conf:.0%} < {r.confidence_threshold:.0%})"); continue
        n_signals += 1
        row["passed_gate"] = True   # report filter: gate-passers incl. later portfolio rejects
        # Daily-loss limit (mirror live): once the day's REALIZED losses reach the limit,
        # entries halt for the REST of that UTC day — live latches via rt.halt_daily and
        # auto-resumes next day, so a later winner must not un-halt the replay either.
        # Live sums pnl_usd, which EXCLUDES funding (tracked separately), so funding is
        # added back out of tr.pnl here. (Live additionally counts open drawdown via
        # unrealized_pnl — not reproducible per-headline from bars; known, minor skew.)
        if r.daily_loss_limit_usd:
            day = utc_day(it.time_ms)
            if day not in halted_days:
                realized = sum(tr.pnl + tr.funding_usd for tr in trades
                               if tr.exit_ms <= it.time_ms and utc_day(tr.exit_ms) == day)
                if realized <= -abs(r.daily_loss_limit_usd):
                    halted_days.add(day)
            if day in halted_days:
                finish("daily loss limit hit"); continue
        # Mirror live portfolio caps: one position per market, and a max concurrent count
        # (exposure cap = max_concurrent x max_notional, so the position count binds first).
        if open_positions.get(market.name, 0) > it.time_ms:
            finish("position already open on this market"); continue
        if sum(1 for ex in open_positions.values() if ex > it.time_ms) >= r.max_concurrent_positions:
            finish("max concurrent positions reached"); continue
        prev = last_entry_ms.get(market.symbol)
        if prev and (it.time_ms - prev) < r.per_ticker_cooldown_seconds * 1000:
            finish("ticker cooldown (already in this trade)"); continue
        # Exposure budget (mirror live): clamp notional to remaining room; reject if none.
        exposure = sum(t2.notional for t2 in open_trade.values() if t2.exit_ms > it.time_ms)
        remaining = r.max_total_exposure_usd - exposure
        if remaining < MIN_ORDER_USD:
            finish(f"exposure budget exhausted (room ${remaining:.0f})"); continue
        # Pre-move: always RECORDED for gate-passers (edge_report slices it to validate the
        # guard before enabling); the haircut/reject APPLIES only when thresholds are set.
        pre_move = await measure_pre_move(hl, market.name, it.time_ms, a.direction,
                                          r.pre_move_lookback_seconds)
        if pre_move is not None:
            row["pre_move_pct"] = pre_move
            eff_conf, mg_notes, mg_reject = apply_move_guard(eff_conf, pre_move,
                                                             a.time_sensitivity, r)
            if mg_notes:
                row["confidence"] = eff_conf
                row["rationale"] = f"[{', '.join(mg_notes)}] {row['rationale']}"
            if mg_reject:
                finish(mg_reject); continue
            if eff_conf < r.confidence_threshold:
                finish(f"below gate after move guard ({eff_conf:.0%} < "
                       f"{r.confidence_threshold:.0%})"); continue
        # Skeptic entry confirmation (mirrors live: post-gate, pre-fill). The confirmer
        # sees the point-in-time regime (analyzer.regime_context was set for this item
        # above) plus tape + pre-move reconstructed at the news time. Fails OPEN.
        if confirm_cache is not None:
            verdict = confirm_cache.get(it.id)
            if verdict is None:
                from ..analysis.market_context import build_market_context
                ctx = await build_market_context(hl, market, it.time_ms)
                verdict = await analyzer.confirm(it, a, market_context=ctx,
                                                 pre_move_pct=pre_move,
                                                 age_seconds=BACKTEST_FRESH_AGE_S)
                if verdict is not None:
                    confirm_cache.put(it.id, verdict)
                    confirm_cache.save()
            if verdict is not None:
                row["confirm_agree"] = verdict["agree_direction"]
                row["confirm_confidence"] = verdict["confidence"]
                row["confirm_risk"] = verdict["risk"]
                if not verdict["agree_direction"]:
                    finish(f"confirmation veto: {verdict['risk'][:160]}"); continue
                if ca.confirm_rule != "veto_only":
                    eff_conf = min(eff_conf, verdict["confidence"])
                    row["confidence"] = eff_conf
                    if eff_conf < ca.confirm_gate:
                        finish(f"confirmation confidence {verdict['confidence']:.2f} -> "
                               f"min {eff_conf:.2f} < confirm gate {ca.confirm_gate:.2f}")
                        continue
        trade = await simulate(hl, market, a.direction, it.time_ms, eff_conf,
                               a.time_sensitivity, r, news_id=it.id, notional_cap=remaining)
        if trade:
            trade.pre_move_pct = pre_move
        if not trade:
            n_no_data += 1
            no_data_drops.append((market.name, it.time_ms))
            finish("no backtest candle data at news time (not listed yet or data gap; "
                   "not a live tradability issue)"); continue
        # Register the entry only on a REAL fill (mirrors live note_entry): a signal that
        # couldn't be filled must not set the cooldown or pollute the catalyst memory for
        # later same-ticker signals.
        last_entry_ms[market.symbol] = it.time_ms
        open_positions[market.name] = trade.exit_ms   # occupies a slot until it exits
        open_trade[market.name] = trade
        open_row[market.name] = row
        entered.append({"ts": it.time_ms, "symbol": market.symbol, "side": a.direction,
                        "reason": a.rationale})
        trade.title, trade.link = it.title, it.link
        trades.append(trade)
        row.update(status="traded", reason=trade.reason, symbol=trade.symbol, side=trade.side,
                   entry_px=trade.entry_px, exit_px=trade.exit_px, pnl=trade.pnl,
                   exit_reason=trade.reason, notional=trade.notional, size=trade.size,
                   leverage=trade.leverage, exit_ms=trade.exit_ms, stop_px=trade.stop_px,
                   tp_px=trade.tp_px, trail_pct=trade.trail_pct,
                   mae_pct=trade.mae_pct, mfe_pct=trade.mfe_pct,
                   funding_usd=trade.funding_usd, candle_interval=trade.candle_interval,
                   pre_move_pct=trade.pre_move_pct)
        rows.append(row)
    cache.save()
    if n_no_data:
        log.warning("%d gate-passing signal(s) dropped for missing candle data: %s",
                    n_no_data, ", ".join(f"{m}@{utc_day(t)}" for m, t in no_data_drops))

    return {
        "window": window, "n_news": len(seen), "n_candidates": len(candidates),
        "n_analyzed": len(analyses), "n_signals": n_signals, "n_no_data": n_no_data,
        "funding_stats": dict(funding_cache().stats), "model": model_full,
        "threshold": r.confidence_threshold, "account_size_usd": r.account_size_usd,
        "usage": analyzer.usage_summary(),
        "exit_config": {
            ts: {"hold_h": round(r.exit_horizons.get(ts, r.time_exit_seconds) / 3600, 1),
                 "stop_pct": r.stop_loss_by_sensitivity.get(ts, r.stop_loss_pct),
                 "trail_pct": r.trail_pct_by_sensitivity.get(ts, 0.0),
                 "tp_pct": (0.0 if r.trail_pct_by_sensitivity.get(ts, 0) > 0 else r.take_profit_pct)}
            for ts in ("immediate", "hours", "days")},
        "trades": trades, "rows": rows, "analyses": analyses,
        "items_by_id": {i.id: i for i in candidates}, "all_items": candidates,
        "regime_timeline": {"briefs": len(boundaries),
                            "refresh_h": regime_refresh_seconds / 3600,
                            "lookback_d": regime_lookback_days},
    }
