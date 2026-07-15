"""Arena dashboard: live leaderboard (5 models, equity/return/PnL/open) + a per-news feed
showing every model's verdict side-by-side, streamed over one websocket. Localhost-bound."""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..analysis.pricing import counts_cost_usd
from ..analysis.universe import ALIASES
from ..bus import Event, EventBus
from ..config import Config
from ..models import to_jsonable

log = logging.getLogger("hlbot.arena.ui")

STATIC_DIR = pathlib.Path(__file__).parent / "static"

# Candle intervals the price-chart proxy accepts (Hyperliquid candleSnapshot intervals) ->
# bucket length in seconds, used to derive a default lookback window.
CANDLE_INTERVALS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}
DEFAULT_CANDLE_BARS = 1000         # default history depth per request (HL caps at 5000)
MARKETS_CACHE_SECONDS = 10.0       # metaAndAssetCtxs is heavyweight; the UI polls this freely

# Public-exposure hardening for the candle proxy: every upstream fetch leaves from THIS
# box's IP, which Hyperliquid rate-limits — and anything else trading from this box shares
# that IP. Clamp the span a single URL can request, cap per-viewer request rate (real IP
# comes from Cloudflare's header when tunnelled; every TCP peer is 127.0.0.1 then), and
# bound concurrent upstream fetches across all viewers. The frontend pages up to ~4.5k
# bars per request (zoom-scaled) and stays under all three limits.
MAX_CANDLE_BARS = 5000             # span clamp per request (HL's own per-snapshot cap)
CANDLE_RATE_WINDOW = 60.0          # per-IP sliding window, seconds
CANDLE_RATE_MAX = 40               # requests allowed per IP per window
CANDLE_CONCURRENCY = 4             # simultaneous upstream HL candle fetches (all viewers)
CANDLE_CACHE_MAX = 64              # LRU entries for fully-past (immutable) history pages

SNAPSHOT_FEED_ITEMS = 50           # analyzed news items seeding the feed on page load


def _serialize(event: Event) -> dict:
    return {"topic": event.topic, "ts": event.ts, "payload": to_jsonable(event.payload)}


def create_arena_app(bus: EventBus, store, config: Config, lanes, hl=None) -> FastAPI:
    app = FastAPI(title="hlbot arena")
    # Candle pages are the big payloads (a 4.5k-bar page is ~600KB of JSON, ~10x smaller
    # gzipped) — matters through the tunnel, harmless on localhost.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    entrants = [{"key": l.key, "model": l.model, "gate": l.gate, "starting": l.capital_usd,
                 "live": l.live} for l in lanes]

    async def leaderboard() -> list[dict]:
        # dry_run must be PER LANE (not the global runtime flag): with a mixed live/paper
        # fleet the global flag is False, which would query paper lanes' closed trades with
        # dry_run=0 and show them as zero-history. Mirrors arena_capital_loop exactly.
        rows = []
        for l in lanes:
            try:
                stats = await store.lane_stats(l.model, dry_run=not l.live)
            except Exception:  # noqa: BLE001 - display only
                stats = {"n": 0, "wins": 0, "realized": 0.0}
            n, wins, realized = stats["n"], stats["wins"], stats["realized"]
            upnl = l.open_unrealized()
            halted, halt_why = config.runtime.is_daily_halted(l.model)
            rows.append({"key": l.key, "model": l.model, "gate": l.gate, "live": l.live,
                         "starting": l.capital_usd, "realized": realized, "unrealized": upnl,
                         "current": l.capital_usd + realized + upnl, "open": l.pm.open_count(),
                         "trades": n, "wins": wins,
                         "win_rate": (wins / n) if n else None,
                         "halted": halted, "halt_reason": halt_why})
        rows.sort(key=lambda r: -r["current"])
        return rows

    try:
        _hl_ws = config.base_url.replace("https://", "wss://", 1) + "/ws"
    except Exception:  # noqa: BLE001 - SDK constants unavailable: fall back to mainnet
        _hl_ws = "wss://api.hyperliquid.xyz/ws"

    async def snapshot_payload() -> dict:
        snap = await store.snapshot()   # news / analyses / decisions / positions (all lanes)
        # Feed seed = the first PAGE of the feed's own pagination (news_page: analyzed items
        # only, each with ALL its analysis/decision rows). The raw snapshot tables can't seed
        # the feed: 50 analysis ROWS ≈ only ~10 items (one per model), while the 50 raw news
        # rows (filtered/deduped included) push the client's pagination cursor (feedBefore =
        # oldest seeded news) far past the items whose verdicts were actually delivered —
        # "load more" then pages strictly before the cursor and skips everything in between.
        # `before` is now+24h: item time_ms is source-stamped, so allow modest clock skew.
        page = await store.news_page(int(time.time() * 1000) + 86_400_000,
                                     limit=SNAPSHOT_FEED_ITEMS)
        snap.update(news=page["news"], analyses=page["analyses"], decisions=page["decisions"])
        # Deeper closed-trade seed for the trade-history panel (snapshot() caps at 50 rows
        # of open+closed mixed; the panel wants a real cross-model history).
        snap["closed_positions"] = await store.recent_closed_positions()
        # Authoritative open-position seed (all lanes, uncapped): the client REBUILDS its
        # open-positions map from this on every (re)connect, so a position closed while a
        # viewer was disconnected can't linger as a ghost card. The mixed `positions`
        # table stays for the traded-map seed and older clients.
        snap["open_positions"] = await store.open_positions()
        snap["entrants"] = entrants
        snap["leaderboard"] = await leaderboard()
        snap["started_ms"] = await store.arena_started_ms()   # competition start (header + usage modal)
        snap["chart"] = {"hl_ws": _hl_ws}   # live price chart streams candles from HL's public WS
        snap["runtime"] = {"dry_run": config.runtime.dry_run}
        snap["aliases"] = ALIASES   # company-name -> ticker bank (headline highlighting)
        return snap

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "arena.html")

    @app.get("/api/arena/snapshot")
    async def snapshot() -> JSONResponse:
        return JSONResponse(await snapshot_payload())

    @app.get("/api/arena/detail")
    async def detail(id: str = "") -> JSONResponse:
        """News + per-model analyses/decisions for ONE item. The modal fetches this when a
        clicked trade's news has scrolled out of the snapshot window (old history rows)."""
        if not id or len(id) > 128:   # ids are "tg:chan:msgid"/Tree _ids; public endpoint
            return JSONResponse({"error": "missing id"}, status_code=400)
        return JSONResponse(await store.news_detail(id))

    @app.get("/api/arena/model_trades")
    async def model_trades(model: str = "") -> JSONResponse:
        """Full closed-trade history for ONE lane — backs the per-model modal opened by
        clicking a leaderboard chip. Restricted to entrant models: the endpoint is public
        once tunnelled, so arbitrary strings must not become DB probes."""
        if not any(l.model == model for l in lanes):
            return JSONResponse({"error": "unknown model"}, status_code=404)
        return JSONResponse({"model": model,
                             "closed": await store.model_closed_positions(model)})

    @app.get("/api/arena/usage")
    async def usage_stats() -> JSONResponse:
        """Per-model spend + speed for the usage modal. `analysis` aggregates THIS arena
        DB's news analyses (count / mean latency / summed per-analysis cost); `ledger` is
        the arena's persistent token ledger (every API call since arena start, incl.
        prompt-cache warms and regime briefs; own file since the 2026-07-13 cutover —
        pre-arena production Sonnet usage excluded) with estimated $ cost."""
        analysis = await store.analysis_usage()
        ledger: dict = {}
        try:
            p = pathlib.Path(config.app.token_ledger_file)
            if p.exists():
                ledger = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - ledger is best-effort everywhere else too
            log.warning("usage: could not read token ledger %s", config.app.token_ledger_file)
        for model, entry in ledger.items():
            if isinstance(entry, dict):
                entry["est_cost_usd"] = counts_cost_usd(model, entry)
        return JSONResponse({"analysis": analysis, "ledger": ledger})

    @app.get("/api/arena/news")
    async def news_history(before: int = 0, limit: int = 50) -> JSONResponse:
        """Older analyzed-news pages (news + analyses + decisions) for the feed's
        "load older news" button. `before` = oldest time_ms already loaded client-side."""
        if before <= 0:
            return JSONResponse({"error": "missing before"}, status_code=400)
        return JSONResponse(await store.news_page(before, limit=min(max(limit, 1), 200)))

    _markets_cache: dict = {"ts": 0.0, "data": []}

    async def fetch_markets() -> list[dict]:
        """Every tradable market across the allowed dexes + live context (price, 24h vol, OI,
        funding) — feeds the chart top bar + the searchable market switcher."""
        out: list[dict] = []
        for dex in config.app.filters.allowed_dexes:
            try:
                meta, ctxs = await hl.meta_and_ctxs_raw(dex)
            except Exception:  # noqa: BLE001 - a failed dex just drops out this round
                log.warning("markets: metaAndAssetCtxs failed for dex=%r", dex)
                continue
            for a, ctx in zip(meta.get("universe", []), ctxs):
                if a.get("isDelisted"):
                    continue

                def f(key: str, _ctx=ctx) -> float:
                    try:
                        return float(_ctx.get(key) or 0)
                    except (TypeError, ValueError):
                        return 0.0

                name = a["name"]
                mark = f("markPx")
                out.append({
                    "name": name, "symbol": name.split(":")[-1].upper(), "dex": dex,
                    "asset_class": "crypto" if dex == "" else "equity",
                    "px": f("midPx") or mark, "mark_px": mark, "oracle_px": f("oraclePx"),
                    "prev_day_px": f("prevDayPx"), "day_ntl_vlm": f("dayNtlVlm"),
                    "open_interest_usd": f("openInterest") * mark, "funding": f("funding"),
                    "max_leverage": int(a.get("maxLeverage", 1) or 1),
                })
        out.sort(key=lambda m: -m["day_ntl_vlm"])
        return out

    _markets_lock = asyncio.Lock()

    @app.get("/api/markets")
    async def markets() -> JSONResponse:
        if hl is None:
            return JSONResponse({"ts": 0, "markets": []})
        async with _markets_lock:   # concurrent requests share one fetch, not duplicate it
            now = time.time()
            if now - _markets_cache["ts"] > MARKETS_CACHE_SECONDS:
                # Stamp BEFORE fetching: while HL is failing, retries happen once per TTL,
                # not inline on every request for the duration of the outage.
                _markets_cache["ts"] = now
                data = await fetch_markets()
                if data or not _markets_cache["data"]:   # keep last good data over a failed refresh
                    _markets_cache["data"] = data
        return JSONResponse({"ts": int(_markets_cache["ts"] * 1000),
                             "markets": _markets_cache["data"]})

    _candle_sem = asyncio.Semaphore(CANDLE_CONCURRENCY)
    _candle_hits: dict[str, list[float]] = {}

    def _client_ip(request: Request) -> str:
        # Behind cloudflared every TCP peer is 127.0.0.1; Cloudflare injects the viewer's
        # real IP in this header. Direct localhost use has no header and keys as itself.
        return (request.headers.get("cf-connecting-ip")
                or (request.client.host if request.client else "?"))

    def _candle_rate_ok(ip: str) -> bool:
        now = time.time()
        cutoff = now - CANDLE_RATE_WINDOW
        hits = _candle_hits.setdefault(ip, [])
        while hits and hits[0] < cutoff:
            hits.pop(0)
        if len(hits) >= CANDLE_RATE_MAX:
            return False
        hits.append(now)
        if len(_candle_hits) > 512:   # shed idle IPs so the map can't grow unbounded
            for k in [k for k, v in _candle_hits.items() if not v or v[-1] < cutoff]:
                _candle_hits.pop(k, None)
        return True

    # Fully-past pages are immutable (the tail bar can still change, so only cache pages
    # whose end sits at least one bucket behind now). History pagination requests the same
    # (coin, interval, start, end) again on interval flip-flops and from every public
    # viewer paging back from the same seed — those become HL-round-trip-free.
    _candle_cache: dict[tuple, list] = {}

    @app.get("/api/candles")
    async def candles(request: Request, coin: str, interval: str = "5m",
                      start: int = 0, end: int = 0) -> JSONResponse:
        """Proxy Hyperliquid candle history for the price chart (live tail streams client-side
        straight from HL's public WS). Read-only; span/rate/concurrency-limited (see the
        hardening constants above)."""
        if hl is None:
            return JSONResponse({"error": "no market client"}, status_code=503)
        if interval not in CANDLE_INTERVALS:
            return JSONResponse({"error": f"bad interval {interval!r}"}, status_code=400)
        if not coin or len(coin) > 32:
            return JSONResponse({"error": "bad coin"}, status_code=400)
        if not _candle_rate_ok(_client_ip(request)):
            return JSONResponse({"error": "rate limited"}, status_code=429)
        ival_ms = CANDLE_INTERVALS[interval] * 1000
        end_ms = end or int(time.time() * 1000)
        start_ms = start or end_ms - ival_ms * DEFAULT_CANDLE_BARS
        start_ms = max(start_ms, end_ms - ival_ms * MAX_CANDLE_BARS)
        immutable = end_ms < time.time() * 1000 - ival_ms
        key = (coin, interval, start_ms, end_ms)
        if immutable and key in _candle_cache:
            return JSONResponse(_candle_cache[key])
        try:
            async with _candle_sem:
                data = await hl.candles(coin, interval, start_ms, end_ms)
        except Exception:  # noqa: BLE001
            log.exception("candle fetch failed for %s %s", coin, interval)
            return JSONResponse({"error": "candle fetch failed"}, status_code=502)
        if immutable and data:
            while len(_candle_cache) >= CANDLE_CACHE_MAX:   # dicts iterate in insert order
                _candle_cache.pop(next(iter(_candle_cache)))
            _candle_cache[key] = data
        return JSONResponse(data)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        q = bus.subscribe()
        try:
            await websocket.send_json({"topic": "snapshot", "ts": 0,
                                       "payload": await snapshot_payload()})
            while True:
                await websocket.send_json(_serialize(await q.get()))
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("Arena UI websocket error")
        finally:
            bus.unsubscribe(q)

    return app
