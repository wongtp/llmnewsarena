"""FastAPI dashboard: live event feed over websocket + control endpoints
(dry-run toggle / kill switch). Localhost-bound by default."""
from __future__ import annotations

import asyncio
import logging
import pathlib
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..bus import Event, EventBus
from ..config import Config
from ..models import to_jsonable

log = logging.getLogger("hlbot.ui")

STATIC_DIR = pathlib.Path(__file__).parent / "static"

# Candle intervals the chart proxy accepts (Hyperliquid candleSnapshot intervals) with the
# bucket length in seconds, used to derive a default lookback window.
CANDLE_INTERVALS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800, "1M": 2592000,
}
DEFAULT_CANDLE_BARS = 500          # default history depth per request (HL caps at 5000)
MARKETS_CACHE_SECONDS = 10.0       # metaAndAssetCtxs is heavyweight; the UI polls this freely
COINMETA_OK_TTL = 600.0            # CoinGecko MC/FDV cache on success...
COINMETA_ERR_TTL = 90.0            # ...and on failure (retry sooner, but never hammer)


def _serialize(event: Event) -> dict:
    return {"topic": event.topic, "ts": event.ts, "payload": to_jsonable(event.payload)}


def create_app(bus: EventBus, store, config: Config, capital=None, hl=None,
               tg_source=None, tree=None, pm=None) -> FastAPI:
    app = FastAPI(title="hlbot dashboard")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def runtime_state() -> dict:
        rt = config.runtime
        return {"dry_run": rt.dry_run, "trading_halted": rt.trading_halted,
                "halt_reason": rt.halt_reason}

    def capital_state() -> dict:
        return capital.state() if capital else {"starting": None, "current": None}

    # The chart streams live candles straight from Hyperliquid's public websocket (no auth;
    # the backend only proxies history/meta). Derive the WS endpoint from the configured
    # network so testnet setups chart testnet data.
    try:
        _hl_ws = config.base_url.replace("https://", "wss://", 1) + "/ws"
    except Exception:  # noqa: BLE001 - SDK constants unavailable: fall back to mainnet
        _hl_ws = "wss://api.hyperliquid.xyz/ws"
    chart_cfg = {"hl_ws": _hl_ws}

    async def snapshot_payload() -> dict:
        snap = await store.snapshot()
        snap["runtime"] = runtime_state()
        snap["capital"] = capital_state()
        snap["chart"] = chart_cfg
        return snap

    _LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
    _FWD_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")

    def authorized(req: Request) -> bool:
        """Guard the mutating control endpoint. If an auth_token is configured, require it
        (header or query). Otherwise allow only DIRECT localhost clients: a request relayed
        by a reverse proxy on this box also shows client.host=127.0.0.1, so any forwarding
        header disqualifies localhost trust — set ui.auth_token if you proxy the dashboard."""
        tok = config.app.ui.auth_token
        if tok:
            return req.headers.get("x-auth-token") == tok or req.query_params.get("token") == tok
        if any(h in req.headers for h in _FWD_HEADERS):
            return False
        host = (req.client.host if req.client else "") or ""
        return host in _LOCAL_HOSTS

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/snapshot")
    async def snapshot() -> JSONResponse:
        return JSONResponse(await snapshot_payload())

    @app.get("/api/health")
    async def health() -> JSONResponse:
        """Feed-link health for the header indicators (Telegram MTProto + Tree WS)."""
        def feed_state(src) -> dict:
            try:
                return src.status() if src else {"enabled": False, "connected": False}
            except Exception:  # noqa: BLE001 - display-only
                return {"enabled": False, "connected": False}
        return JSONResponse({"ts": int(time.time() * 1000),
                             "telegram": feed_state(tg_source),
                             "tree": feed_state(tree)})

    _markets_cache: dict = {"ts": 0.0, "data": []}

    async def fetch_markets() -> list[dict]:
        """Every tradable market across the allowed dexes with its live context (price,
        24h volume, OI, funding) — feeds the chart top bar and the market-switcher modal."""
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

                def f(key: str, _ctx=ctx) -> float:   # bind ctx per iteration (late-binding trap)
                    try:
                        return float(_ctx.get(key) or 0)
                    except (TypeError, ValueError):
                        return 0.0

                name = a["name"]
                mark = f("markPx")
                out.append({
                    "name": name,
                    "symbol": name.split(":")[-1].upper(),
                    "dex": dex,
                    "asset_class": "crypto" if dex == "" else "equity",
                    "px": f("midPx") or mark,
                    "mark_px": mark,
                    "oracle_px": f("oraclePx"),
                    "prev_day_px": f("prevDayPx"),
                    "day_ntl_vlm": f("dayNtlVlm"),
                    "open_interest_usd": f("openInterest") * mark,
                    "funding": f("funding"),
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

    @app.get("/api/candles")
    async def candles(coin: str, interval: str = "5m", start: int = 0, end: int = 0) -> JSONResponse:
        """History proxy for the chart (the live tail streams over HL's public WS)."""
        if hl is None:
            return JSONResponse({"error": "trading client not attached"}, status_code=503)
        if interval not in CANDLE_INTERVALS:
            return JSONResponse({"error": f"bad interval {interval!r}"}, status_code=400)
        now_ms = int(time.time() * 1000)
        end_ms = end or now_ms
        start_ms = start or end_ms - CANDLE_INTERVALS[interval] * 1000 * DEFAULT_CANDLE_BARS
        try:
            return JSONResponse(await hl.candles(coin, interval, start_ms, end_ms))
        except Exception:  # noqa: BLE001
            log.exception("candle fetch failed for %s %s", coin, interval)
            return JSONResponse({"error": "candle fetch failed"}, status_code=502)

    _coinmeta_cache: dict[str, tuple[float, dict]] = {}

    @app.get("/api/coinmeta")
    async def coinmeta(symbol: str = "") -> JSONResponse:
        """Best-effort market cap / FDV via CoinGecko's public API (HL's perp API has no
        supply data). Cached; any failure just returns nulls and the UI shows an em dash."""
        sym = symbol.strip()
        if not sym:
            return JSONResponse({"error": "symbol required"}, status_code=400)
        # HL's kPEPE/kBONK-style names are 1000x-unit perps of the underlying token; MC/FDV
        # belong to the underlying (lowercase-k prefix only, so KAVA stays KAVA).
        base = (sym[1:] if len(sym) > 2 and sym[0] == "k" and sym[1:].isupper() else sym).upper()
        now = time.time()
        hit = _coinmeta_cache.get(base)
        if hit and now - hit[0] < (COINMETA_OK_TTL if hit[1].get("ok") else COINMETA_ERR_TTL):
            return JSONResponse(hit[1])
        data = {"ok": False, "symbol": base, "market_cap": None, "fdv": None, "image": None}
        try:
            import httpx
            async with httpx.AsyncClient(timeout=6.0) as client:
                r = await client.get("https://api.coingecko.com/api/v3/search",
                                     params={"query": base})
                r.raise_for_status()
                coins = (r.json() or {}).get("coins") or []
                # Search results are ranked by market cap, so the first exact symbol match
                # is the major asset (BTC -> bitcoin, not a namesake fork).
                cid = next((c.get("id") for c in coins
                            if (c.get("symbol") or "").upper() == base), None)
                if cid:
                    r = await client.get("https://api.coingecko.com/api/v3/coins/markets",
                                         params={"vs_currency": "usd", "ids": cid})
                    r.raise_for_status()
                    rows = r.json() or []
                    if rows:
                        data = {"ok": True, "symbol": base, "id": cid,
                                "market_cap": rows[0].get("market_cap"),
                                "fdv": rows[0].get("fully_diluted_valuation"),
                                "image": rows[0].get("image")}   # icon fallback for the pair button
        except Exception:  # noqa: BLE001 - display-only nicety; never surface an error
            log.debug("coinmeta lookup failed for %s", base)
        if len(_coinmeta_cache) >= 512:   # keyed by client-supplied ?symbol= — bound it
            _coinmeta_cache.pop(next(iter(_coinmeta_cache)))
        _coinmeta_cache[base] = (now, data)
        return JSONResponse(data)

    @app.post("/api/control")
    async def control(req: Request) -> JSONResponse:
        if not authorized(req):
            return JSONResponse({"error": "unauthorized"}, status_code=403)
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001 - malformed body is a client error, not a 500
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        action = body.get("action")
        rt = config.runtime
        if action == "toggle_dry_run":
            rt.set_dry_run(not rt.dry_run)
            await bus.publish("status", f"Mode -> {'DRY-RUN' if rt.dry_run else 'LIVE'}")
        elif action == "halt":
            rt.halt("manual kill switch")
            await bus.publish("status", "Trading HALTED (manual)")
        elif action == "resume":
            rt.resume()
            await bus.publish("status", "Trading resumed")
        elif action == "reset_baseline" and capital:
            capital.reset()
            await bus.publish("capital", capital.state())
            await bus.publish("status", "Capital baseline reset to current equity")
        elif action == "close_position":
            # Manual emergency close. Keyed by position id (never index/symbol) and
            # confirmed client-side; force_close fails SAFE — an unfilled LIVE close
            # keeps the position tracked and the monitor retrying.
            if pm is None:
                return JSONResponse({"ok": False, "error": "position manager not attached"},
                                    status_code=503)
            pos = pm.position_by_id(str(body.get("id") or ""))
            if pos is None:
                return JSONResponse({"ok": False, "error": "position not found "
                                     "(already closed?)"}, status_code=404)
            log.warning("Manual close requested via UI for %s (%s)", pos.id, pos.market)
            if await pm.force_close(pos, "manual close (UI)"):
                return JSONResponse({"ok": True, "pnl_usd": pos.pnl_usd})
            return JSONResponse({"ok": False, "error": "close did not fill — position still "
                                 "open, the bot keeps managing it"}, status_code=502)
        return JSONResponse(runtime_state())

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        q = bus.subscribe()  # all topics
        try:
            await websocket.send_json({"topic": "snapshot", "ts": 0,
                                       "payload": await snapshot_payload()})
            while True:
                event = await q.get()
                await websocket.send_json(_serialize(event))
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            log.exception("UI websocket error")
        finally:
            bus.unsubscribe(q)

    return app
