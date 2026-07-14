import asyncio
import tempfile

import httpx

from hlbot.bus import EventBus
from hlbot.config import Config
from hlbot.store.db import Store
from hlbot.ui.server import create_app


class FakeHL:
    """Stand-in for HLClient's market-data surface: two dexes of meta+ctx, one candle batch."""

    async def meta_and_ctxs_raw(self, dex: str = ""):
        if dex == "":
            meta = {"universe": [
                {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
                {"name": "HYPE", "szDecimals": 2, "maxLeverage": 5},
                {"name": "GONE", "szDecimals": 0, "maxLeverage": 3, "isDelisted": True},
            ]}
            ctxs = [
                {"markPx": "61300", "midPx": "61301", "oraclePx": "61299", "prevDayPx": "62000",
                 "dayNtlVlm": "2.9e9", "openInterest": "32000", "funding": "-0.0000031"},
                {"markPx": "56.2", "midPx": "56.25", "oraclePx": "56.21", "prevDayPx": "62.4",
                 "dayNtlVlm": "8.0e8", "openInterest": "20000000", "funding": "0.0000125"},
                {"markPx": "1", "midPx": "1", "oraclePx": "1", "prevDayPx": "1",
                 "dayNtlVlm": "999e9", "openInterest": "0", "funding": "0"},
            ]
            return [meta, ctxs]
        meta = {"universe": [{"name": f"{dex}:MRVL", "szDecimals": 2, "maxLeverage": 10}]}
        ctxs = [{"markPx": "90.1", "midPx": "90.2", "oraclePx": "90.0", "prevDayPx": "88.0",
                 "dayNtlVlm": "5.0e7", "openInterest": "10000", "funding": "0.0000010"}]
        return [meta, ctxs]

    async def candles(self, coin, interval, start_ms, end_ms):
        assert start_ms < end_ms
        return [{"t": 0, "T": 60000, "s": coin, "i": interval,
                 "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "n": 3}]


def _store():
    return Store(path=tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name)


async def _exercise():
    cfg = Config()
    cfg.runtime.dry_run = True
    store = await _store().init()
    app = create_app(EventBus(), store, cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        snap = await ac.get("/api/snapshot")
        assert snap.status_code == 200
        assert snap.json()["runtime"]["dry_run"] is True
        # chart config (HL websocket endpoint) rides along on every snapshot
        assert snap.json()["chart"]["hl_ws"].startswith("wss://")

        toggled = await ac.post("/api/control", json={"action": "toggle_dry_run"})
        assert toggled.json()["dry_run"] is False

        halted = await ac.post("/api/control", json={"action": "halt"})
        assert halted.json()["trading_halted"] is True

        index = await ac.get("/")
        assert index.status_code == 200 and "hlbot" in index.text
        assert 'id="chart"' in index.text

        # without a trading client attached the chart endpoints degrade gracefully
        assert (await ac.get("/api/markets")).json()["markets"] == []
        assert (await ac.get("/api/candles", params={"coin": "BTC"})).status_code == 503
        assert (await ac.get("/api/coinmeta")).status_code == 400
    await store.close()


def test_ui_endpoints():
    asyncio.run(_exercise())


class FakeFeed:
    def __init__(self, enabled, connected):
        self._st = {"enabled": enabled, "connected": connected}

    def status(self):
        return self._st


async def _exercise_health():
    cfg = Config()
    store = await _store().init()
    # no sources attached -> everything reports disabled (never 500s)
    app = create_app(EventBus(), store, cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        h = (await ac.get("/api/health")).json()
        assert h["telegram"] == {"enabled": False, "connected": False}
        assert h["tree"] == {"enabled": False, "connected": False}

    app = create_app(EventBus(), store, cfg,
                     tg_source=FakeFeed(True, True), tree=FakeFeed(True, False))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        h = (await ac.get("/api/health")).json()
        assert h["telegram"] == {"enabled": True, "connected": True}
        assert h["tree"] == {"enabled": True, "connected": False}
    await store.close()


def test_health_endpoint():
    asyncio.run(_exercise_health())


async def _exercise_chart_api():
    cfg = Config()
    cfg.runtime.dry_run = True
    store = await _store().init()
    app = create_app(EventBus(), store, cfg, hl=FakeHL())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        mk = await ac.get("/api/markets")
        assert mk.status_code == 200
        ms = mk.json()["markets"]
        names = [m["name"] for m in ms]
        assert "BTC" in names and "xyz:MRVL" in names
        assert "GONE" not in names                      # delisted markets are dropped
        vols = [m["day_ntl_vlm"] for m in ms]
        assert vols == sorted(vols, reverse=True)       # modal lists top volume first
        mrvl = next(m for m in ms if m["name"] == "xyz:MRVL")
        assert (mrvl["symbol"], mrvl["dex"], mrvl["asset_class"]) == ("MRVL", "xyz", "equity")
        btc = next(m for m in ms if m["name"] == "BTC")
        assert btc["open_interest_usd"] == 32000 * 61300   # OI is base units * mark

        c = await ac.get("/api/candles", params={"coin": "xyz:MRVL", "interval": "1m"})
        assert c.status_code == 200
        assert c.json()[0]["s"] == "xyz:MRVL" and c.json()[0]["i"] == "1m"
        bad = await ac.get("/api/candles", params={"coin": "BTC", "interval": "7m"})
        assert bad.status_code == 400

        js = await ac.get("/static/lightweight-charts.js")
        assert js.status_code == 200 and "LightweightCharts" in js.text
    await store.close()


def test_chart_api():
    asyncio.run(_exercise_chart_api())


class FakePM:
    """position_by_id/force_close surface used by the manual-close endpoint."""

    def __init__(self, pos=None, close_ok=True):
        self.pos = pos
        self.close_ok = close_ok
        self.closed = []

    def position_by_id(self, pid):
        return self.pos if (self.pos and self.pos.id == pid) else None

    async def force_close(self, pos, reason):
        self.closed.append((pos.id, reason))
        if self.close_ok:
            pos.pnl_usd = 5.0
            self.pos = None
            return True
        return False


def _pos(pid="deadbeef0001"):
    from hlbot.models import Position, now_ms
    t = now_ms()
    return Position(id=pid, news_id="n", market="BTC", symbol="BTC", dex="", side="long",
                    size=0.1, entry_px=100.0, stop_loss=97.0, take_profit=0.0, leverage=5,
                    notional_usd=10.0, opened_ms=t, time_exit_ms=t + 1000, dry_run=True)


async def _exercise_close_position():
    cfg = Config()
    store = await _store().init()

    # no position manager attached -> 503, never a crash
    app = create_app(EventBus(), store, cfg)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/api/control", json={"action": "close_position", "id": "x"})
        assert r.status_code == 503

    pm = FakePM(pos=_pos())
    app = create_app(EventBus(), store, cfg, pm=pm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/api/control", json={"action": "close_position", "id": "nope"})
        assert r.status_code == 404 and pm.closed == []

        r = await ac.post("/api/control", json={"action": "close_position",
                                                "id": "deadbeef0001"})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert pm.closed == [("deadbeef0001", "manual close (UI)")]

    pm = FakePM(pos=_pos(), close_ok=False)   # live close didn't fill -> 502 + still open
    app = create_app(EventBus(), store, cfg, pm=pm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        r = await ac.post("/api/control", json={"action": "close_position",
                                                "id": "deadbeef0001"})
        assert r.status_code == 502 and "still" in r.json()["error"]
        assert pm.pos is not None
    await store.close()


def test_close_position_endpoint():
    asyncio.run(_exercise_close_position())
