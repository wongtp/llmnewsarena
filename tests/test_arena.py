"""Arena-stack regression tests: per-lane paper/live isolation, model_id-partitioned
restore, pre-arena DB migration, the leaderboard's per-lane dry_run accounting, and the
trail high-water-mark restart round-trip. These pin the exact failure modes that would
send paper lanes live (or lose live positions) at go-live."""
import asyncio
import tempfile
import time
import types

from fastapi.testclient import TestClient

from hlbot.arena.lane import TradingLane
from hlbot.bus import EventBus
from hlbot.config import Config
from hlbot.models import Analysis, Decision, Market, NewsItem, Position, now_ms
from hlbot.store.db import Store
from hlbot.trading.executor import Executor
from hlbot.trading.hl_client import HLClient
from hlbot.trading.position_manager import PositionManager
from hlbot.ui.arena_server import create_arena_app


def _tmp():
    return tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name


def _market():
    return Market("xyz:MRVL", "MRVL", "xyz", "equity", 2, 5)


def _decision(side="long", entry=100.0):
    return Decision(news_id="n", action="enter", reason="r", market=_market(), side=side,
                    notional_usd=1000.0, size=10.0, leverage=5, entry_px=entry,
                    stop_loss=entry * 0.97, take_profit=0.0, time_exit_seconds=3600,
                    confidence=0.9)


def _item():
    return NewsItem("n", "t", "b", "X", "http://l", now_ms(), now_ms())


def _analysis(model="claude-sonnet-4-6"):
    return Analysis("n", "MRVL", "equity", "long", 0.9, "hours", False, "r", model)


class FakeLiveHL:
    """Minimal live-order surface: canned fill responses + call log."""
    def __init__(self, fill_sz=10.0, px=100.0):
        self.calls = []
        self.fill_sz, self.px = fill_sz, px

    async def set_leverage(self, market, lev):
        self.calls.append(("lev", lev))
        return {}

    async def market_open(self, market, is_buy, size, slippage):
        self.calls.append(("open", size))
        return {"status": "ok", "response": {"data": {"statuses": [
            {"filled": {"avgPx": str(self.px), "totalSz": str(size)}}]}}}

    async def market_close(self, market, slippage):
        self.calls.append(("close",))
        return {"status": "ok", "response": {"data": {"statuses": [
            {"filled": {"avgPx": str(self.px), "totalSz": str(self.fill_sz)}}]}}}

    async def place_stop(self, market, is_buy_to_close, size, trigger_px, slippage):
        self.calls.append(("stop", trigger_px))
        return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 7}}]}}}

    async def cancel_order(self, market, oid):
        self.calls.append(("cancel", oid))
        return {}

    async def mid(self, market):
        return self.px


def _lane(cfg, store, hl, *, live, model="claude-sonnet-4-6"):
    return TradingLane(key="t", model=model, gate=0.80, capital_usd=10_000.0, config=cfg,
                       hl=hl, store=store, bus=EventBus(), universe=None, live=live)


# ---- per-lane paper/live isolation (the go-live safety property) ---------------
def test_paper_lane_stays_paper_when_global_flag_is_live():
    async def run():
        cfg = Config()
        cfg.runtime.dry_run = False        # one live lane flips the global flag at go-live
        store = await Store(path=_tmp()).init()
        lane = _lane(cfg, store, FakeLiveHL(), live=False)
        pos = await lane.executor.open(_decision(), _item(), _analysis())
        await store.close()
        return pos, lane

    pos, lane = asyncio.run(run())
    assert pos is not None and pos.dry_run is True     # paper despite global dry_run=False
    assert lane.hl.calls == []                          # NO real order left this lane
    assert lane.risk.dry_run_override is True


def test_live_lane_opens_live_despite_global_dry_run():
    async def run():
        cfg = Config()
        cfg.runtime.dry_run = True         # runtime toggled dry must not strand a live lane
        store = await Store(path=_tmp()).init()
        hl = FakeLiveHL()
        lane = _lane(cfg, store, hl, live=True)
        pos = await lane.executor.open(_decision(), _item(), _analysis())
        await store.close()
        return pos, hl

    pos, hl = asyncio.run(run())
    assert pos is not None and pos.dry_run is False
    assert ("open", 10.0) in hl.calls                  # real order path taken
    assert pos.stop_order_id == 7                      # exchange backstop stop rested


# ---- model_id partition: restore only YOUR lane's positions + cooldowns --------
def test_lane_restore_filters_by_model_id():
    async def run():
        cfg = Config()
        store = await Store(path=_tmp()).init()
        now = int(time.time() * 1000)
        for pid, model in (("a1", "model-A"), ("b1", "model-B")):
            await store.upsert_position(Position(
                id=pid, news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz", side="long",
                size=1.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0, leverage=5,
                notional_usd=100.0, opened_ms=now, time_exit_ms=now + 10**9, dry_run=True,
                model_id=model))
        lane = _lane(cfg, store, FakeLiveHL(), live=False, model="model-A")
        await lane.restore()
        cooldowns = dict(lane.risk._last_entry_ms)
        # lane B restores independently and must not inherit A's cooldown either
        lane_b = _lane(cfg, store, FakeLiveHL(), live=False, model="model-B")
        await lane_b.restore()
        await store.close()
        return lane, cooldowns, lane_b

    lane, cooldowns, lane_b = asyncio.run(run())
    assert set(lane.pm._open) == {"a1"}                # only its own open position
    assert "MRVL" in cooldowns                          # its own entry cools it down
    assert set(lane_b.pm._open) == {"b1"}


def test_store_migration_from_pre_arena_db():
    # A DB created before model_id existed must open cleanly (the restart path) —
    # the ALTER migration has to run BEFORE the schema's model_id index is created.
    import sqlite3
    path = _tmp()
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE positions (
        id TEXT PRIMARY KEY, news_id TEXT, opened_ms INTEGER, closed_ms INTEGER,
        symbol TEXT, dex TEXT, side TEXT, size REAL, entry_px REAL, exit_px REAL,
        stop_loss REAL, take_profit REAL, leverage INTEGER, notional REAL,
        status TEXT, pnl REAL, exit_reason TEXT, dry_run INTEGER, json TEXT)""")
    con.commit()
    con.close()

    async def run():
        store = await Store(path).init()   # must not raise
        stats = await store.lane_stats("m", dry_run=True)
        await store.close()
        return stats

    assert asyncio.run(run()) == {"n": 0, "wins": 0, "realized": 0.0}


# ---- leaderboard accounting: per-lane dry_run + SQL aggregate ------------------
def _closed_pos(pid, model, pnl, dry):
    now = int(time.time() * 1000)
    return Position(id=pid, news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                    side="long", size=1.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0,
                    leverage=5, notional_usd=100.0, opened_ms=now - 1000, time_exit_ms=now,
                    dry_run=dry, model_id=model, status="closed", pnl_usd=pnl, closed_ms=now)


def test_lane_stats_aggregate_partitions_by_mode():
    async def run():
        store = await Store(path=_tmp()).init()
        await store.upsert_position(_closed_pos("p1", "m", +50.0, dry=True))
        await store.upsert_position(_closed_pos("p2", "m", -20.0, dry=True))
        await store.upsert_position(_closed_pos("p3", "m", +999.0, dry=False))  # live row
        paper = await store.lane_stats("m", dry_run=True)
        live = await store.lane_stats("m", dry_run=False)
        await store.close()
        return paper, live

    paper, live = asyncio.run(run())
    assert paper == {"n": 2, "wins": 1, "realized": 30.0}
    assert live == {"n": 1, "wins": 1, "realized": 999.0}


def test_arena_leaderboard_uses_per_lane_mode_not_global_flag():
    class _FakePM:
        def open_count(self):
            return 0

    class _FakeLane:
        def __init__(self, key, model, live):
            self.key, self.model, self.gate, self.live = key, model, 0.8, live
            self.capital_usd = 10_000.0
            self.pm = _FakePM()

        def open_unrealized(self):
            return 0.0

    async def seed(store):
        await store.upsert_position(_closed_pos("p1", "paper-model", +100.0, dry=True))
        await store.upsert_position(_closed_pos("p2", "live-model", +7.0, dry=False))

    cfg = Config()
    cfg.runtime.dry_run = False   # mixed fleet: global flag is False the moment one lane is live
    store = Store(path=_tmp())
    asyncio.run(store.init())
    asyncio.run(seed(store))
    lanes = [_FakeLane("paper", "paper-model", live=False),
             _FakeLane("live", "live-model", live=True)]
    app = create_arena_app(EventBus(), store, cfg, lanes)
    client = TestClient(app)
    snap = client.get("/api/arena/snapshot").json()
    rows = {r["key"]: r for r in snap["leaderboard"]}
    # The paper lane's PAPER history must survive the global flag being False.
    assert rows["paper"]["realized"] == 100.0 and rows["paper"]["trades"] == 1
    assert rows["live"]["realized"] == 7.0
    # Trade-history seed: all closed trades across lanes, newest first, full row payload.
    closed = snap["closed_positions"]
    assert {c["id"] for c in closed} == {"p1", "p2"}
    assert closed[0]["model_id"] and closed[0]["pnl_usd"] is not None
    # Headline-highlighting bank: the name->ticker aliases ride along in the snapshot.
    assert snap["aliases"].get("spacex") == "SPCX"


def test_arena_detail_endpoint_serves_old_items():
    # The modal fetches /api/arena/detail for trades whose news scrolled out of the
    # snapshot window — it must return the news row + every model's analysis/decision.
    from hlbot.models import Analysis, Decision, NewsItem, now_ms

    class _FakeLane2:
        key, model, gate, live, capital_usd = "x", "m", 0.8, False, 1.0

        class pm:
            @staticmethod
            def open_count():
                return 0

        @staticmethod
        def open_unrealized():
            return 0.0

    cfg = Config()
    store = Store(path=_tmp())

    async def seed():
        await store.init()
        await store.save_news(NewsItem(id="old1", title="ACME beats", body="big beat",
                                       source="s", link=None, time_ms=now_ms() - 10**9,
                                       received_ms=now_ms() - 10**9))
        await store.save_analysis(Analysis("old1", "ACME", "equity", "long", 0.9, "days",
                                           False, "strong catalyst", "model-A"))
        await store.save_decision(Decision(news_id="old1", action="enter", reason="r",
                                           market=_market(), side="long", notional_usd=5000.0,
                                           size=1.0, leverage=5, entry_px=100.0,
                                           stop_loss=97.0, take_profit=0.0,
                                           time_exit_seconds=3600, confidence=0.9))

    asyncio.run(seed())
    app = create_arena_app(EventBus(), store, cfg, [_FakeLane2()])
    client = TestClient(app)
    d = client.get("/api/arena/detail", params={"id": "old1"}).json()
    assert d["news"]["title"] == "ACME beats" and d["news"]["time_ms"]
    assert d["analyses"][0]["model"] == "model-A" and d["analyses"][0]["direction"] == "long"
    assert d["decisions"][0]["action"] == "enter" and d["decisions"][0]["notional_usd"] == 5000.0
    assert client.get("/api/arena/detail").status_code == 400   # missing id
    empty = client.get("/api/arena/detail", params={"id": "nope"}).json()
    assert empty == {"news": None, "analyses": [], "decisions": []}


def test_arena_news_pagination_pages_older_analyzed_items():
    # /api/arena/news drives the feed's "load older news" button: strictly-older analyzed
    # items, newest first; un-analyzed rows (filtered/deduped) must not consume page slots.
    from hlbot.models import Analysis, NewsItem, now_ms

    class _FakeLane3:
        key, model, gate, live, capital_usd = "x", "m", 0.8, False, 1.0

        class pm:
            @staticmethod
            def open_count():
                return 0

        @staticmethod
        def open_unrealized():
            return 0.0

    cfg = Config()
    store = Store(path=_tmp())
    t0 = now_ms() - 10**9

    async def seed():
        await store.init()
        for i in range(1, 6):
            await store.save_news(NewsItem(id=f"n{i}", title=f"headline {i}", body="",
                                           source="s", link=None, time_ms=t0 + i * 1000,
                                           received_ms=t0 + i * 1000))
            if i != 4:   # n4 was filtered before analysis — must be skipped by the pager
                await store.save_analysis(Analysis(f"n{i}", "ACME", "equity", "long", 0.9,
                                                   "days", False, "r", "model-A"))

    asyncio.run(seed())
    app = create_arena_app(EventBus(), store, cfg, [_FakeLane3()])
    client = TestClient(app)
    page = client.get("/api/arena/news", params={"before": t0 + 5000, "limit": 2}).json()
    assert [n["id"] for n in page["news"]] == ["n3", "n2"]   # newest first, n4 skipped
    assert {a["news_id"] for a in page["analyses"]} == {"n3", "n2"}
    page2 = client.get("/api/arena/news", params={"before": t0 + 2000}).json()
    assert [n["id"] for n in page2["news"]] == ["n1"]
    assert client.get("/api/arena/news", params={"before": t0 + 1000}).json()["news"] == []
    assert client.get("/api/arena/news").status_code == 400   # missing before


def test_arena_snapshot_feed_seed_is_contiguous_first_page():
    # The snapshot must seed the feed with the first PAGE of the same pagination the
    # "load more" button uses: analyzed items only, newest first, with EVERY model's
    # analysis row. Seeding from the raw tables (50 analysis rows ≈ 10 items at 5 models
    # each, news incl. un-analyzed rows) made the client cursor overshoot the delivered
    # verdicts, so "load more" skipped the analyzed items in between (mid-feed gap).
    from hlbot.models import Analysis, NewsItem, now_ms

    class _FakeLane4:
        key, model, gate, live, capital_usd = "x", "m", 0.8, False, 1.0

        class pm:
            @staticmethod
            def open_count():
                return 0

        @staticmethod
        def open_unrealized():
            return 0.0

    cfg = Config()
    store = Store(path=_tmp())
    t0 = now_ms() - 10**9
    models = [f"model-{k}" for k in "ABCDE"]

    async def seed():
        await store.init()
        for i in range(1, 13):   # 11 analyzed x 5 models = 55 analyses (> the old 50-row cap)
            await store.save_news(NewsItem(id=f"n{i}", title=f"headline {i}", body="",
                                           source="s", link=None, time_ms=t0 + i * 1000,
                                           received_ms=t0 + i * 1000))
            if i != 5:   # n5 was filtered before analysis — must not appear in the seed
                for m in models:
                    await store.save_analysis(Analysis(f"n{i}", "ACME", "equity", "long",
                                                       0.9, "days", False, "r", m))

    asyncio.run(seed())
    app = create_arena_app(EventBus(), store, cfg, [_FakeLane4()])
    client = TestClient(app)
    snap = client.get("/api/arena/snapshot").json()
    assert [n["id"] for n in snap["news"]] == [f"n{i}" for i in range(12, 0, -1) if i != 5]
    assert len(snap["analyses"]) == 55   # all 5 models for every seeded item, uncapped
    # Contiguity: paging from the oldest seeded item must come back empty, not reveal
    # items the seed silently skipped.
    oldest = min(n["time_ms"] for n in snap["news"])
    assert client.get("/api/arena/news", params={"before": oldest}).json()["news"] == []


def test_arena_snapshot_seeds_open_positions_uncapped():
    # The client REBUILDS its open-positions map from snapshot["open_positions"] on every
    # (re)connect — it must contain every open row across lanes (closed rows excluded),
    # NOT the mixed 50-row positions table, or positions closed while a viewer was
    # disconnected linger as ghosts / open ones older than the 50 newest rows vanish.
    class _FakeLane6:
        key, model, gate, live, capital_usd = "x", "model-A", 0.8, False, 1.0

        class pm:
            @staticmethod
            def open_count():
                return 0

        @staticmethod
        def open_unrealized():
            return 0.0

    cfg = Config()
    store = Store(path=_tmp())
    now = int(time.time() * 1000)

    async def seed():
        await store.init()
        await store.upsert_position(_closed_pos("c1", "model-A", +50.0, dry=True))
        for i, model in enumerate(["model-A", "model-B"]):
            await store.upsert_position(Position(
                id=f"o{i}", news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                side="long", size=1.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0,
                leverage=5, notional_usd=100.0, opened_ms=now - 10**8,  # older than 50 newest
                time_exit_ms=now + 10**9, dry_run=True, model_id=model))

    asyncio.run(seed())
    app = create_arena_app(EventBus(), store, cfg, [_FakeLane6()])
    client = TestClient(app)
    snap = client.get("/api/arena/snapshot").json()
    assert {p["id"] for p in snap["open_positions"]} == {"o0", "o1"}
    assert all(p["status"] == "open" for p in snap["open_positions"])


def test_arena_usage_endpoint_aggregates_per_model():
    # /api/arena/usage backs the usage modal: per-model analysis count / mean latency /
    # summed cost from the DB, merged with the token ledger priced via pricing.py.
    import json as _json
    import pathlib as _pathlib

    class _FakeLane4:
        key, model, gate, live, capital_usd = "x", "model-A", 0.8, False, 1.0

        class pm:
            @staticmethod
            def open_count():
                return 0

        @staticmethod
        def open_unrealized():
            return 0.0

    cfg = Config()
    ledger_path = _tmp()
    _pathlib.Path(ledger_path).write_text(_json.dumps(
        {"model-A": {"calls": 3, "input": 1_000_000, "output": 100_000,
                     "cache_read": 0, "cache_creation": 0}}), encoding="utf-8")
    cfg.app.token_ledger_file = ledger_path
    store = Store(path=_tmp())

    async def seed():
        await store.init()
        await store.save_news(_item())
        for lat, cost in ((2000, 0.01), (4000, 0.03), (10000, 0.05)):
            a = Analysis("n", "MRVL", "equity", "long", 0.9, "days", False, "r", "model-A")
            a.latency_ms, a.cost_usd = lat, cost
            await store.save_analysis(a)

    asyncio.run(seed())
    app = create_arena_app(EventBus(), store, cfg, [_FakeLane4()])
    client = TestClient(app)
    u = client.get("/api/arena/usage").json()
    row = {r["model"]: r for r in u["analysis"]}["model-A"]
    assert row["n"] == 3 and abs(row["cost"] - 0.09) < 1e-9
    assert abs(row["avg_ms"] - 16000 / 3) < 1 and row["med_ms"] == 4000   # mean ≠ median
    assert row["since_ms"] > 0   # per-model first-analysis ts (usage modal start date)
    led = u["ledger"]["model-A"]
    assert led["calls"] == 3 and led["est_cost_usd"] > 0   # priced from the token counts
    # Competition start date rides the snapshot (header "since <date>" + usage modal).
    snap = client.get("/api/arena/snapshot").json()
    assert snap["started_ms"] == row["since_ms"]


def test_arena_model_trades_endpoint_filters_by_lane():
    # /api/arena/model_trades backs the per-model modal (leaderboard chip click): the full
    # closed history for ONE lane, and non-entrant model strings must 404 (public endpoint,
    # not a DB probe).
    class _FakeLane5:
        key, model, gate, live, capital_usd = "x", "model-A", 0.8, False, 1.0

        class pm:
            @staticmethod
            def open_count():
                return 0

        @staticmethod
        def open_unrealized():
            return 0.0

    cfg = Config()
    store = Store(path=_tmp())

    async def seed():
        await store.init()
        await store.upsert_position(_closed_pos("a1", "model-A", +50.0, dry=True))
        await store.upsert_position(_closed_pos("a2", "model-A", -20.0, dry=True))
        await store.upsert_position(_closed_pos("b1", "model-B", +999.0, dry=True))

    asyncio.run(seed())
    app = create_arena_app(EventBus(), store, cfg, [_FakeLane5()])
    client = TestClient(app)
    d = client.get("/api/arena/model_trades", params={"model": "model-A"}).json()
    assert {p["id"] for p in d["closed"]} == {"a1", "a2"}          # only its own lane
    assert all(p["model_id"] == "model-A" for p in d["closed"])
    assert d["closed"][0]["pnl_usd"] is not None                   # full row payload
    assert client.get("/api/arena/model_trades",
                      params={"model": "model-B"}).status_code == 404   # not an entrant
    assert client.get("/api/arena/model_trades").status_code == 404    # missing param


def test_candles_proxy_is_clamped_and_rate_limited():
    # /api/candles is public once the dashboard is tunnelled: a single URL must not be
    # able to request unbounded history, and one client must not be able to hammer HL
    # from this box's IP (per-IP sliding window keyed by Cloudflare's viewer-IP header).
    from hlbot.ui.arena_server import CANDLE_RATE_MAX, MAX_CANDLE_BARS

    class _FakeMarketHL:
        def __init__(self):
            self.calls = []

        async def candles(self, coin, interval, start_ms, end_ms):
            self.calls.append((coin, interval, start_ms, end_ms))
            return []

    cfg = Config()
    store = Store(path=_tmp())
    asyncio.run(store.init())
    hl = _FakeMarketHL()
    app = create_arena_app(EventBus(), store, cfg, [], hl=hl)
    client = TestClient(app)

    # Span clamp: ask for ~10x the allowed 5m history — upstream sees at most MAX bars.
    end = int(time.time() * 1000)
    start = end - 300_000 * MAX_CANDLE_BARS * 10
    assert client.get("/api/candles", params={"coin": "BTC", "interval": "5m",
                                              "start": start, "end": end}).status_code == 200
    _, _, got_start, got_end = hl.calls[-1]
    assert got_end - got_start == 300_000 * MAX_CANDLE_BARS

    assert client.get("/api/candles", params={"coin": "BTC", "interval": "9m"}).status_code == 400
    assert client.get("/api/candles", params={"coin": "x" * 33}).status_code == 400

    # Per-IP rate limit: a distinct viewer IP gets its own budget, then 429s.
    hdr = {"cf-connecting-ip": "203.0.113.7"}
    codes = [client.get("/api/candles", params={"coin": "BTC"}, headers=hdr).status_code
             for _ in range(CANDLE_RATE_MAX + 1)]
    assert codes[:CANDLE_RATE_MAX] == [200] * CANDLE_RATE_MAX and codes[-1] == 429
    # ...without exhausting other viewers' budgets.
    assert client.get("/api/candles", params={"coin": "BTC"},
                      headers={"cf-connecting-ip": "203.0.113.8"}).status_code == 200


# ---- trail high-water mark survives a restart ----------------------------------
def test_peak_px_restore_roundtrip_preserves_trailing_stop():
    async def run():
        cfg = Config()
        store = await Store(path=_tmp()).init()
        now = int(time.time() * 1000)
        pos = Position(id="p1", news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                       side="long", size=1.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0,
                       leverage=5, notional_usd=100.0, opened_ms=now, time_exit_ms=now + 10**9,
                       dry_run=True, trail_pct=0.08, peak_px=120.0)   # advanced high-water mark
        pm = PositionManager(cfg, None, Executor(cfg, None, store, EventBus()))
        pm.track(pos)
        eff_before, label_before = pm._eff_stop_label(pos)
        await store.upsert_position(pos)

        pm2 = PositionManager(cfg, None, Executor(cfg, None, store, EventBus()))
        await pm2.restore(store)                      # the restart
        restored = pm2._open["p1"]
        eff_after, label_after = pm2._eff_stop_label(restored)
        await store.close()
        return restored, (eff_before, label_before), (eff_after, label_after)

    restored, before, after = asyncio.run(run())
    assert restored.peak_px == 120.0                  # HWM survived the round-trip
    assert after == before                            # trailing stop did NOT widen
    assert after[1] == "trailing stop" and abs(after[0] - 120.0 * 0.92) < 1e-9


# ---- partial live IOC close: remainder stays tracked and protected --------------
def test_partial_live_close_keeps_remainder_tracked():
    async def run():
        cfg = Config()
        store = await Store(path=_tmp()).init()
        hl = FakeLiveHL(fill_sz=4.0, px=110.0)        # close only fills 4 of 10
        ex = Executor(cfg, hl, store, EventBus())
        now = now_ms()
        pos = Position(id="p1", news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                       side="long", size=10.0, entry_px=100.0, stop_loss=97.0,
                       take_profit=0.0, leverage=5, notional_usd=1000.0, opened_ms=now,
                       time_exit_ms=now + 10**9, dry_run=False, stop_order_id=7)
        first = await ex.close(pos, "time exit")
        calls_after_first = list(hl.calls)
        hl.fill_sz = pos.size                          # next attempt fills the remainder
        second = await ex.close(pos, "time exit")
        await store.close()
        return first, second, pos, calls_after_first

    first, second, pos, calls = asyncio.run(run())
    assert first is None                               # NOT recorded as fully closed
    assert abs(pos.size - 6.0) < 1e-9                  # remainder still tracked
    assert abs(pos.notional_usd - 600.0) < 1e-9
    assert pos.partial_pnl_usd > 0                     # the filled slice was realized
    assert ("cancel", 7) not in calls                  # backstop stop NOT cancelled on partial
    assert second is not None and second.status == "closed"
    # Final PnL = both slices: 10 * (110-100) minus fees on both legs.
    fees = (100.0 + 110.0) * 10.0 * 0.00045
    assert abs(second.pnl_usd - (100.0 - fees)) < 1e-6


def test_parse_fill_none_response():
    # SDK market_close returns None when no matching position exists on the exchange.
    px, sz, err = HLClient.parse_fill(None)
    assert px is None and sz is None and "no position" in err
