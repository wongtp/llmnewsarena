"""Exchange-side protective stop: a reduce-only stop is placed on entry (live), cancelled on a
normal close, and the position manager reconciles positions the exchange closed on its own."""
import asyncio
import tempfile

from hlbot.bus import EventBus
from hlbot.config import Config
from hlbot.models import Analysis, Decision, Market, NewsItem, Position, now_ms
from hlbot.store.db import Store
from hlbot.trading.executor import Executor
from hlbot.trading.hl_client import HLClient
from hlbot.trading.position_manager import PositionManager


def _tmp():
    return tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name


def _ok_fill(px, sz):
    return {"status": "ok", "response": {"data": {"statuses": [
        {"filled": {"avgPx": str(px), "totalSz": str(sz)}}]}}}


def _resting(oid):
    return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": oid}}]}}}


class FakeExchangeHL:
    """Mimics the order-placement surface the live Executor uses, recording each call."""
    def __init__(self, fill_px=100.0, fill_sz=1.0, stop_result=None):
        self.fill_px, self.fill_sz = fill_px, fill_sz
        self.stop_result = stop_result if stop_result is not None else _resting(777)
        self.leverage_calls, self.stops, self.cancels, self.market_closes = [], [], [], []

    async def mid(self, market):
        return self.fill_px

    async def set_leverage(self, market, leverage):
        self.leverage_calls.append((market.name, leverage))

    async def market_open(self, market, is_buy, size, slippage):
        return _ok_fill(self.fill_px, self.fill_sz)

    async def place_stop(self, market, is_buy_to_close, size, trigger_px, slippage):
        self.stops.append({"market": market.name, "is_buy_to_close": is_buy_to_close,
                           "size": size, "trigger_px": trigger_px})
        return self.stop_result

    async def cancel_order(self, market, oid):
        self.cancels.append((market.name, oid))

    async def market_close(self, market, slippage):
        self.market_closes.append(market.name)
        return _ok_fill(self.fill_px, self.fill_sz)


def _decision(side="long", entry=100.0):
    market = Market("xyz:MRVL", "MRVL", "xyz", "equity", 2, 5)
    sl = entry * 0.97 if side == "long" else entry * 1.03
    return Decision(news_id="n", action="enter", reason="r", market=market, side=side,
                    notional_usd=entry, size=1.0, leverage=5, entry_px=entry, stop_loss=sl,
                    take_profit=0.0, time_exit_seconds=1800, confidence=0.9), market


async def _open_live(hl, side="long"):
    cfg = Config()
    cfg.runtime.dry_run = False
    store = await Store(path=_tmp()).init()
    ex = Executor(cfg, hl, store, EventBus())
    dec, market = _decision(side)
    item = NewsItem("n", "t", "b", "X", "http://l", now_ms(), now_ms())
    an = Analysis("n", "MRVL", "equity", side, 0.9, "immediate", False, "r", "m")
    pos = await ex.open(dec, item, an)
    return cfg, store, ex, market, pos


# ---- resting-oid parser -----------------------------------------------------
def test_parse_resting_oid():
    assert HLClient.parse_resting_oid(_resting(12345)) == 12345
    assert HLClient.parse_resting_oid(_ok_fill(100.0, 1.0)) is None      # filled, not resting
    assert HLClient.parse_resting_oid({"status": "err"}) is None
    assert HLClient.parse_resting_oid({"status": "ok", "response": {"data": {"statuses": [
        {"error": "bad"}]}}}) is None


# ---- stop placed on entry, correct side, oid recorded -----------------------
def test_live_long_entry_places_reduce_only_sell_stop():
    hl = FakeExchangeHL(stop_result=_resting(777))
    cfg, store, ex, market, pos = asyncio.run(_open_live(hl, "long"))
    assert pos.stop_order_id == 777
    assert len(hl.stops) == 1
    assert hl.stops[0]["is_buy_to_close"] is False     # closing a LONG = sell
    assert abs(hl.stops[0]["trigger_px"] - pos.stop_loss) < 1e-9
    asyncio.run(store.close())


def test_live_short_entry_places_reduce_only_buy_stop():
    hl = FakeExchangeHL(stop_result=_resting(888))
    cfg, store, ex, market, pos = asyncio.run(_open_live(hl, "short"))
    assert pos.stop_order_id == 888
    assert hl.stops[0]["is_buy_to_close"] is True       # closing a SHORT = buy
    asyncio.run(store.close())


def test_live_slipped_fill_reanchors_stop_to_fill_price():
    # Decision: entry 100 -> SL 97 (3% away). The fill slips to 102: the stop must keep
    # the 3% distance from the REAL entry (97 * 1.02) and the exchange stop rests there,
    # instead of inheriting the pre-trade level (which would tighten the stop to ~1%).
    hl = FakeExchangeHL(fill_px=102.0, stop_result=_resting(9))
    cfg, store, ex, market, pos = asyncio.run(_open_live(hl, "long"))
    assert abs(pos.stop_loss - 97.0 * 1.02) < 1e-9
    assert abs(hl.stops[0]["trigger_px"] - pos.stop_loss) < 1e-9
    asyncio.run(store.close())


def test_entry_survives_stop_placement_failure():
    # place_stop didn't rest (e.g. rejected) -> entry still succeeds, no oid, bot-side stop only.
    hl = FakeExchangeHL(stop_result={"status": "ok", "response": {"data": {"statuses": [
        {"error": "tick size"}]}}})
    cfg, store, ex, market, pos = asyncio.run(_open_live(hl, "long"))
    assert pos is not None and pos.status == "open" and pos.stop_order_id == 0
    asyncio.run(store.close())


# ---- normal close cancels the resting stop ----------------------------------
def test_close_cancels_resting_stop():
    async def run():
        hl = FakeExchangeHL(stop_result=_resting(777))
        cfg, store, ex, market, pos = await _open_live(hl, "long")
        closed = await ex.close(pos, "take profit")
        await store.close()
        return hl, closed
    hl, closed = asyncio.run(run())
    assert closed.status == "closed"
    assert hl.market_closes == ["xyz:MRVL"]
    assert hl.cancels == [("xyz:MRVL", 777)]            # resting stop cancelled
    assert closed.stop_order_id == 0


# ---- mark_closed: record without sending an order ---------------------------
def test_paper_position_closes_on_paper_even_when_runtime_is_live():
    # A dry-run position carried into a LIVE runtime must close on PAPER — never send a real
    # market_close for a position that never existed on the exchange.
    async def run():
        hl = FakeExchangeHL()
        cfg = Config(); cfg.runtime.dry_run = False        # runtime is LIVE...
        store = await Store(path=_tmp()).init()
        ex = Executor(cfg, hl, store, EventBus())
        pos = Position(id="p", news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                       side="long", size=1.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0,
                       leverage=5, notional_usd=100.0, opened_ms=now_ms(), time_exit_ms=10**13,
                       dry_run=True)                       # ...but the position is PAPER
        closed = await ex.close(pos, "time exit", exit_px=99.0)
        await store.close()
        return hl, closed
    hl, closed = asyncio.run(run())
    assert closed and closed.status == "closed"
    assert hl.market_closes == []                          # NO real close order was sent


def test_mark_closed_records_pnl_without_order():
    async def run():
        hl = FakeExchangeHL()
        cfg = Config(); cfg.runtime.dry_run = False
        store = await Store(path=_tmp()).init()
        ex = Executor(cfg, hl, store, EventBus())
        pos = Position(id="p", news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                       side="long", size=2.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0,
                       leverage=5, notional_usd=200.0, opened_ms=now_ms(), time_exit_ms=10**13,
                       dry_run=False)
        out = await ex.mark_closed(pos, "stop loss (exchange)", 97.0)
        await store.close()
        return hl, out
    hl, out = asyncio.run(run())
    assert out.status == "closed" and out.exit_reason == "stop loss (exchange)"
    assert out.exit_px == 97.0 and out.pnl_usd < 0      # loss at the stop, incl. fees
    assert hl.market_closes == []                       # NO close order was sent


# ---- reconciliation: detect exchange-closed positions -----------------------
class FakeHLPos:
    """position_state fake: {dex: {coin: size}} (+ optional per-coin funding)."""

    def __init__(self, by_dex, raise_dex=None, funding=None):
        self.by_dex, self.raise_dex = by_dex, raise_dex
        self.funding = funding or {}

    async def position_state(self, dex):
        if self.raise_dex is not None and dex == self.raise_dex:
            raise RuntimeError("user_state failed")
        return {c: {"szi": sz, "funding": self.funding.get(c, 0.0)}
                for c, sz in self.by_dex.get(dex, {}).items()}

    async def mid(self, market):
        return None                                  # price-driven exits stay out of the way

    async def funding_rate(self, market):
        return 0.0


class FakeExec:
    def __init__(self):
        self.marked = []
        self.bus = EventBus()

    async def mark_closed(self, pos, reason, px):
        self.marked.append((pos.id, reason, px))
        return pos


def _live(pid, market="xyz:MRVL", symbol="MRVL", dex="xyz", dry_run=False, age_ms=60_000):
    return Position(id=pid, news_id="n", market=market, symbol=symbol, dex=dex, side="long",
                    size=1.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0, leverage=5,
                    notional_usd=100.0, opened_ms=now_ms() - age_ms, time_exit_ms=10**13,
                    dry_run=dry_run)


def _pm(hl, positions):
    pm = PositionManager(config=Config(), hl=hl, executor=FakeExec())
    for p in positions:
        pm._open[p.id] = p
    return pm


async def _reconcile(pm):
    await pm._reconcile_exchange_closes(await pm._position_states())


def test_reconcile_marks_absent_live_position_closed():
    # MRVL is gone from the exchange (stop fired); GME is still open; AAPL is a paper position.
    hl = FakeHLPos({"xyz": {"xyz:GME": 5.0}})
    pm = _pm(hl, [_live("mrvl"), _live("gme", "xyz:GME", "GME"),
                  _live("aapl", "xyz:AAPL", "AAPL", dry_run=True)])
    # Debounce: ONE absent poll must not close (a single glitched user_state response would
    # otherwise record a close for real exposure) — it takes two consecutive misses.
    asyncio.run(_reconcile(pm))
    assert pm.executor.marked == [] and "mrvl" in pm._open
    asyncio.run(_reconcile(pm))
    assert pm.executor.marked == [("mrvl", "stop loss (exchange)", 97.0)]
    assert set(pm._open) == {"gme", "aapl"}             # present + paper kept; absent removed


def test_reconcile_keeps_paper_positions():
    # Pure dry-run book: nothing to reconcile even though the exchange shows it flat.
    hl = FakeHLPos({"xyz": {}})
    pm = _pm(hl, [_live("p", dry_run=True)])
    asyncio.run(_reconcile(pm))
    assert pm.executor.marked == [] and set(pm._open) == {"p"}


def test_reconcile_fails_safe_on_fetch_error():
    hl = FakeHLPos({}, raise_dex="xyz")
    pm = _pm(hl, [_live("mrvl")])
    asyncio.run(_reconcile(pm))
    assert pm.executor.marked == [] and set(pm._open) == {"mrvl"}   # can't confirm -> keep


def test_reconcile_grace_period_protects_fresh_entry():
    hl = FakeHLPos({"xyz": {}})                          # exchange shows flat...
    pm = _pm(hl, [_live("mrvl", age_ms=0)])              # ...but it was opened just now
    asyncio.run(_reconcile(pm))
    assert pm.executor.marked == [] and set(pm._open) == {"mrvl"}   # within grace -> keep


# ---- funding-paid tracking ----------------------------------------------------
def test_live_funding_copied_from_exchange_state():
    # The exchange reports cumFunding sinceOpen; the matched LIVE position picks it up.
    hl = FakeHLPos({"xyz": {"xyz:MRVL": 1.0}}, funding={"xyz:MRVL": 1.23})
    pm = _pm(hl, [_live("mrvl")])

    async def run():
        states = await pm._position_states()
        pm._update_live_funding(states)
    asyncio.run(run())
    assert abs(pm._open["mrvl"].funding_usd - 1.23) < 1e-9


def test_paper_funding_accrues_estimate_with_side_sign():
    # rate 0.0001/h on $100 notional for exactly 1h: long pays +$0.01, short receives -$0.01.
    class RateHL(FakeHLPos):
        async def funding_rate(self, market):
            return 0.0001

    async def accrue(side):
        hl = RateHL({})
        pos = _live("p", dry_run=True)
        pos.side = side
        pm = _pm(hl, [pos])
        market = Market(pos.market, pos.symbol, pos.dex, "equity", 2, 5)
        pm._funding_t[pos.id] = __import__("time").monotonic() - 3600.0   # 1h since last accrual
        await pm._accrue_paper_funding(pos, market)
        return pos.funding_usd

    paid_long = asyncio.run(accrue("long"))
    paid_short = asyncio.run(accrue("short"))
    assert abs(paid_long - 0.01) < 5e-4
    assert abs(paid_short + 0.01) < 5e-4


def test_tick_payload_includes_funding():
    # Full monitor pass over a paper position: the published tick carries funding.
    class TickHL(FakeHLPos):
        async def mid(self, market):
            return 100.0

        async def funding_rate(self, market):
            return 0.0

    class TickExec(FakeExec):
        def __init__(self):
            super().__init__()

            class _S:
                async def upsert_position(self, pos):
                    pass
            self.store = _S()

    hl = TickHL({})
    pos = _live("p", dry_run=True)
    pos.funding_usd = 0.42
    pm = PositionManager(config=Config(), hl=hl, executor=TickExec())
    pm._open[pos.id] = pos

    async def run():
        q = pm.executor.bus.subscribe("positions.tick")
        await pm._check_all()
        ev = q.get_nowait()
        return ev.payload["ticks"][0]
    tick = asyncio.run(run())
    assert tick["id"] == "p" and abs(tick["funding"] - 0.42) < 1e-9
