"""Regression tests for the audit-round fixes:
 #2  failed LIVE close must NOT mark a position closed (returns None, stays open)
 #3  daily-loss halt auto-resumes next UTC day; a manual kill never does
 #5  backtest exposure clamp (simulate notional_cap) + utc_day helper
 #10 dedup memory restored from persisted ids
"""
import asyncio

from hlbot.backtest.engine import scale_notional, simulate, utc_day
from hlbot.config import Config, FiltersConfig
from hlbot.models import Market, NewsItem, Position, now_ms
from hlbot.news.dedup import Dedup
from hlbot.trading.executor import Executor


# --------------------------------------------------------------------------- #
# #10 dedup persistence
# --------------------------------------------------------------------------- #
def test_dedup_restore_seeds_seen_ids():
    d = Dedup(FiltersConfig(), memory=100)
    d.restore(["tg:c:3", "tg:c:2", "tg:c:1"])   # newest-first (as the store returns them)
    dupe = NewsItem(id="tg:c:2", title="t", body="b", source="s", link=None,
                    time_ms=now_ms(), received_ms=now_ms())
    keep, reason = d.check(dupe)
    assert keep is False and reason == "duplicate id"
    fresh = NewsItem(id="tg:c:9", title="t", body="b", source="s", link=None,
                     time_ms=now_ms(), received_ms=now_ms())
    keep2, _ = d.check(fresh)
    assert keep2 is True


# --------------------------------------------------------------------------- #
# #3 daily-loss auto-reset
# --------------------------------------------------------------------------- #
def test_daily_loss_halt_auto_resumes_next_day(monkeypatch, tmp_path):
    import hlbot.config as cfgmod
    monkeypatch.setattr(cfgmod, "RUNTIME_STATE_FILE", str(tmp_path / "rt.json"))
    rt = cfgmod.RuntimeState(dry_run=True)
    rt.halt_daily("daily loss limit hit (-1600)")
    assert rt.trading_halted and rt.halt_is_daily
    assert rt.maybe_auto_resume() is False and rt.trading_halted   # same UTC day -> stays
    rt.halt_day = "2000-01-01"                                      # pretend it was an old day
    assert rt.maybe_auto_resume() is True and not rt.trading_halted  # rolled -> auto-resume


def test_manual_kill_never_auto_resumes(monkeypatch, tmp_path):
    import hlbot.config as cfgmod
    monkeypatch.setattr(cfgmod, "RUNTIME_STATE_FILE", str(tmp_path / "rt.json"))
    rt = cfgmod.RuntimeState(dry_run=True)
    rt.halt("manual kill switch")
    rt.halt_day = "2000-01-01"
    assert rt.maybe_auto_resume() is False and rt.trading_halted    # manual halts persist
    # and a daily-loss halt must not downgrade an active manual kill
    rt.halt_daily("daily loss")
    assert rt.halt_is_daily is False


# --------------------------------------------------------------------------- #
# #2 failed live close keeps the position
# --------------------------------------------------------------------------- #
class _FakeStore:
    async def upsert_position(self, p):  # noqa: ANN001
        pass


class _FakeBus:
    async def publish(self, *a, **k):
        pass


class _FailCloseHL:
    """A live close that never fills (e.g. exchange rejects the order)."""
    async def market_close(self, market, slippage):  # noqa: ANN001
        return {"status": "err", "response": "rejected"}

    async def mid(self, market):  # noqa: ANN001
        return 100.0


def _open_pos(side="long"):
    return Position(id="p1", news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                    side=side, size=10.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0,
                    leverage=5, notional_usd=1000.0, opened_ms=0, time_exit_ms=10**13,
                    dry_run=False)


def test_live_close_failure_returns_none_and_keeps_position():
    cfg = Config()
    cfg.runtime.dry_run = False
    ex = Executor(cfg, _FailCloseHL(), _FakeStore(), _FakeBus())
    pos = _open_pos()
    out = asyncio.run(ex.close(pos, "stop loss"))
    assert out is None                 # signals "did not close" to the caller
    assert pos.status == "open"        # never marked closed -> manager keeps managing it


# --------------------------------------------------------------------------- #
# #5 exposure clamp + utc_day
# --------------------------------------------------------------------------- #
class _CandleHL:
    class _Info:
        def candles_snapshot(self, name, interval, start, end):  # noqa: ANN001
            return [{"t": start + i * 60000, "T": start + (i + 1) * 60000,
                     "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0} for i in range(200)]

    def __init__(self):
        self.info = self._Info()


def test_simulate_notional_cap_clamps_exposure(monkeypatch, tmp_path):
    from hlbot.backtest import engine as bt_engine
    monkeypatch.setattr(bt_engine, "CANDLE_CACHE_DIR", tmp_path)  # keep data/ clean
    cfg = Config()
    r = cfg.app.risk
    mkt = Market(name="BTC", symbol="BTC", dex="", asset_class="crypto",
                 sz_decimals=2, max_leverage=5)
    news = 1_700_000_000_000
    capped = asyncio.run(simulate(_CandleHL(), mkt, "long", news, 0.9, "days", r,
                                  notional_cap=500.0))
    # Sizing mirrors live (risk.py): size rounds to sz_decimals, notional recomputed from it.
    assert capped is not None
    exp_size = round(500.0 / capped.entry_px, mkt.sz_decimals)
    assert abs(capped.size - exp_size) < 1e-9
    assert abs(capped.notional - exp_size * capped.entry_px) < 1e-9
    full = asyncio.run(simulate(_CandleHL(), mkt, "long", news, 0.9, "days", r))
    exp_full = round(scale_notional(0.9, r) / full.entry_px, mkt.sz_decimals)
    assert abs(full.size - exp_full) < 1e-9   # uncapped = tiered size
    assert abs(full.notional - exp_full * full.entry_px) < 1e-9


def test_utc_day_boundary():
    assert utc_day(0) == "1970-01-01"
    assert utc_day(1_700_000_000_000) == utc_day(1_700_000_000_000 + 60_000)  # same day
