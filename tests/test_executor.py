import asyncio
import tempfile

from hlbot.bus import EventBus
from hlbot.config import Config
from hlbot.models import Analysis, Decision, Market, NewsItem, now_ms
from hlbot.store.db import Store
from hlbot.trading.executor import Executor


class FakeHL:
    def __init__(self, price):
        self.price = price

    async def mid(self, market):
        return self.price


def _tmp():
    return tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name


async def _scenario(entry, exit_px, side):
    cfg = Config()
    cfg.runtime.dry_run = True
    store = await Store(path=_tmp()).init()
    ex = Executor(cfg, FakeHL(exit_px), store, EventBus())
    market = Market("xyz:MRVL", "MRVL", "xyz", "equity", 2, 5)
    sl = entry * 0.985 if side == "long" else entry * 1.015
    tp = entry * 1.03 if side == "long" else entry * 0.97
    dec = Decision(news_id="n", action="enter", reason="r", market=market, side=side,
                   notional_usd=entry, size=1.0, leverage=3, entry_px=entry,
                   stop_loss=sl, take_profit=tp, time_exit_seconds=1800, confidence=0.9)
    item = NewsItem("n", "t", "b", "X", "http://l", now_ms(), now_ms())
    an = Analysis("n", "MRVL", "equity", side, 0.9, "immediate", False, "r", "m")
    pos = await ex.open(dec, item, an)
    # dry-run models adverse fill slippage on entry (L4)
    slip = cfg.app.risk.dry_run_slippage_pct
    expected_entry = entry * (1 + slip) if side == "long" else entry * (1 - slip)
    assert pos and pos.status == "open" and abs(pos.entry_px - expected_entry) < 1e-6
    closed = await ex.close(pos, "take profit")
    await store.close()
    return closed


def test_long_profit():
    closed = asyncio.run(_scenario(100.0, 110.0, "long"))
    assert closed.status == "closed" and closed.pnl_usd > 0


def test_short_profit():
    closed = asyncio.run(_scenario(100.0, 90.0, "short"))
    assert closed.pnl_usd > 0


def test_long_loss():
    closed = asyncio.run(_scenario(100.0, 95.0, "long"))
    assert closed.pnl_usd < 0
