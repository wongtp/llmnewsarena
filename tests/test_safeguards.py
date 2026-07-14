"""Contrary-news exit safeguard: bearish news on a long we hold (or bullish on a
short) closes the position immediately, even below the entry gate."""
import asyncio

from hlbot.config import Config
from hlbot.models import Analysis, Decision, Market, NewsItem, Position, now_ms
from hlbot.pipeline import Pipeline


class FakeBus:
    async def publish(self, *a, **k):
        pass


class FakeStore:
    async def save_news(self, *a):
        pass

    async def save_analysis(self, *a):
        pass

    async def save_decision(self, *a):
        pass


class FakeDedup:
    def check(self, item):
        return True, ""


class FakeAnalyzer:
    def __init__(self, an):
        self.an = an

    async def analyze(self, item, universe):
        return self.an


class FakeUniverse:
    def __init__(self, mkt):
        self.mkt = mkt

    def resolve(self, ticker, asset_class):
        return self.mkt


class FakeRisk:
    def __init__(self, cfg):
        self.r = cfg.app.risk
        self.evaluated = False

    async def evaluate(self, item, analysis, market):
        self.evaluated = True
        return Decision(news_id=item.id, action="reject", reason="position already open",
                        market=market)

    def note_entry(self, symbol):
        pass


class FakePM:
    def __init__(self, pos):
        self._pos = pos
        self.closed = None

    def has_open(self, name):
        return self._pos is not None and self._pos.market == name

    def position_for(self, name):
        return self._pos if (self._pos and self._pos.market == name) else None

    async def force_close(self, pos, reason):
        self.closed = (pos, reason)
        self._pos = None

    def track(self, pos):
        pass


def _pos(side="long"):
    return Position(id="p", news_id="n0", market="xyz:GME", symbol="GME", dex="xyz", side=side,
                    size=1.0, entry_px=30.0, stop_loss=29.0, take_profit=0.0, leverage=5,
                    notional_usd=30.0, opened_ms=0, time_exit_ms=10**13, dry_run=True)


def _pipe(analysis, pos):
    cfg = Config()
    mkt = Market("xyz:GME", "GME", "xyz", "equity", 2, 5)
    risk = FakeRisk(cfg)
    pm = FakePM(pos)
    pipe = Pipeline(bus=FakeBus(), store=FakeStore(), dedup=FakeDedup(),
                    analyzer=FakeAnalyzer(analysis), universe=FakeUniverse(mkt),
                    risk=risk, executor=None, position_manager=pm)
    return pipe, pm, risk


def _drive(pipe, item):
    async def go():
        await pipe.on_news(item)
        await pipe.drain()
    asyncio.run(go())


def _news():
    return NewsItem("n1", "RK account possibly hacked", "discount bullish posts", "X", None,
                    now_ms(), now_ms())


def test_contrary_news_closes_long():
    an = Analysis("n1", "GME", "equity", "short", 0.55, "hours", False, "RK hacked", "m")
    pipe, pm, risk = _pipe(an, _pos("long"))
    _drive(pipe, _news())
    assert pm.closed and pm.closed[1] == "contrary news exit"
    assert not risk.evaluated   # short-circuits before the entry path


def test_low_confidence_contrary_below_floor_does_not_close():
    an = Analysis("n1", "GME", "equity", "short", 0.30, "hours", False, "weak", "m")  # < 0.50 floor
    pipe, pm, risk = _pipe(an, _pos("long"))
    _drive(pipe, _news())
    assert pm.closed is None
    assert risk.evaluated       # falls through to the normal (reject) path


def test_same_direction_news_does_not_close():
    an = Analysis("n1", "GME", "equity", "long", 0.90, "hours", False, "more bullish", "m")
    pipe, pm, risk = _pipe(an, _pos("long"))
    _drive(pipe, _news())
    assert pm.closed is None     # confirming news, not contrary -> hold
    assert risk.evaluated
