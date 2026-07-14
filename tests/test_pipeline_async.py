"""Async pipeline behavior: ingestion (`on_news`) never blocks on analysis, a headline
burst is analyzed concurrently, the risk->execute section stays strictly serialized,
and drain() completes all in-flight work."""
import asyncio
import time

from hlbot.config import Config
from hlbot.models import Analysis, Decision, Market, NewsItem, now_ms
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


class PassDedup:
    def check(self, item):
        return True, ""


class SlowAnalyzer:
    """Counts concurrent analyze() calls so tests can assert burst parallelism."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def analyze(self, item, universe):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return Analysis(item.id, "MRVL", "equity", "long", 0.9, "hours", False, "r", "m")


class FakeUniverse:
    def resolve(self, ticker, asset_class):
        return Market("xyz:MRVL", "MRVL", "xyz", "equity", 2, 5)


class SerialRisk:
    """Asserts the trade section is never entered concurrently (the _trade_lock invariant)."""

    def __init__(self, cfg):
        self.r = cfg.app.risk
        self.active = 0
        self.max_active = 0
        self.evaluated = []

    async def evaluate(self, item, analysis, market):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        self.evaluated.append(item.id)
        return Decision(news_id=item.id, action="reject", reason="test", market=market)

    def note_entry(self, symbol):
        pass


class FakePM:
    def has_open(self, name):
        return False

    def position_for(self, name):
        return None

    def track(self, pos):
        pass


def _item(i):
    return NewsItem(f"n{i}", "t", f"body {i}", "X", None, now_ms(), now_ms())


def _pipe(analyzer, risk, dedup=None):
    return Pipeline(bus=FakeBus(), store=FakeStore(), dedup=dedup or PassDedup(),
                    analyzer=analyzer, universe=FakeUniverse(), risk=risk,
                    executor=None, position_manager=FakePM())


def test_burst_is_ingested_fast_analyzed_concurrently_traded_serially():
    an = SlowAnalyzer(delay=0.05)
    risk = SerialRisk(Config())
    pipe = _pipe(an, risk)

    async def go():
        t0 = time.monotonic()
        for i in range(4):
            await pipe.on_news(_item(i))
        enqueue_s = time.monotonic() - t0
        await pipe.drain()
        return enqueue_s

    enqueue_s = asyncio.run(go())
    # Serial analysis would take >= 4 x 50ms = 200ms; a generous ceiling below that still
    # proves ingest never waited on an analysis without flaking on a loaded CI box.
    assert enqueue_s < 0.15
    assert an.max_active > 1         # the burst was analyzed in parallel...
    assert risk.max_active == 1      # ...but the trade section stayed serialized
    assert sorted(risk.evaluated) == ["n0", "n1", "n2", "n3"]   # nothing lost


def test_duplicates_never_reach_the_analyzer():
    class DupDedup:
        def __init__(self):
            self.seen = set()

        def check(self, item):
            if item.id in self.seen:
                return False, "duplicate id"
            self.seen.add(item.id)
            return True, ""

    an = SlowAnalyzer(delay=0.0)
    risk = SerialRisk(Config())
    pipe = _pipe(an, risk, dedup=DupDedup())

    async def go():
        await pipe.on_news(_item(1))
        await pipe.on_news(_item(1))
        await pipe.drain()

    asyncio.run(go())
    assert risk.evaluated == ["n1"]
