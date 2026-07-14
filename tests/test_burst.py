"""Burst aggregation: buffer semantics, the combined pseudo-item, and the pipeline's
piece-1-acts-instantly / piece-2-acts-holistically behavior (analyzer.burst_window_seconds,
default 0 = off)."""
import asyncio
import types

from hlbot.analysis.burst import MAX_PIECES, BurstBuffer, build_burst_item
from hlbot.models import Analysis, Decision, Market, NewsItem
from hlbot.pipeline import Pipeline


def _item(i, body, t_ms):
    return NewsItem(f"n{i}", "tradfi", body, "Telegram:trad_fin", None, t_ms, t_ms)


# ---- BurstBuffer ------------------------------------------------------------------


def test_buffer_disabled_at_zero_window():
    b = BurstBuffer(0)
    b.add("MU", 1000, "eps line")
    assert not b.enabled and b.prior("MU", 2000) == []


def test_buffer_window_and_ordering():
    b = BurstBuffer(300)                       # 300s window
    b.add("MU", 1_000_000, "eps line")
    b.add("MU", 1_060_000, "rev line")         # 60s later
    assert b.prior("MU", 1_100_000) == ["eps line", "rev line"]
    assert b.prior("MU", 1_000_000 + 301_000) == ["rev line"]   # first piece aged out


def test_buffer_ticker_isolation_and_dup_text():
    b = BurstBuffer(300)
    b.add("MU", 1000, "eps line")
    b.add("MU", 2000, "eps line")              # exact re-print ignored
    b.add("AMD", 1500, "amd line")
    assert b.prior("MU", 3000) == ["eps line"]
    assert b.prior("AMD", 3000) == ["amd line"]
    assert b.prior("mu", 3000) == ["eps line"]  # case-insensitive ticker key


def test_buffer_caps_pieces():
    b = BurstBuffer(300)
    for i in range(MAX_PIECES + 3):
        b.add("MU", 1000 + i, f"line {i}")
    assert len(b.prior("MU", 2000)) == MAX_PIECES


def test_build_burst_item_deterministic_and_chronological():
    it = _item(2, "guidance line", 5000)
    combined = build_burst_item(it, ["eps line", "rev line"])
    assert combined.id == "n2+b2"                       # deterministic -> replay cache key
    assert combined.body == "eps line\nrev line\nguidance line"
    assert combined.source == it.source and combined.time_ms == it.time_ms


# ---- Pipeline integration ----------------------------------------------------------


class _Bus:
    async def publish(self, *a, **k):
        pass


class _Store:
    async def save_news(self, *a):
        pass

    async def save_analysis(self, *a):
        pass

    async def save_decision(self, *a):
        pass


class _Dedup:
    def check(self, item):
        return True, ""


class _Universe:
    def resolve(self, ticker, asset_class):
        return Market("xyz:MU", "MU", "xyz", "equity", 2, 5)


class _Risk:
    def __init__(self):
        self.seen = []          # (news_id, confidence) the trade section acted on
        self.r = types.SimpleNamespace(contrary_exit_min_confidence=0.0)

    async def evaluate(self, item, analysis, market):
        self.seen.append((analysis.news_id, analysis.confidence))
        return Decision(news_id=item.id, action="reject", reason="test", market=market)

    def note_entry(self, symbol):
        pass


class _PM:
    def has_open(self, name):
        return False

    def position_for(self, name):
        return None


class _BurstAnalyzer:
    """Solo pieces score 0.7; a combined burst body (multi-line) scores 0.9."""

    def __init__(self, window):
        self.cfg = types.SimpleNamespace(burst_window_seconds=window)
        self.calls = []

    async def analyze(self, item, universe):
        self.calls.append(item)
        conf = 0.9 if "\n" in item.body else 0.7
        return Analysis(item.id, "MU", "equity", "long", conf, "days", False, "r", "m")


def _pipe(an):
    return Pipeline(bus=_Bus(), store=_Store(), dedup=_Dedup(), analyzer=an,
                    universe=_Universe(), risk=_Risk(), executor=None,
                    position_manager=_PM())


def _run(pipe, items):
    async def go():
        for it in items:
            await pipe.on_news(it)
            await pipe.drain()     # process pieces in arrival order (wire lines are sparse)
    asyncio.run(go())


def test_first_piece_acts_instantly_second_acts_on_combined():
    an = _BurstAnalyzer(window=300)
    pipe = _pipe(an)
    _run(pipe, [_item(1, "*MU Q2 EPS $4.78, EST. $4.39", 1_000_000),
                _item(2, "*MU SEES Q3 EPS $8.22, EST. $4.71", 1_030_000)])
    # piece 1: one solo call, trade acted on the solo verdict (no waiting)
    # piece 2: solo call + combined call; trade acted on the COMBINED verdict
    assert [c.id for c in an.calls] == ["n1", "n2", "n2+b1"]
    assert "EPS $4.78" in an.calls[-1].body and "SEES Q3" in an.calls[-1].body
    assert pipe.risk.seen == [("n1", 0.7), ("n2+b1", 0.9)]


def test_pieces_outside_window_do_not_combine():
    an = _BurstAnalyzer(window=300)
    pipe = _pipe(an)
    _run(pipe, [_item(1, "eps line", 1_000_000),
                _item(2, "guidance line", 1_000_000 + 600_000)])   # 600s later
    assert [c.id for c in an.calls] == ["n1", "n2"]
    assert pipe.risk.seen == [("n1", 0.7), ("n2", 0.7)]


def test_burst_off_by_default_never_combines():
    an = _BurstAnalyzer(window=0)
    pipe = _pipe(an)
    _run(pipe, [_item(1, "eps line", 1_000_000),
                _item(2, "rev line", 1_010_000)])
    assert [c.id for c in an.calls] == ["n1", "n2"]
