import asyncio

from hlbot.config import Config
from hlbot.models import Analysis, Market, NewsItem, now_ms
from hlbot.trading.hl_client import HLClient
from hlbot.trading.risk import (
    RiskEngine,
    adjust_confidence,
    is_listing_news,
    liquidity_penalty,
)


class FakeUniverse:
    def __init__(self, mentioned=True):
        self._m = mentioned

    def mentions(self, symbol, text, hints=()):
        return self._m


def _book(bid_px, ask_px, bid_sz=1000.0, ask_sz=1000.0):
    return {"levels": [[{"px": str(bid_px), "sz": str(bid_sz)}],
                       [{"px": str(ask_px), "sz": str(ask_sz)}]], "time": 0}


class FakeHL:
    def __init__(self, price=100.0):
        self.price = price

    async def mid(self, market):
        return self.price


class FakeHLBook(FakeHL):
    """FakeHL that also serves an L2 book (for the spread/depth liquidity guard)."""
    def __init__(self, price=100.0, book=None, raise_book=False):
        super().__init__(price)
        self._book = book if book is not None else _book(99.95, 100.05)
        self._raise_book = raise_book

    async def l2_book(self, market):
        if self._raise_book:
            raise RuntimeError("book fetch failed")
        return self._book


class FakeStore:
    def __init__(self, pnl=0.0):
        self.pnl = pnl

    async def realized_pnl_today(self, dry_run, model_id=None):
        return self.pnl


class FakePM:
    def __init__(self, unrealized=0.0):
        self._open = {}
        self.exposure = 0.0
        self.unrealized = unrealized

    def has_open(self, name):
        return name in self._open

    def open_count(self):
        return len(self._open)

    def total_exposure(self):
        return self.exposure

    def unrealized_pnl(self, dry_run):
        return self.unrealized


def _market():
    return Market("xyz:MRVL", "MRVL", "xyz", "equity", 2, 5)


def _news():
    return NewsItem("n1", "t", "b", "X", None, now_ms(), now_ms())


def _analysis(conf=0.95, direction="long", stale=False, relation="derived"):
    return Analysis("n1", "MRVL", "equity", direction, conf, "immediate", stale, "x", "m",
                    subject_relation=relation)


def _config():
    c = Config()
    c.runtime.dry_run = True
    return c


def _run(engine, *a):
    return asyncio.run(engine.evaluate(*a))


def test_enter_high_confidence_max_size():
    c = _config()
    eng = RiskEngine(c, FakeHL(100.0), FakeStore(), FakePM())
    d = _run(eng, _news(), _analysis(conf=0.95), _market())
    assert d.action == "enter" and d.side == "long" and d.size > 0
    assert d.stop_loss < d.entry_px < d.take_profit   # immediate: fixed TP
    assert d.leverage == min(c.app.risk.max_leverage, 5)
    # confidence above min_confidence_for_max -> near max notional
    assert d.notional_usd > (c.app.risk.base_notional_usd + c.app.risk.max_notional_usd) / 2


def test_reject_low_confidence():
    eng = RiskEngine(_config(), FakeHL(), FakeStore(), FakePM())
    d = _run(eng, _news(), _analysis(conf=0.5), _market())
    assert d.action == "reject" and "confidence" in d.reason


def test_reject_stale():
    eng = RiskEngine(_config(), FakeHL(), FakeStore(), FakePM())
    d = _run(eng, _news(), _analysis(stale=True), _market())
    assert d.action == "reject" and "stale" in d.reason


def test_reject_untradable_market():
    eng = RiskEngine(_config(), FakeHL(), FakeStore(), FakePM())
    d = _run(eng, _news(), _analysis(), None)
    assert d.action == "reject" and "not tradable" in d.reason


def test_daily_loss_limit_halts():
    c = _config()
    eng = RiskEngine(c, FakeHL(), FakeStore(pnl=-c.app.risk.daily_loss_limit_usd - 1), FakePM())
    d = _run(eng, _news(), _analysis(), _market())
    assert d.action == "reject"
    assert c.runtime.trading_halted


def test_daily_loss_limit_counts_unrealized():
    # No realized loss yet, but open positions are deep underwater -> still halt new entries.
    c = _config()
    pm = FakePM(unrealized=-c.app.risk.daily_loss_limit_usd - 1)
    eng = RiskEngine(c, FakeHL(100.0), FakeStore(pnl=0.0), pm)
    d = _run(eng, _news(), _analysis(conf=0.95), _market())
    assert d.action == "reject" and "daily loss" in d.reason
    assert c.runtime.trading_halted


def test_daily_loss_limit_realized_plus_unrealized():
    # Neither alone trips the limit, but realized + unrealized together does.
    c = _config()
    half = -(c.app.risk.daily_loss_limit_usd / 2 + 1)
    eng = RiskEngine(c, FakeHL(100.0), FakeStore(pnl=half), FakePM(unrealized=half))
    d = _run(eng, _news(), _analysis(conf=0.95), _market())
    assert d.action == "reject" and "daily loss" in d.reason
    assert c.runtime.trading_halted


def test_short_direction_sl_above_tp_below():
    eng = RiskEngine(_config(), FakeHL(100.0), FakeStore(), FakePM())
    d = _run(eng, _news(), _analysis(direction="short"), _market())
    assert d.side == "short"
    assert d.take_profit < d.entry_px < d.stop_loss   # immediate: fixed TP below, stop above


def test_is_listing_news():
    assert is_listing_news("Upbit will list IO (io.net)")
    assert is_listing_news("Binance Futures Will List XYZUSDT")
    assert not is_listing_news("Marvell signs multi-year AI silicon deal")
    assert not is_listing_news("Coinbase invests in Ethena via ENA purchase")  # no listing verb


def test_duplicate_signal_suppressed():
    eng = RiskEngine(_config(), FakeHL(), FakeStore(), FakePM())
    d1 = _run(eng, _news(), _analysis(conf=0.95, direction="long"), _market())
    assert d1.action == "enter"
    d2 = _run(eng, _news(), _analysis(conf=0.95, direction="long"), _market())
    assert d2.action == "reject" and "duplicate" in d2.reason


def test_duplicate_suppressed_even_after_faded_first():
    # The GME case: a faded (<gate) read must still suppress a near-identical higher one.
    eng = RiskEngine(_config(), FakeHL(), FakeStore(), FakePM())
    d1 = _run(eng, _news(), _analysis(conf=0.60, direction="long"), _market())
    assert d1.action == "reject"                       # below gate (faded)
    d2 = _run(eng, _news(), _analysis(conf=0.95, direction="long"), _market())
    assert d2.action == "reject" and "duplicate" in d2.reason


def test_opposite_direction_not_deduped():
    eng = RiskEngine(_config(), FakeHL(), FakeStore(), FakePM())
    _run(eng, _news(), _analysis(conf=0.95, direction="long"), _market())
    d2 = _run(eng, _news(), _analysis(conf=0.95, direction="short"), _market())
    assert "duplicate" not in d2.reason                # different direction = different event


def test_duplicate_not_suppressed_for_different_event():
    # The SNDK bug: Western Digital earnings -> infer SNDK long, then SanDisk's OWN earnings
    # -> SNDK long a minute later. Same (ticker, direction) but DIFFERENT stories -> must NOT
    # block each other (low token overlap).
    eng = RiskEngine(_config(), FakeHL(100.0), FakeStore(), FakePM())
    wd = NewsItem("n1", "", "WESTERN DIGITAL 2Q NET REV 3.02B ADJ EPS 2.13 BEATS",
                  "X", None, now_ms(), now_ms())
    sk = NewsItem("n2", "", "SANDISK CORP Q2 REVENUE 3025 MILLION VS EST 2638 ADJ EPS 6.2 SNDK",
                  "X", None, now_ms(), now_ms())
    a = _analysis(conf=0.92, direction="long")
    d1 = _run(eng, wd, a, _market())
    assert d1.action == "enter"
    d2 = _run(eng, sk, a, _market())
    assert "duplicate" not in (d2.reason or "")        # different catalyst, not a dup


def test_dup_similarity_helpers():
    from hlbot.trading.risk import dup_fingerprint, dup_similarity
    same = "BTC BREAKS 100K ON ETF INFLOWS"
    assert dup_similarity(dup_fingerprint(same), dup_fingerprint(same)) == 1.0
    wd = dup_fingerprint("WESTERN DIGITAL 2Q NET REV 3.02B ADJ EPS 2.13")
    sk = dup_fingerprint("SANDISK CORP Q2 REVENUE 3025 MILLION ADJ EPS 6.2")
    assert dup_similarity(wd, sk) < 0.5                # different companies' earnings


def test_liquidity_penalty():
    r = Config().app.risk
    thin = Market("IO", "IO", "", "crypto", 2, 5, day_volume_usd=2_000_000)
    mid = Market("X", "X", "", "crypto", 2, 5, day_volume_usd=20_000_000)
    big = Market("BTC", "BTC", "", "crypto", 2, 5, day_volume_usd=2_000_000_000)
    equity = Market("xyz:MRVL", "MRVL", "xyz", "equity", 2, 5)
    assert liquidity_penalty(thin, r) == r.liquidity_penalty_low
    assert liquidity_penalty(mid, r) == r.liquidity_penalty_med
    assert liquidity_penalty(big, r) == 0.0
    assert liquidity_penalty(equity, r) == 0.0   # equities not penalized here


def test_adjust_confidence_stacks_penalties():
    r = Config().app.risk
    thin = Market("IO", "IO", "", "crypto", 2, 5, day_volume_usd=2_000_000)
    conf, notes = adjust_confidence(0.82, thin, "Upbit will list IO", r)
    assert abs(conf - (0.82 - r.liquidity_penalty_low - r.listing_penalty)) < 1e-9
    assert len(notes) == 2


# ---- liquidity (spread/depth) guard --------------------------------------
def test_book_spread_and_depth():
    bid, ask, mid, spread = HLClient.book_spread(_book(98.0, 102.0))
    assert (bid, ask, mid) == (98.0, 102.0, 100.0)
    assert abs(spread - 0.04) < 1e-9
    assert HLClient.book_spread({"levels": [[], []]}) is None          # empty book -> None
    assert HLClient.book_spread(_book(101.0, 100.0)) is None           # crossed -> None
    assert HLClient.top_depth_usd(_book(100.0, 100.0, 5.0, 9.0)) == 500.0  # thinner side
    assert HLClient.top_depth_usd({"levels": [[], []]}) == 0.0         # one-sided/empty -> 0


def test_top_depth_sums_near_touch_levels():
    # A tiny BEST quote with real depth right behind it must NOT read as illiquid: the metric
    # sums the top N levels, not just the touch (the MRVL '$13 top-of-book' false alarm).
    bids = [{"px": "100", "sz": "0.1"}] + [{"px": "99.9", "sz": "50"}] * 5   # ~$25k behind a $10 touch
    asks = [{"px": "100.1", "sz": "0.1"}] + [{"px": "100.2", "sz": "60"}] * 5
    depth = HLClient.top_depth_usd({"levels": [bids, asks]})
    assert depth > 20_000        # near-touch depth reflects the real book, not the $10 best level


def _guard(market, hl):
    eng = RiskEngine(_config(), hl, FakeStore(), FakePM())
    return asyncio.run(eng.liquidity_guard(market))


def test_liquidity_guard_rejects_wide_spread():
    reason = _guard(_market(), FakeHLBook(book=_book(96.0, 104.0)))  # 8% spread on xyz
    assert reason and "spread" in reason


def test_liquidity_guard_allows_tight_spread():
    assert _guard(_market(), FakeHLBook(book=_book(99.95, 100.05))) is None


def test_liquidity_guard_skips_non_listed_dex():
    crypto = Market("BTC", "BTC", "", "crypto", 2, 5)  # "" not in spread_guard_dexes
    assert _guard(crypto, FakeHLBook(book=_book(90.0, 110.0))) is None


def test_liquidity_guard_fails_open_on_error():
    assert _guard(_market(), FakeHLBook(raise_book=True)) is None       # book error -> allow
    assert _guard(_market(), FakeHL()) is None                          # no l2_book method -> allow


def test_liquidity_guard_min_depth():
    c = _config()
    c.app.risk.min_top_depth_usd = 1_000_000.0   # require deep book
    eng = RiskEngine(c, FakeHLBook(book=_book(99.95, 100.05, 1.0, 1.0)), FakeStore(), FakePM())
    reason = asyncio.run(eng.liquidity_guard(_market()))
    assert reason and "depth" in reason


def test_evaluate_rejects_thin_book_entry():
    eng = RiskEngine(_config(), FakeHLBook(book=_book(96.0, 104.0)), FakeStore(), FakePM())
    d = _run(eng, _news(), _analysis(conf=0.95), _market())
    assert d.action == "reject" and "spread" in d.reason


# ---- indirect-mention de-prioritization ----------------------------------
def test_adjust_confidence_indirect_penalty():
    r = Config().app.risk
    mkt = _market()  # equity -> no liquidity/new-listing penalty
    conf, notes = adjust_confidence(0.90, mkt, "Alphabet capex up", r, mentioned=False)
    assert abs(conf - (0.90 - r.indirect_mention_penalty)) < 1e-9
    assert any("indirect" in n for n in notes)
    conf2, notes2 = adjust_confidence(0.90, mkt, "Marvell $MRVL", r, mentioned=True)
    assert conf2 == 0.90 and not any("indirect" in n for n in notes2)


def test_evaluate_indirect_mention_drops_below_gate():
    eng = RiskEngine(_config(), FakeHL(100.0), FakeStore(), FakePM(),
                     universe=FakeUniverse(mentioned=False))
    d = _run(eng, _news(), _analysis(conf=0.82), _market())  # 0.82 - 0.15 = 0.67 < gate
    assert d.action == "reject" and "indirect" in d.reason


def test_evaluate_direct_mention_no_penalty():
    eng = RiskEngine(_config(), FakeHL(100.0), FakeStore(), FakePM(),
                     universe=FakeUniverse(mentioned=True))
    d = _run(eng, _news(), _analysis(conf=0.82), _market())
    assert d.action == "enter"


def test_evaluate_analyzer_direct_overrides_failed_regex():
    # Analyzer flags the asset as the news SUBJECT (e.g. "SpaceX" -> SPCX) even though the
    # regex direct-mention check fails -> no indirect haircut, so 0.82 clears the gate.
    eng = RiskEngine(_config(), FakeHL(100.0), FakeStore(), FakePM(),
                     universe=FakeUniverse(mentioned=False))
    d = _run(eng, _news(), _analysis(conf=0.82, relation="direct"), _market())
    assert d.action == "enter"


# ---- per-symbol penalty --------------------------------------------------
def test_symbol_penalty_applies_to_named_ticker():
    r = Config().app.risk
    r.symbol_penalties = {"NVDA": 0.05}
    nvda = Market("xyz:NVDA", "NVDA", "xyz", "equity", 2, 5)
    conf, notes = adjust_confidence(0.90, nvda, "Nvidia beats", r)
    assert abs(conf - 0.85) < 1e-9 and any("NVDA" in n for n in notes)
    # a different ticker is untouched
    conf2, notes2 = adjust_confidence(0.90, _market(), "Marvell news", r)
    assert conf2 == 0.90 and not any("NVDA" in n for n in notes2)


# --- arena: per-lane daily-loss halt isolation ------------------------------------------------

def test_per_lane_daily_halt_isolates():
    """One lane hitting its daily-loss limit must NOT freeze the other lanes' wallets, and must
    NOT set the global kill switch."""
    c = _config()
    rt = c.runtime
    a = RiskEngine(c, FakeHL(100.0), FakeStore(pnl=-c.app.risk.daily_loss_limit_usd - 1), FakePM())
    a.model_id = "modelA"
    b = RiskEngine(c, FakeHL(100.0), FakeStore(pnl=0.0), FakePM())
    b.model_id = "modelB"

    da = _run(a, _news(), _analysis(), _market())
    assert da.action == "reject" and "daily loss" in da.reason
    assert not rt.trading_halted                       # global kill must NOT be tripped
    assert rt.is_daily_halted("modelA")[0] is True
    assert rt.is_daily_halted("modelB")[0] is False

    # lane A stays halted on the next eval (pre-loss-check short-circuit); lane B trades fine
    da2 = _run(a, _news(), _analysis(), _market())
    assert da2.action == "reject" and "daily loss halt" in da2.reason
    db = _run(b, _news(), _analysis(conf=0.95), _market())
    assert db.action == "enter"


def test_manual_kill_is_global_across_lanes():
    """The manual kill switch stays global — it halts every lane, then resume() clears all
    (global + every per-lane daily halt)."""
    c = _config()
    a = RiskEngine(c, FakeHL(100.0), FakeStore(), FakePM()); a.model_id = "modelA"
    b = RiskEngine(c, FakeHL(100.0), FakeStore(), FakePM()); b.model_id = "modelB"
    c.runtime.halt("manual kill")
    assert _run(a, _news(), _analysis(conf=0.95), _market()).action == "reject"
    assert _run(b, _news(), _analysis(conf=0.95), _market()).action == "reject"
    c.runtime.resume()
    assert _run(b, _news(), _analysis(conf=0.95), _market()).action == "enter"


def test_per_lane_halt_auto_expires_next_day():
    """A per-lane daily halt set on a prior UTC day auto-resumes."""
    c = _config()
    rt = c.runtime
    rt.halt_daily("loss", model_id="modelA")
    assert rt.is_daily_halted("modelA")[0] is True
    rt._model_daily["modelA"][0] = "1970-01-01"        # pretend it was set on a past day
    assert rt.is_daily_halted("modelA")[0] is False     # lazily expired
