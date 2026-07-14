import asyncio
import types

from hlbot.backtest import engine as bt_engine
from hlbot.backtest.engine import fetch_candles, pick_entry, pnl_usd, walk_candles


def _c(t, o, h, l, cl):
    return {"t": t, "T": t + 60_000, "o": o, "h": h, "l": l, "c": cl}


async def _fast_sleep(*_a, **_k):
    return None


def _fake_hl(snap):
    return types.SimpleNamespace(info=types.SimpleNamespace(candles_snapshot=snap))


def test_fetch_candles_retries_transient_error(monkeypatch, tmp_path):
    monkeypatch.setattr(bt_engine.asyncio, "sleep", _fast_sleep)  # skip backoff
    monkeypatch.setattr(bt_engine, "CANDLE_CACHE_DIR", tmp_path)  # isolate disk cache
    calls = {"n": 0}

    def snap(name, interval, s, e):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient blip")
        return [{"t": 0}]

    out = asyncio.run(fetch_candles(_fake_hl(snap), "BTC", 0, 1000, "1m"))
    assert out == [{"t": 0}] and calls["n"] == 3   # recovered instead of dropping the trade
    # past window -> cached on disk; a repeat call must not hit the API at all
    out2 = asyncio.run(fetch_candles(_fake_hl(snap), "BTC", 0, 1000, "1m"))
    assert out2 == [{"t": 0}] and calls["n"] == 3


def test_fetch_candles_exhausts_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(bt_engine.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(bt_engine, "CANDLE_CACHE_DIR", tmp_path)

    def snap(name, interval, s, e):
        raise RuntimeError("down")

    assert asyncio.run(fetch_candles(_fake_hl(snap), "BTC", 0, 1000, "1m", retries=2)) == []
    assert not list(tmp_path.iterdir())   # failures must never be cached


def test_fetch_candles_empty_not_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(bt_engine, "CANDLE_CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def snap(name, interval, s, e):
        calls["n"] += 1
        return []   # genuine no-data (e.g. not listed yet) must NOT trigger retries

    assert asyncio.run(fetch_candles(_fake_hl(snap), "BTC", 0, 1000, "1m")) == []
    assert calls["n"] == 1


def test_analysis_cache_tolerates_schema_drift(tmp_path):
    # A cache written by another checkout may carry fields Analysis lacks (and miss
    # fields it has) — loading must drop unknowns and default the missing.
    import json as _json

    p = tmp_path / "cache.json"
    p.write_text(_json.dumps({"n1": {
        "news_id": "n1", "ticker": "BTC", "asset_class": "crypto", "direction": "long",
        "confidence": 0.9, "time_sensitivity": "days", "is_stale": False,
        "rationale": "x", "model": "m", "error": None,
        "related_tickers": ["ETH"], "latency_ms": 1234, "cost_usd": 0.01}}))
    a = bt_engine.AnalysisCache(str(p)).get("n1")
    assert a is not None and a.ticker == "BTC" and a.subject_relation == "derived"


def test_fetch_candles_future_window_not_cached(monkeypatch, tmp_path):
    import time as _time

    monkeypatch.setattr(bt_engine, "CANDLE_CACHE_DIR", tmp_path)
    future_end = int((_time.time() + 3600) * 1000)   # window still open -> partial data

    def snap(name, interval, s, e):
        return [{"t": 0}]

    asyncio.run(fetch_candles(_fake_hl(snap), "BTC", 0, future_end, "1m"))
    assert not list(tmp_path.iterdir())


def test_pick_entry_uses_first_bar_at_or_after_news():
    candles = [_c(0, 10, 11, 9, 10), _c(60_000, 10, 11, 9, 10), _c(120_000, 12, 13, 11, 12)]
    idx, px = pick_entry(candles, 60_000)
    assert idx == 1 and px == 10
    idx2, px2 = pick_entry(candles, 70_000)   # mid-bar -> next bar open
    assert idx2 == 2 and px2 == 12


def test_pick_entry_rejects_entry_too_far_after_news():
    # Asset wasn't trading at news time; first candle is 2 days later (e.g. a later listing).
    listing = 2 * 86_400_000
    candles = [_c(listing, 100, 110, 95, 105), _c(listing + 60_000, 105, 112, 100, 108)]
    idx, px = pick_entry(candles, 0, max_delay_ms=7_200_000)   # 2h tolerance
    assert idx is None and px is None
    # Same data, but news right before the candle -> valid entry.
    idx2, px2 = pick_entry(candles, listing - 30_000, max_delay_ms=7_200_000)
    assert idx2 == 0 and px2 == 100


def test_walk_long_take_profit():
    candles = [_c(0, 100, 104, 99, 103)]   # high 104 >= tp 103
    w = walk_candles("long", 100, 98.5, 103, 0.0, candles, 0, 10**13)
    assert w.reason == "take profit" and w.exit_px == 103


def test_walk_long_stop_loss_priority():
    candles = [_c(0, 100, 104, 98, 103)]   # both tp(103) and sl(98.5) inside -> SL first
    w = walk_candles("long", 100, 98.5, 103, 0.0, candles, 0, 10**13)
    assert w.reason == "stop loss" and w.exit_px == 98.5


def test_walk_short_take_profit():
    candles = [_c(0, 100, 100.5, 96, 97)]  # low 96 <= tp 97 (short profits on drop)
    w = walk_candles("short", 100, 101.5, 97, 0.0, candles, 0, 10**13)
    assert w.reason == "take profit" and w.exit_px == 97


def test_walk_time_exit():
    candles = [_c(0, 100, 100.4, 99.6, 100.2)]  # nothing hit; horizon passed
    w = walk_candles("long", 100, 98.5, 103, 0.0, candles, 0, 0)
    assert w.reason == "time exit" and w.exit_px == 100.2


def test_walk_long_trailing_stop():
    # rises to 110 (peak), then pulls back; trail 3% -> stop at 110*0.97=106.7
    candles = [_c(0, 100, 110, 100, 109), _c(60_000, 109, 109, 106, 106)]
    w = walk_candles("long", 100, 95, 0.0, 0.03, candles, 0, 10**13)
    assert w.reason == "trailing stop" and round(w.exit_px, 2) == 106.7


def test_walk_short_trailing_stop():
    # falls to 90 (best), then bounces; trail 3% -> stop at 90*1.03=92.7
    candles = [_c(0, 100, 100, 90, 91), _c(60_000, 91, 93, 91, 93)]
    w = walk_candles("short", 100, 105, 0.0, 0.03, candles, 0, 10**13)
    assert w.reason == "trailing stop" and round(w.exit_px, 2) == 92.7


def test_walk_long_breakeven_stop():
    # Pops to 102.5 (arms at +2%), then fades back through entry: exit AT entry, not -5% stop.
    candles = [_c(0, 100, 102.5, 100, 102), _c(60_000, 102, 102, 99.5, 99.6)]
    w = walk_candles("long", 100, 95, 0.0, 0.0, candles, 0, 10**13,
                     breakeven_arm_pct=0.02, breakeven_offset_pct=0.0)
    assert w.reason == "breakeven stop" and w.exit_px == 100.0
    # Same path with the feature off rides down to the hard stop.
    w_off = walk_candles("long", 100, 95, 0.0, 0.0,
                         candles + [_c(120_000, 99.5, 99.5, 94, 94.5)], 0, 10**13)
    assert w_off.reason == "stop loss" and w_off.exit_px == 95


def test_walk_short_breakeven_stop_with_offset():
    # Drops to 97.5 (arms at +2% favorable), bounces; offset 0.2% locks exit at 99.8.
    candles = [_c(0, 100, 100, 97.5, 98), _c(60_000, 98, 100.5, 98, 100.4)]
    w = walk_candles("short", 100, 105, 0.0, 0.0, candles, 0, 10**13,
                     breakeven_arm_pct=0.02, breakeven_offset_pct=0.002)
    assert w.reason == "breakeven stop" and round(w.exit_px, 2) == 99.8


def test_walk_breakeven_not_armed_by_same_bar_spike():
    # Spike to +2.5% and fade happen in the SAME bar: peak updates after checks, so the
    # bar's own fade can't exit at breakeven (no intra-bar arm — pessimistic). The NEXT
    # bar arms off the recorded peak and, opening already below the floor, gap-fills at
    # its open (99.6) — not at entry (live polling would have got ~100) and not at the
    # hard stop.
    candles = [_c(0, 100, 102.5, 99.5, 99.6), _c(60_000, 99.6, 99.9, 95, 95)]
    w = walk_candles("long", 100, 95, 0.0, 0.0, candles, 0, 10**13,
                     breakeven_arm_pct=0.02, breakeven_offset_pct=0.0)
    assert w.reason == "breakeven stop (gap)" and w.exit_px == 99.6
    # Within the spike bar itself nothing fires: the position survives bar 1.
    w1 = walk_candles("long", 100, 95, 0.0, 0.0, candles[:1], 0, 10**13)
    assert w1.reason == "time exit (data end)"


def test_walk_trailing_beats_breakeven_after_big_run():
    # After +11% the 3% trail (106.7) sits above the breakeven floor (100) -> trailing wins.
    candles = [_c(0, 100, 110, 100, 109), _c(60_000, 109, 109, 106, 106)]
    w = walk_candles("long", 100, 95, 0.0, 0.03, candles, 0, 10**13,
                     breakeven_arm_pct=0.02, breakeven_offset_pct=0.0)
    assert w.reason == "trailing stop" and round(w.exit_px, 2) == 106.7


def test_walk_breakeven_gap_fills_at_open():
    # Armed on bar 1; bar 2 OPENS below the breakeven floor -> fill at the open.
    candles = [_c(0, 100, 102.5, 100, 102), _c(60_000, 98, 99, 97, 98.5)]
    w = walk_candles("long", 100, 95, 0.0, 0.0, candles, 0, 10**13,
                     breakeven_arm_pct=0.02, breakeven_offset_pct=0.0)
    assert w.reason == "breakeven stop (gap)" and w.exit_px == 98


def test_walk_long_gap_through_stop_fills_at_open():
    # 2nd bar OPENS at 95, below the 98.5 stop: a stop order can't fill at 98.5 — fill = open.
    candles = [_c(0, 100, 101, 99, 100), _c(60_000, 95, 96, 94, 95)]
    w = walk_candles("long", 100, 98.5, 0.0, 0.0, candles, 0, 10**13)
    assert w.reason == "stop loss (gap)" and w.exit_px == 95


def test_walk_short_gap_through_stop_fills_at_open():
    candles = [_c(0, 100, 101, 99, 100), _c(60_000, 106, 107, 105, 106)]
    w = walk_candles("short", 100, 101.5, 0.0, 0.0, candles, 0, 10**13)
    assert w.reason == "stop loss (gap)" and w.exit_px == 106


def test_walk_entry_bar_cannot_gap_stop():
    # The entry bar opens AT the entry price, which is above the stop by construction.
    candles = [_c(0, 100, 104, 99, 103)]
    w = walk_candles("long", 100, 98.5, 103, 0.0, candles, 0, 10**13)
    assert "gap" not in w.reason


def test_walk_mae_mfe_dip_then_rip():
    # Long dips to 97 (MAE 3%) then runs to 110 (MFE 10%) before the time exit on bar 2.
    candles = [_c(0, 100, 101, 97, 99), _c(60_000, 99, 110, 99, 108)]
    w = walk_candles("long", 100, 90, 0.0, 0.0, candles, 0, 120_000)
    assert w.reason == "time exit"
    assert round(w.mae_pct, 4) == 0.03 and round(w.mfe_pct, 4) == 0.10
    assert w.time_to_peak_ms == 120_000   # 2nd bar's close time minus entry bar open (0)


def test_walk_mae_mfe_pop_then_stop():
    # Long pops to 106 first, then collapses through the stop: MFE 6%, MAE >= stop distance.
    candles = [_c(0, 100, 106, 99, 105), _c(60_000, 105, 105, 95, 95)]
    w = walk_candles("long", 100, 97, 0.0, 0.0, candles, 0, 10**13)
    assert w.reason == "stop loss" and round(w.mfe_pct, 4) == 0.06
    assert w.mae_pct >= 0.03   # full exit-bar low counted (upper-bound convention)


def test_walk_short_mae_mfe_signs():
    # Short: adverse = highs above entry, favorable = lows below entry.
    candles = [_c(0, 100, 103, 96, 97)]
    w = walk_candles("short", 100, 105, 96.5, 0.0, candles, 0, 10**13)
    assert w.reason == "take profit"
    assert round(w.mae_pct, 4) == 0.03 and round(w.mfe_pct, 4) == 0.04


def test_pnl_sign():
    assert pnl_usd("short", 100, 90, 1.0) > 0    # short + price down = profit
    assert pnl_usd("long", 100, 90, 1.0) < 0


# ---- backtest slippage + funding ------------------------------------------------


class _R:
    dry_run_slippage_pct = 0.001
    backtest_slippage_pct = None
    backtest_stop_slippage_pct = 0.002


def test_bt_slippage_defaults_to_dry_run_model():
    from hlbot.backtest.engine import bt_slippage_pct
    assert bt_slippage_pct(_R()) == 0.001
    r = _R(); r.backtest_slippage_pct = 0.0
    assert bt_slippage_pct(r) == 0.0   # explicit 0 disables (not "fall back")


def test_slip_entry_and_exit_are_adverse():
    from hlbot.backtest.engine import slip_entry, slip_exit
    r = _R()
    assert slip_entry("long", 100, r) > 100 and slip_entry("short", 100, r) < 100
    assert slip_exit("long", 100, "take profit", r) < 100
    assert slip_exit("short", 100, "take profit", r) > 100


def test_slip_exit_stop_pays_extra():
    from hlbot.backtest.engine import slip_exit
    r = _R()
    tp = slip_exit("long", 100, "take profit", r)
    sl = slip_exit("long", 100, "stop loss (gap)", r)
    assert sl < tp   # stop exits modelled worse than discretionary closes


def test_funding_sign_long_pays_short_receives():
    from hlbot.backtest.engine import funding_usd_from_rates
    rates = [(3_600_000, 0.0001), (7_200_000, 0.0001)]   # two positive hourly prints
    long_cost = funding_usd_from_rates(rates, "long", 10_000, 0, 8_000_000)
    short_cost = funding_usd_from_rates(rates, "short", 10_000, 0, 8_000_000)
    assert long_cost == 2.0 and short_cost == -2.0   # + = cost, - = received


def test_funding_only_counts_events_inside_hold():
    from hlbot.backtest.engine import funding_usd_from_rates
    rates = [(1_000, 0.5), (3_600_000, 0.0001), (9_999_999, 0.5)]
    cost = funding_usd_from_rates(rates, "long", 10_000, 2_000, 8_000_000)
    assert cost == 1.0   # only the 1h print inside (start, end] counts
