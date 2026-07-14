"""Already-moved entry guard: reference price extraction, signed move, haircut/reject
policy (incl. the 'days' regime-change exemption), and fail-open behavior."""
from hlbot.trading.risk import apply_move_guard, ref_price_from_candles, signed_move_pct


def _c(t, o, h, l, cl):
    return {"t": t, "T": t + 60_000, "o": o, "h": h, "l": l, "c": cl}


class _R:
    pre_move_lookback_seconds = 180
    pre_move_haircut_pct = 0.0125
    pre_move_penalty = 0.05
    pre_move_reject_pct = 0.03


# ---- reference price -------------------------------------------------------------


def test_ref_price_is_last_close_before_news():
    candles = [_c(0, 100, 101, 99, 100.5), _c(60_000, 100.5, 102, 100, 101.5),
               _c(120_000, 101.5, 105, 101, 104)]
    # news at 130s: last bar ENDING at/before is bar 2 (T=120s) -> close 101.5
    assert ref_price_from_candles(candles, 130_000) == 101.5


def test_ref_price_falls_back_to_containing_bar_open():
    candles = [_c(120_000, 101.5, 105, 101, 104)]
    # news mid-bar with no earlier bar -> the bar's open predates the headline
    assert ref_price_from_candles(candles, 150_000) == 101.5


def test_ref_price_none_when_no_bar_predates_news():
    candles = [_c(120_000, 101.5, 105, 101, 104)]
    assert ref_price_from_candles(candles, 60_000) is None
    assert ref_price_from_candles([], 60_000) is None


# ---- signed move ------------------------------------------------------------------


def test_signed_move_direction_convention():
    assert signed_move_pct(100, 102, "long") > 0     # already ran up = bad for a long
    assert signed_move_pct(100, 102, "short") < 0    # ran up = better short entry
    assert signed_move_pct(100, 98, "short") > 0     # already dumped = bad for a short
    assert signed_move_pct(0, 102, "long") == 0.0    # garbage ref -> fail open


# ---- guard policy -----------------------------------------------------------------


def test_guard_no_adjustment_for_adverse_or_small_moves():
    conf, notes, reject = apply_move_guard(0.85, -0.02, "hours", _R())
    assert conf == 0.85 and not notes and reject is None
    conf, notes, reject = apply_move_guard(0.85, 0.01, "hours", _R())
    assert conf == 0.85 and not notes and reject is None   # below the 1.25% haircut


def test_guard_haircut_between_thresholds():
    conf, notes, reject = apply_move_guard(0.85, 0.02, "hours", _R())
    assert abs(conf - 0.80) < 1e-9 and reject is None
    assert notes and "already-moved" in notes[0]


def test_guard_rejects_big_move_for_short_horizons():
    for ts in ("immediate", "hours"):
        conf, _notes, reject = apply_move_guard(0.85, 0.035, ts, _R())
        assert reject is not None and "already moved" in reject
        assert conf == 0.85   # confidence untouched on reject path


def test_guard_days_is_haircut_only():
    # A 3.5% pop must NOT kill a regime-change catalyst — haircut instead.
    conf, notes, reject = apply_move_guard(0.85, 0.035, "days", _R())
    assert reject is None and abs(conf - 0.80) < 1e-9 and notes


def test_guard_disabled_when_thresholds_zero():
    class Off(_R):
        pre_move_haircut_pct = 0.0
        pre_move_reject_pct = 0.0
    conf, notes, reject = apply_move_guard(0.85, 0.10, "hours", Off())
    assert conf == 0.85 and not notes and reject is None
