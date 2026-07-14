"""Walk-forward optimizer: splits, evaluation, stability flags, and the
engine-faithful per-entry simulation."""
from hlbot.backtest.optimizer import (
    combo_exit,
    evaluate,
    grid_combos,
    max_horizon_s,
    simulate_entry,
    split_holdout,
    split_rolling,
    stability,
    sweep,
)


class _R:
    take_profit_pct = 0.03
    exit_horizons = {"immediate": 3600, "hours": 21600, "days": 259200}
    dry_run_slippage_pct = 0.0
    backtest_slippage_pct = None
    backtest_stop_slippage_pct = 0.0
    base_notional_usd = 1000.0
    size_tiers = [[0.8, 1000.0]]
    max_notional_usd = 1000.0
    premarket_symbols = []
    premarket_size_factor = 1.0


class _Mkt:
    symbol = "TEST"
    name = "TEST"


def _entries(n, start_ms=0, step_ms=1000):
    return [{"time_ms": start_ms + i * step_ms, "side": "long", "conf": 0.85,
             "time_sensitivity": "hours", "market": _Mkt()} for i in range(n)]


def _c(t, o, h, l, cl):
    return {"t": t, "T": t + 60_000, "o": o, "h": h, "l": l, "c": cl}


# ---- splits ---------------------------------------------------------------------


def test_split_holdout_is_chronological():
    ents = _entries(10)
    ents.reverse()   # input order scrambled; split must follow time_ms
    [(train, val)] = split_holdout(ents, train_frac=0.7)
    t_max = max(ents[i]["time_ms"] for i in train)
    v_min = min(ents[i]["time_ms"] for i in val)
    assert len(train) == 7 and len(val) == 3 and t_max < v_min


def test_split_rolling_validates_each_trade_once():
    ents = _entries(10)
    folds = split_rolling(ents, n_folds=4)
    assert len(folds) == 4
    seen = [i for _, val in folds for i in val]
    assert sorted(seen) == sorted(set(seen))   # no index validated twice
    for train, val in folds:
        assert max(ents[i]["time_ms"] for i in train) < min(ents[i]["time_ms"] for i in val)


def test_split_rolling_tiny_set_falls_back_to_holdout():
    folds = split_rolling(_entries(3), n_folds=4)
    assert len(folds) == 1


# ---- evaluate / stability --------------------------------------------------------


def test_evaluate_flags_train_only_winner():
    # Combo A: one huge early (train-period) win, validation flat-negative.
    # Combo B: steady small wins everywhere -> must outrank A on validation.
    results = {
        ("A",): [100.0, 100.0, 100.0, -10.0, -10.0, -10.0],
        ("B",): [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }
    splits = [([0, 1, 2], [3, 4, 5])]
    rows = evaluate(results, splits)
    assert rows[0]["combo"] == ("B",)          # validation winner first
    a = next(x for x in rows if x["combo"] == ("A",))
    assert a["val"] < 0 and a["rank_shift"] > 0   # looked best in train, fell on val


def test_evaluate_handles_none_pnls():
    results = {("A",): [None, 5.0, None, 5.0]}
    rows = evaluate(results, [([0, 1], [2, 3])])
    assert rows[0]["val"] == 5.0 and rows[0]["val_n"] == 1


def test_stability_flags_spike():
    grid = {"stops": [0.03, 0.05, 0.07], "trails": [0.08], "holds": [72]}
    results = {
        (0.03, 0.08, 72): [10.0],
        (0.05, 0.08, 72): [500.0],   # lucky spike: neighbors are 10 and 12
        (0.07, 0.08, 72): [12.0],
    }
    stab = stability(results, grid)
    assert stab[(0.05, 0.08, 72)]["spike"] > 400
    assert abs(stab[(0.03, 0.08, 72)]["spike"]) < 500   # edge combo compared to its one neighbor


# ---- simulation parity -----------------------------------------------------------


def test_simulate_entry_no_candles_is_none():
    e = {**_entries(1)[0], "_candles": []}
    assert simulate_entry(e, 0.03, 0.08, 72, _R()) is None


def test_simulate_entry_take_profit_path():
    # immediate => fixed 3% TP; candle pops 4% -> exit at 103, pnl > 0 net of fees
    e = {"time_ms": 0, "side": "long", "conf": 0.85, "time_sensitivity": "immediate",
         "market": _Mkt(), "_candles": [_c(0, 100, 104, 99.5, 103)], "_frates": None}
    pnl = simulate_entry(e, 0.03, 0.08, 72, _R())
    assert pnl is not None and 25 < pnl < 30   # 3% on $1000 minus ~0.09% fees


def test_simulate_entry_funding_subtracted():
    e = {"time_ms": 0, "side": "long", "conf": 0.85, "time_sensitivity": "days",
         "market": _Mkt(),
         # flat tape spanning 5h: data ends before the 72h horizon -> hold > 2h funding floor
         "_candles": [_c(k * 3600_000, 100, 100.5, 99.5, 100) for k in range(5)],
         "_frates": [(3 * 3600_000, 0.001)]}   # one 0.1% hourly print inside the hold
    base = simulate_entry({**e, "_frates": None}, 0.03, 0.0, 72, _R())
    with_funding = simulate_entry(e, 0.03, 0.0, 72, _R())
    assert with_funding is not None and base is not None
    assert abs((base - with_funding) - 1.0) < 1e-9   # $1000 * 0.001 = $1 of carry


def test_simulate_entry_breakeven_changes_exit():
    # days/no-trail: pops +2.5% (arms), fades to entry, then to the -3% stop. With
    # breakeven the exit is ~entry (small fee loss); without, the full stop loss.
    e = {"time_ms": 0, "side": "long", "conf": 0.85, "time_sensitivity": "days",
         "market": _Mkt(),
         "_candles": [_c(0, 100, 102.5, 100, 102), _c(60_000, 102, 102, 99.8, 99.9),
                      _c(120_000, 99.9, 99.9, 96.5, 97)],
         "_frates": None}
    on = simulate_entry(e, 0.03, 0.0, 72, _R(), be_arm=0.02, be_offset=0.0)
    off = simulate_entry(e, 0.03, 0.0, 72, _R())
    assert on is not None and off is not None
    assert off < -25 < on < 0    # off: ~-3% of $1000; on: ~fees only


def test_sweep_breakeven_includes_off_baseline():
    from hlbot.backtest.optimizer import sweep_breakeven
    e = {"time_ms": 0, "side": "long", "conf": 0.85, "time_sensitivity": "days",
         "market": _Mkt(), "_candles": [_c(0, 100, 101, 99.5, 100.5)], "_frates": None}
    grid = {"arms": [0.0, 0.02], "offsets": [0.0]}
    out = sweep_breakeven([e], grid, _R(), 0.03, 0.0, 72)
    assert set(out) == {(0.0, 0.0), (0.02, 0.0)}
    assert all(len(v) == 1 for v in out.values())


def test_combo_exit_immediate_keeps_fixed_tp():
    horizon, stop, tp, trail = combo_exit("immediate", 0.05, 0.08, 96, _R())
    assert horizon == 3600 and tp == _R.take_profit_pct and trail == 0.0


def test_max_horizon_and_grid():
    assert max_horizon_s("days", [48, 96]) == 96 * 3600
    assert max_horizon_s("hours", [48, 96]) == 6 * 3600
    grid = {"stops": [0.03], "trails": [0.05, 0.08], "holds": [72]}
    assert len(grid_combos(grid)) == 2
    res = sweep([], grid, _R())
    assert set(res) == {(0.03, 0.05, 72), (0.03, 0.08, 72)}
