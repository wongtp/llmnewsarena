"""Tests for the edge-report aggregation helpers (source expectancy + calibration)."""
from hlbot.backtest.edge import (
    calibration,
    excursion_stats,
    gate_sweep_oos,
    group_by,
    kelly_table,
    market_type,
    outcome_pnl,
    outcomes,
    source_key,
    suggest_gate,
    suggest_gate_oos,
    summarize,
)


def _rows():
    return [
        {"direction": "long", "confidence": 0.90, "status": "traded", "pnl": 120.0,
         "source": "Twitter", "title": "Marvell beat (@WuBlockchain)", "dex": "xyz",
         "ticker": "MRVL", "time_sensitivity": "days"},
        {"direction": "short", "confidence": 0.60, "status": "rejected", "would_pnl": -30.0,
         "source": "Telegram:trad_fin", "title": "trad_fin", "dex": "",
         "ticker": "BTC", "time_sensitivity": "hours"},
        # directional but no realized/would-be PnL -> excluded
        {"direction": "long", "confidence": 0.70, "status": "rejected", "source": "Twitter",
         "title": "x (@foo)", "dex": "", "ticker": "ETH"},
        # not directional -> excluded
        {"direction": "none", "confidence": 0.20, "status": "rejected", "ticker": None},
        # below min_conf 0.5 -> excluded
        {"direction": "long", "confidence": 0.40, "status": "rejected", "would_pnl": 5.0,
         "source": "Twitter", "title": "a (@bar)", "dex": ""},
    ]


def test_source_key_variants():
    assert source_key(_rows()[0]) == "@WuBlockchain"
    assert source_key(_rows()[1]) == "Telegram:trad_fin"
    assert source_key({"source": "Twitter", "title": "no handle here"}) == "Twitter"


def test_market_type_and_outcome_pnl():
    assert market_type({"dex": "xyz"}).startswith("xyz")
    assert market_type({"dex": ""}) == "crypto"
    assert outcome_pnl({"status": "traded", "pnl": 12.0}) == 12.0
    assert outcome_pnl({"status": "rejected", "would_pnl": -3.0}) == -3.0
    assert outcome_pnl({"status": "rejected"}) is None


def test_outcomes_filters_and_min_conf():
    items = outcomes(_rows(), min_conf=0.50)
    assert len(items) == 2                       # traded winner + skipped would-be only
    assert {o["ticker"] for o in items} == {"MRVL", "BTC"}


def test_summarize():
    s = summarize(outcomes(_rows(), 0.5))
    assert s["n"] == 2 and s["wins"] == 1 and s["traded"] == 1
    assert abs(s["total"] - 90.0) < 1e-9 and abs(s["avg"] - 45.0) < 1e-9
    assert abs(s["win_rate"] - 0.5) < 1e-9


def test_calibration_buckets():
    cal = calibration(outcomes(_rows(), 0.5), width=0.05, lo=0.5)
    ranges = [r for r, _ in cal]
    assert (0.60, 0.65) in [(round(a, 2), round(b, 2)) for a, b in ranges]
    assert (0.90, 0.95) in [(round(a, 2), round(b, 2)) for a, b in ranges]


def test_group_by_sorted_desc():
    g = group_by(outcomes(_rows(), 0.5), "source")
    assert g[0][0] == "@WuBlockchain"            # winner sorts first
    assert g[-1][0] == "Telegram:trad_fin"


def test_suggest_gate_prefers_excluding_loser():
    best = suggest_gate(outcomes(_rows(), 0.5), min_trades=1)
    assert abs(best["total"] - 120.0) < 1e-9     # drop the -30 loser, keep the +120 winner
    assert best["gate"] > 0.60


def test_summarize_profit_factor():
    s = summarize([{"pnl": 100.0, "traded": True}, {"pnl": -50.0, "traded": True}])
    assert abs(s["pf"] - 2.0) < 1e-9
    assert summarize([{"pnl": 10.0, "traded": True}])["pf"] == float("inf")
    assert summarize([])["pf"] == 0.0


def _timed(conf_pnl_ts):
    return [{"conf": c, "pnl": p, "time_ms": t, "traded": True}
            for c, p, t in conf_pnl_ts]


def test_suggest_gate_oos_fits_on_train_evaluates_on_val():
    # Train (first 70%): high-conf wins, low-conf loses -> gate lands above 0.7.
    # Validation keeps the same structure, so the gate should hold up.
    items = _timed([
        (0.9, 100, 1), (0.6, -50, 2), (0.9, 80, 3), (0.6, -40, 4), (0.9, 90, 5),
        (0.6, -30, 6), (0.9, 70, 7),
        (0.9, 60, 8), (0.6, -20, 9), (0.9, 50, 10),
    ])
    oos = suggest_gate_oos(items, train_frac=0.7, min_trades=2)
    assert oos and oos["gate"] > 0.6
    assert oos["val"]["total"] > 0 and oos["val"]["n"] == 2   # only the two 0.9 val reads


def test_suggest_gate_oos_too_small_returns_empty():
    assert suggest_gate_oos(_timed([(0.9, 10, 1)]), min_trades=5) == {}


def test_gate_sweep_oos_splits_chronologically():
    items = _timed([(0.9, 10, 1), (0.9, 10, 2), (0.9, -99, 3)])
    [(gate, train, val)] = gate_sweep_oos(items, [0.8], train_frac=0.67)
    assert train["n"] == 2 and val["n"] == 1 and val["total"] == -99


def test_excursion_stats_separates_winners_losers():
    items = [
        {"pnl": 50.0, "mae": 0.01, "mfe": 0.06, "traded": True},
        {"pnl": 40.0, "mae": 0.02, "mfe": 0.05, "traded": True},
        {"pnl": -30.0, "mae": 0.05, "mfe": 0.012, "traded": True},
        {"pnl": 10.0, "mae": None, "mfe": None, "traded": True},   # no excursions -> excluded
    ]
    s = excursion_stats(items)
    assert s["n_scored"] == 3
    assert s["winners_mae"]["n"] == 2 and s["losers_mfe"]["n"] == 1
    assert s["winners_mae"]["p50"] in (0.01, 0.02)
    assert abs(s["losers_mfe"]["p50"] - 0.012) < 1e-9


def _traded(conf, pnl, notional=5000.0):
    return {"status": "traded", "confidence": conf, "pnl": pnl, "notional": notional}


def test_kelly_table_suggests_only_at_min_n():
    tiers = [[0.78, 2500.0], [0.82, 5000.0]]
    # 0.78 tier: 25 trades, 60% win, +100/-50 -> f* = 0.6 - 0.4/2 = 0.40, suggestion set
    rows = [_traded(0.80, 100.0) for _ in range(15)] + [_traded(0.80, -50.0) for _ in range(10)]
    # 0.82 tier: only 3 trades -> no suggestion regardless of edge
    rows += [_traded(0.85, 200.0) for _ in range(3)]
    kt = kelly_table(rows, tiers, account_usd=10_000.0, max_notional=10_000.0, min_n=20)
    t78, t82 = kt
    assert t78["n"] == 25 and abs(t78["kelly"] - 0.40) < 1e-9
    # q-kelly: 0.10 * 10k = $1000 at risk; loss frac 50/5000 = 1% -> $100k, capped at max
    assert t78["suggest"] == 10_000.0
    assert t82["n"] == 3 and t82["suggest"] is None and t82["win"] == 1.0


def test_kelly_table_negative_edge_suggests_zero():
    tiers = [[0.78, 2500.0]]
    rows = [_traded(0.80, -100.0) for _ in range(15)] + [_traded(0.80, 50.0) for _ in range(10)]
    [t] = kelly_table(rows, tiers, account_usd=10_000.0, max_notional=10_000.0, min_n=20)
    assert t["kelly"] < 0 and t["suggest"] == 0.0
