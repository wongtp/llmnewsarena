"""Portfolio metrics: equity curve, drawdown, profit factor, holding/exposure stats."""
from dataclasses import dataclass

from hlbot.backtest.metrics import (
    equity_curve,
    exposure_curve,
    holding_stats,
    max_drawdown,
    profit_factor,
    summary_metrics,
)

H = 3_600_000
DAY = 24 * H


@dataclass
class _T:
    time_ms: int
    exit_ms: int
    pnl: float
    notional: float = 1000.0
    reason: str = "take profit"


def test_equity_curve_daily_cumulative():
    trades = [_T(0, 1 * H, 100.0), _T(2 * H, 3 * H, -40.0),       # day 1: +60
              _T(DAY, DAY + H, 25.0)]                              # day 2: +85 cumulative
    curve = equity_curve(trades)
    assert len(curve) == 2
    assert abs(curve[0][1] - 60.0) < 1e-9 and abs(curve[1][1] - 85.0) < 1e-9


def test_max_drawdown_peak_to_trough():
    curve = [("d1", 100.0), ("d2", 250.0), ("d3", 50.0), ("d4", 300.0)]
    assert abs(max_drawdown(curve) - 200.0) < 1e-9   # 250 -> 50


def test_max_drawdown_counts_initial_losing_streak():
    assert abs(max_drawdown([("d1", -120.0), ("d2", -80.0)]) - 120.0) < 1e-9


def test_profit_factor():
    trades = [_T(0, H, 100.0), _T(0, H, 100.0), _T(0, H, -50.0)]
    assert abs(profit_factor(trades) - 4.0) < 1e-9
    assert profit_factor([_T(0, H, 10.0)]) == float("inf")
    assert profit_factor([]) == 0.0


def test_holding_stats_by_reason():
    trades = [_T(0, 2 * H, 50.0, reason="take profit"),
              _T(0, 4 * H, -20.0, reason="stop loss"),
              _T(0, 6 * H, 30.0, reason="take profit")]
    h = holding_stats(trades)
    assert abs(h["median_hold_h"] - 4.0) < 1e-9
    tp = h["by_reason"]["take profit"]
    assert tp["n"] == 2 and tp["win_rate"] == 1.0 and abs(tp["total"] - 80.0) < 1e-9


def test_exposure_curve_peak_concurrency():
    # Two overlapping positions, then a third after both closed.
    trades = [_T(0, 5 * H, 0.0, notional=2500), _T(H, 3 * H, 0.0, notional=5000),
              _T(10 * H, 12 * H, 0.0, notional=7500)]
    e = exposure_curve(trades)
    assert e["peak_concurrent"] == 2 and abs(e["peak_notional"] - 7500.0) < 1e-9


def test_summary_metrics_shape_and_json_safety():
    m = summary_metrics([_T(0, H, 10.0)], account_size_usd=10_000)
    assert m["profit_factor"] is None              # inf -> None (JSON-safe)
    assert m["max_dd_usd"] == 0.0 and m["peak_concurrent"] == 1
    assert m["max_dd_pct_of_account"] == 0.0
    m2 = summary_metrics([], account_size_usd=0.0)
    assert m2["equity_curve"] == [] and m2["max_dd_pct_of_account"] is None
