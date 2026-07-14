"""Pure exit-parameter sweep + walk-forward evaluation over a FIXED entry set.

Extracted from scripts/optimize.py so the sweep logic is unit-testable and the
chronological train/validation discipline lives in one place. The core trick:
``sweep()`` computes a per-entry PnL VECTOR once per combo, so any chronological
split afterwards is free index slicing — k-fold walk-forward costs nothing beyond
the single full sweep (pure CPU over prefetched candles, $0 of tokens).

In-sample sweeps overfit: the previous optimizer maximized total PnL on the exact
trades it was fit on. ``evaluate()`` reports each combo's PnL on held-out later
data (holdout or rolling-origin folds), and ``stability()`` flags combos whose PnL
collapses one grid-step away (a lucky spike, not a plateau). Pick settings that
win on VALIDATION and sit on a plateau — never the in-sample top line.
"""
from __future__ import annotations

import itertools
from typing import Optional

from ..trading.risk import compute_sl_tp, premarket_factor, scale_notional
from .engine import (
    BACKTEST_MAX_ENTRY_DELAY_S,
    FUNDING_MIN_HOLD_MS,
    funding_usd_from_rates,
    pick_entry,
    pnl_usd,
    slip_entry,
    slip_exit,
    walk_candles,
)

# Default sweep grid (scripts/optimize.py can override via CLI in future; edit here).
DEFAULT_GRID = {
    "stops": [0.03, 0.05, 0.07],          # stop-loss %, applied to all horizons
    "trails": [0.05, 0.065, 0.08, 0.10],  # trailing % for hours/days
    "holds": [48, 72, 96],                # hold cap (hours) for "days" catalysts
}

# Breakeven-stop sweep: arm thresholds x lock-in offsets, swept ON TOP of the current
# exit config (one new mechanism, few knobs). arm 0.0 = feature off — the baseline row
# every other row must beat ON VALIDATION to justify shipping.
BREAKEVEN_GRID = {
    "arms": [0.0, 0.0075, 0.01, 0.015, 0.02, 0.03],
    "offsets": [0.0, 0.002],
}


def combo_exit(ts: str, stop: float, trail: float, days_h: int, r):
    """(horizon_seconds, stop_pct, tp_pct, trail_pct) for a sensitivity under a combo.
    Only the "days" horizon is swept; hours/immediate come from the live config so a
    config change can't silently diverge the sweep from production exits."""
    if ts == "days":
        return days_h * 3600, stop, (0.0 if trail > 0 else r.take_profit_pct), trail
    if ts == "hours":
        return r.exit_horizons.get("hours", 21600), stop, \
            (0.0 if trail > 0 else r.take_profit_pct), trail
    return r.exit_horizons.get("immediate", 3600), stop, r.take_profit_pct, 0.0  # immediate: fixed TP


def max_horizon_s(ts: str, holds: list[int], r=None) -> int:
    """Longest horizon any combo in the grid can need (sizes the candle/funding prefetch)."""
    if ts == "days":
        return max(holds) * 3600
    if r is not None:
        return r.exit_horizons.get(ts, 6 * 3600 if ts == "hours" else 3600)
    return 6 * 3600 if ts == "hours" else 3600


def simulate_entry(e: dict, stop: float, trail: float, days_h: int, r,
                   be_arm: float = 0.0, be_offset: float = 0.0) -> Optional[float]:
    """PnL of one fixed entry under one exit combo. Mirrors engine.simulate exactly:
    adverse entry/exit slippage, SL/TP anchored to the slipped fill, funding subtracted
    from the prefetched rate series (e['_frates']; None = unavailable -> cost 0).
    Returns None when the entry isn't simulable (no candle at news time)."""
    candles = e.get("_candles")
    if not candles:
        return None
    ts, side, news_ms = e["time_sensitivity"], e["side"], e["time_ms"]
    horizon_s, stop_pct, tp_pct, trail_pct = combo_exit(ts, stop, trail, days_h, r)
    idx, raw_entry = pick_entry(candles, news_ms,
                                max_delay_ms=BACKTEST_MAX_ENTRY_DELAY_S * 1000)
    if idx is None or not raw_entry:
        return None
    entry_px = slip_entry(side, raw_entry, r)
    sl, tp = compute_sl_tp(side, entry_px, stop_pct, tp_pct)
    w = walk_candles(side, entry_px, sl, tp, trail_pct, candles, idx,
                     news_ms + horizon_s * 1000,
                     breakeven_arm_pct=be_arm, breakeven_offset_pct=be_offset)
    exit_px = slip_exit(side, w.exit_px, w.reason, r)
    # Prefer the notional the replay ACTUALLY traded (carries the exposure clamp); only
    # recompute from tiers when the entry dict predates that field.
    notional = e.get("notional") or (scale_notional(e["conf"], r)
                                     * premarket_factor(e["market"], r))
    size = notional / entry_px
    pnl = pnl_usd(side, entry_px, exit_px, size)
    rates = e.get("_frates")
    if rates and w.exit_ms - news_ms > FUNDING_MIN_HOLD_MS:
        pnl -= funding_usd_from_rates(rates, side, notional, news_ms, w.exit_ms)
    return pnl


def grid_combos(grid: dict) -> list[tuple]:
    return list(itertools.product(grid["stops"], grid["trails"], grid["holds"]))


def sweep(entries: list[dict], grid: dict, r) -> dict[tuple, list[Optional[float]]]:
    """(stop, trail, hold) -> per-entry PnL vector (None where not simulable). Entries
    keep their input order; splits index into these vectors."""
    return {combo: [simulate_entry(e, *combo, r) for e in entries]
            for combo in grid_combos(grid)}


def sweep_breakeven(entries: list[dict], grid: dict, r,
                    stop: float, trail: float, days_h: int) -> dict[tuple, list]:
    """(arm, offset) -> per-entry PnL vector, with stop/trail/hold FIXED (the current
    config) — isolates the breakeven mechanism instead of co-fitting it with exits."""
    combos = list(itertools.product(sorted(set(grid["arms"])),
                                    sorted(set(grid["offsets"]))))
    return {c: [simulate_entry(e, stop, trail, days_h, r, be_arm=c[0], be_offset=c[1])
                for e in entries] for c in combos}


def _order(entries: list[dict]) -> list[int]:
    return sorted(range(len(entries)), key=lambda i: entries[i]["time_ms"])


def split_holdout(entries: list[dict], train_frac: float = 0.7) -> list[tuple[list, list]]:
    """One chronological split: fit on the first train_frac, validate on the rest."""
    order = _order(entries)
    k = max(1, min(len(order) - 1, int(round(len(order) * train_frac))))
    return [(order[:k], order[k:])]


def split_rolling(entries: list[dict], n_folds: int = 4) -> list[tuple[list, list]]:
    """Rolling-origin folds: chop the timeline into n_folds+1 chunks; fold i validates
    on chunk i+1 having trained on everything before it. Every trade after the first
    chunk gets validated exactly once."""
    order = _order(entries)
    n = len(order)
    if n < n_folds + 1:
        return split_holdout(entries)
    edges = [round(j * n / (n_folds + 1)) for j in range(n_folds + 2)]
    return [(order[:edges[i + 1]], order[edges[i + 1]:edges[i + 2]])
            for i in range(n_folds)]


def _tot(pnls: list[Optional[float]], idx: list[int]) -> tuple[float, int]:
    vals = [pnls[i] for i in idx if pnls[i] is not None]
    return sum(vals), len(vals)


def evaluate(results: dict[tuple, list], splits: list[tuple[list, list]]) -> list[dict]:
    """Per combo: train PnL (mean over folds), validation PnL (mean + worst fold),
    fraction of folds with positive validation PnL, and the in-sample-vs-validation
    rank shift (a big positive shift = the combo looked good only in-sample)."""
    rows = []
    for combo, pnls in results.items():
        train_tots, val_tots, val_n = [], [], 0
        for train_idx, val_idx in splits:
            tt, _ = _tot(pnls, train_idx)
            vt, vn = _tot(pnls, val_idx)
            train_tots.append(tt)
            val_tots.append(vt)
            val_n += vn
        nf = max(1, len(splits))
        rows.append({
            "combo": combo,
            "train": sum(train_tots) / nf,
            "val": sum(val_tots) / nf,
            "val_worst": min(val_tots) if val_tots else 0.0,
            "val_pos_frac": sum(1 for v in val_tots if v > 0) / nf,
            "val_n": val_n,
            "total": _tot(pnls, list(range(len(pnls))))[0],
        })
    by_train = sorted(rows, key=lambda x: -x["train"])
    by_val = sorted(rows, key=lambda x: -x["val"])
    train_rank = {id(x): i for i, x in enumerate(by_train)}
    for i, x in enumerate(by_val):
        x["rank_shift"] = i - train_rank[id(x)]   # >0: fell on validation vs train
    return by_val


def stability(results: dict[tuple, list], grid: dict) -> dict[tuple, dict]:
    """Stability over the (stops, trails, holds) exit grid — see stability_axes."""
    return stability_axes(results,
                          [sorted(grid["stops"]), sorted(grid["trails"]), sorted(grid["holds"])])


def stability_axes(results: dict[tuple, list], axes: list[list]) -> dict[tuple, dict]:
    """Full-set PnL of each combo vs its grid neighbors (±1 step on exactly one axis).
    A robust setting sits on a plateau: spike = pnl - neighbor_mean near 0 and
    neighbor_min healthy. A large positive spike is a lucky in-sample artifact."""
    totals = {c: _tot(p, list(range(len(p))))[0] for c, p in results.items()}

    def neighbors(combo: tuple) -> list[tuple]:
        out = []
        for ax, vals in enumerate(axes):
            j = vals.index(combo[ax])
            for k in (j - 1, j + 1):
                if 0 <= k < len(vals):
                    nb = list(combo)
                    nb[ax] = vals[k]
                    out.append(tuple(nb))
        return out

    out = {}
    for combo, total in totals.items():
        nb = [totals[n] for n in neighbors(combo) if n in totals]
        if not nb:
            out[combo] = {"neighbor_mean": total, "neighbor_min": total, "spike": 0.0}
            continue
        mean = sum(nb) / len(nb)
        out[combo] = {"neighbor_mean": mean, "neighbor_min": min(nb),
                      "spike": total - mean}
    return out
