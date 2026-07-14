"""Generalized exit walker for the exit-strategy lab (scripts/exit_lab.py and
scripts/earnings_bench.py --variant).

`walk_variant` mirrors `engine.walk_candles` semantics EXACTLY (stop checked before TP
within a bar, peak updated after the checks, gap-opens fill at the open, MAE/MFE from
per-bar extremes independent of the trailing peak) and generalizes the exit shape:

- trail_tiers: ((arm_mfe_pct, trail_pct), ...) — each tier arms once the peak has moved
  arm_mfe_pct in our favor (strictly beyond, matching production's `peak > entry`); the
  binding stop is the TIGHTEST armed level. Production trail == ((0.0, trail_pct),).
- tp_px: fixed take-profit that may coexist with a trail (production forces tp=0 when
  trailing; a "far TP" variant monetizes blowoffs instead of giving back trail_pct).
- extension: at the time-exit boundary, if the position's close-price PnL is at least
  extend_min_pnl_pct, the horizon extends once to extend_end_ms ("let winners run");
  losers still exit on time. Requires candles fetched out to extend_end_ms.

Scale-out variants are composed by the callers as two independent walks (a TP leg and a
production leg) — position legs don't interact, so the decomposition is exact.

tests/test_exit_variants.py proves walk_variant reproduces walk_candles bar-for-bar under
production parameters; only that equivalence makes variant deltas trustworthy.
"""
from __future__ import annotations

from .engine import WalkResult

# (name, params) grids for the lab. Tier/TP percentages are fractions of entry.
SCALE_OUT_GRID = [(0.5, 0.04), (0.5, 0.06), (0.5, 0.08)]          # (closed_frac, tp1_pct)
RATCHET_GRID = [
    ((0.0, 0.08), (0.06, 0.05), (0.12, 0.03)),
    ((0.0, 0.08), (0.05, 0.05), (0.10, 0.03)),
    ((0.0, 0.08), (0.08, 0.04)),
]
ARMED_TRAIL_GRID = [((0.03, 0.05),), ((0.04, 0.04),), ((0.04, 0.05),), ((0.06, 0.04),)]
FAR_TP_GRID = [0.10, 0.12, 0.14, 0.15, 0.20]                       # tp_pct with 8% trail
EXTEND_GRID = [(0.00, 120), (0.00, 168), (0.02, 120), (0.02, 168)]  # (min_pnl, total_hours)


def walk_variant(side: str, entry_px: float, stop_px: float, candles: list[dict],
                 start_index: int, horizon_end_ms: int, *,
                 trail_tiers: tuple[tuple[float, float], ...] = (),
                 tp_px: float = 0.0,
                 extend_min_pnl_pct: float | None = None,
                 extend_end_ms: int = 0) -> WalkResult:
    peak = entry_px
    adverse = favor = entry_px
    entry_ms = int(candles[start_index]["t"])
    favor_ms = entry_ms
    last = candles[-1]
    extended = False

    def res(px: float, reason: str, ct: int) -> WalkResult:
        if side == "long":
            mae = max(0.0, (entry_px - adverse) / entry_px)
            mfe = max(0.0, (favor - entry_px) / entry_px)
        else:
            mae = max(0.0, (adverse - entry_px) / entry_px)
            mfe = max(0.0, (entry_px - favor) / entry_px)
        return WalkResult(px, reason, ct, mae, mfe, max(0, favor_ms - entry_ms))

    def eff_stop() -> tuple[float, str]:
        eff, label = stop_px, "stop loss"
        for arm, tr in trail_tiers:
            if side == "long":
                if peak > entry_px * (1 + arm):
                    t = peak * (1 - tr)
                    if t > eff:
                        eff, label = t, "trailing stop"
            else:
                if peak < entry_px * (1 - arm):
                    t = peak * (1 + tr)
                    if t < eff:
                        eff, label = t, "trailing stop"
        return eff, label

    for c in candles[start_index:]:
        o, hi, lo = float(c["o"]), float(c["h"]), float(c["l"])
        close, ct = float(c["c"]), int(c["T"])
        eff, label = eff_stop()
        if side == "long":
            if o <= eff:
                adverse = min(adverse, o)
                return res(o, label + " (gap)", ct)
            adverse = min(adverse, lo)
            if hi > favor:
                favor, favor_ms = hi, ct
            if lo <= eff:
                return res(eff, label, ct)
            if tp_px > 0 and hi >= tp_px:
                return res(tp_px, "take profit", ct)
            peak = max(peak, hi)
        else:
            if o >= eff:
                adverse = max(adverse, o)
                return res(o, label + " (gap)", ct)
            adverse = max(adverse, hi)
            if lo < favor:
                favor, favor_ms = lo, ct
            if hi >= eff:
                return res(eff, label, ct)
            if tp_px > 0 and lo <= tp_px:
                return res(tp_px, "take profit", ct)
            peak = min(peak, lo)
        if ct >= horizon_end_ms:
            pnl = (close - entry_px) / entry_px if side == "long" \
                else (entry_px - close) / entry_px
            if (extend_min_pnl_pct is not None and not extended
                    and extend_end_ms > horizon_end_ms and pnl >= extend_min_pnl_pct):
                extended = True
                horizon_end_ms = extend_end_ms
                continue
            return res(close, "time exit (extended)" if extended else "time exit", ct)
    return res(float(last["c"]), "time exit (data end)", int(last["T"]))
