"""Walk-forward grid-search of exit parameters (stop %, trailing %, days hold) over the
FIXED set of trades the live-faithful replay actually took — without spending tokens.

Entries come straight from the latest `backtest.py --source telegram` (live-replay)
HTML report, so they already reflect the progressive regime, catalyst memory, BTC
rule, listing/price guard and concurrency caps. We hold those entries/sides/sizes
constant and re-walk each one's candles under every (stop, trail, hold) combo —
including the engine's slippage, gap-fill and funding model, so the objective
matches what the backtest now reports.

The table is sorted by VALIDATION PnL (held-out later data), not in-sample PnL:
    python scripts/optimize.py                       # rolling-origin, 4 folds
    python scripts/optimize.py --mode holdout --train-frac 0.7
    python scripts/optimize.py --report data/backtest_report_sonnet_telegram.html

Read it with scripts/../src/hlbot/backtest/optimizer.py's caveats in mind: prefer a
combo that wins on validation AND sits on a stability plateau (low |spike|, healthy
neighbor-min). A combo that tops train but falls hard on validation (rank shift > 0)
did not generalize. Edit DEFAULT_GRID in hlbot/backtest/optimizer.py to change the grid.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.backtest.engine import fetch_candles, funding_cache  # noqa: E402
from hlbot.backtest.optimizer import (  # noqa: E402
    BREAKEVEN_GRID,
    DEFAULT_GRID,
    evaluate,
    max_horizon_s,
    split_holdout,
    split_rolling,
    stability,
    stability_axes,
    sweep,
    sweep_breakeven,
)
from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/backtest_report_sonnet_telegram.html",
                    help="live-replay report whose ACTUAL trades become the fixed entry set")
    ap.add_argument("--cache", default="data/bt_cache_replay_sonnet.json",
                    help="replay analysis cache (asset_class lookup for market resolution)")
    ap.add_argument("--mode", choices=["rolling", "holdout"], default="rolling",
                    help="walk-forward style: rolling-origin folds or one chronological holdout")
    ap.add_argument("--folds", type=int, default=4, help="rolling-origin fold count")
    ap.add_argument("--train-frac", type=float, default=0.7, help="holdout train fraction")
    ap.add_argument("--breakeven", action="store_true",
                    help="sweep the breakeven stop (arm x offset) ON TOP of the current "
                         "exit config instead of the (stop, trail, hold) grid; the arm=0 "
                         "row is the feature-off baseline")
    args = ap.parse_args()

    cfg = Config()
    r = cfg.app.risk
    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()

    # Fixed entry set = the trades the live-faithful replay actually took (progressive
    # regime + catalyst memory + BTC rule + listing/price guard + concurrency caps already
    # baked in). We hold entries/sides/sizes constant and sweep ONLY the exit params.
    html = pathlib.Path(args.report).read_text(encoding="utf-8")
    data, _ = json.JSONDecoder().raw_decode(
        html, html.index("const DATA = ") + len("const DATA = "))
    try:
        acache = json.loads(pathlib.Path(args.cache).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        acache = {}
    entries = []
    for row in data["rows"]:
        if row.get("status") != "traded":
            continue
        ac = (acache.get(row["id"]) or {}).get("asset_class")
        mkt = universe.resolve(row["ticker"], ac)
        if not mkt:
            continue
        entries.append({"market": mkt, "side": row["side"], "conf": row["confidence"],
                        "time_sensitivity": row["time_sensitivity"], "time_ms": row["time_ms"],
                        "notional": row.get("notional")})   # replayed size incl. exposure clamp
    win = data.get("window", [0, 0])
    fmt = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%m-%d")
    print(f"Fixed entry set: {len(entries)} replay trades · window {fmt(win[0])} → {fmt(win[1])}\n")

    grid = dict(DEFAULT_GRID)
    # Always score the CURRENT config combo, even if it's off-grid.
    cur = (r.stop_loss_by_sensitivity.get("days", r.stop_loss_pct),
           r.trail_pct_by_sensitivity.get("days", 0.0),
           int(r.exit_horizons.get("days", 0) / 3600))
    for axis, v in zip(("stops", "trails", "holds"), cur):
        if v not in grid[axis]:
            grid[axis] = sorted([*grid[axis], v])

    # Pre-fetch candles + funding rates once per entry (reused across all combos) —
    # the only API load; everything after is pure CPU.
    print("Fetching candles + funding per entry (once)…")
    for e in entries:
        e["_candles"], e["_frates"] = [], None
        mh = max_horizon_s(e["time_sensitivity"], grid["holds"], r)
        start, end = e["time_ms"] - 60_000, e["time_ms"] + mh * 1000 + 120_000
        for iv in (["1m", "5m", "15m"] if mh <= 6 * 3600 else ["5m", "15m", "1h"]):
            cs = await fetch_candles(hl, e["market"].name, start, end, iv)
            if cs:
                e["_candles"] = cs
                break
        if e["_candles"]:
            e["_frates"] = await funding_cache().rates(hl, e["market"].name,
                                                       e["time_ms"], end)
    n_excluded = sum(1 for e in entries if not e["_candles"])
    if n_excluded:
        print(f"  !! {n_excluded}/{len(entries)} entries excluded from ALL combos "
              f"(no candle data)")
    n_nofund = sum(1 for e in entries if e["_candles"] and e["_frates"] is None)
    if n_nofund:
        print(f"  !! funding history unavailable for {n_nofund} entries (cost 0 assumed)")

    splits = (split_rolling(entries, args.folds) if args.mode == "rolling"
              else split_holdout(entries, args.train_frac))
    label = (f"rolling-origin · {len(splits)} folds" if args.mode == "rolling"
             else f"holdout · {args.train_frac:.0%} train")

    if args.breakeven:
        results = sweep_breakeven(entries, BREAKEVEN_GRID, r, *cur)
        rows = evaluate(results, splits)
        stab = stability_axes(results, [sorted(set(BREAKEVEN_GRID["arms"])),
                                        sorted(set(BREAKEVEN_GRID["offsets"]))])
        base = next(x for x in rows if x["combo"] == (0.0, 0.0))   # feature off
        print(f"\n=== BREAKEVEN sweep · {len(entries)} fixed entries · {label} · "
              f"exits fixed at stop {cur[0]}, trail {cur[1]}, hold {cur[2]}h ===")
        print(f"  baseline (off): train ${base['train']:+.0f} · val ${base['val']:+.0f} · "
              f"total ${base['total']:+.0f}\n")
        print(f"  {'arm':>6} {'offset':>7}  {'train$':>8} {'val$':>8} {'worst$':>8} "
              f"{'folds+':>6} {'dVal$':>8} {'dTotal$':>8} {'spike$':>8}")
        print("  " + "-" * 76)
        for x in rows:
            arm, off = x["combo"]
            s = stab[x["combo"]]
            mark = "  <- off (baseline)" if (arm, off) == (0.0, 0.0) else ""
            print(f"  {arm*100:>5.2f}% {off*100:>6.2f}%  {x['train']:>+8.0f} {x['val']:>+8.0f} "
                  f"{x['val_worst']:>+8.0f} {x['val_pos_frac']*100:>5.0f}% "
                  f"{x['val'] - base['val']:>+8.0f} {x['total'] - base['total']:>+8.0f} "
                  f"{s['spike']:>+8.0f}{mark}")
        print("\n  Ship only if a CONTIGUOUS arm range beats the off-baseline on dVal$ —")
        print("  a single good arm value between losers is sampling noise, not a mechanism.")
        return

    results = sweep(entries, grid, r)
    rows = evaluate(results, splits)
    stab = stability(results, grid)

    print(f"\n=== {len(rows)} combos · {len(entries)} fixed entries · {label} · "
          f"sorted by VALIDATION PnL ===")
    print(f"  (current config days-tier: stop {cur[0]}, trail {cur[1]}, hold {cur[2]}h)\n")
    print(f"  {'stop':>5} {'trail':>6} {'hold':>5}  {'train$':>8} {'val$':>8} {'worst$':>8} "
          f"{'folds+':>6} {'shift':>5} {'spike$':>8} {'nbMin$':>8}")
    print("  " + "-" * 78)
    for x in rows:
        stop, trail, hold = x["combo"]
        s = stab[x["combo"]]
        flags = []
        if x["val"] <= 0:
            flags.append("val<=0")
        if x["rank_shift"] > max(2, len(rows) // 4):
            flags.append("does not generalize")
        if s["spike"] > 0 and s["spike"] > 0.5 * max(1.0, abs(s["neighbor_mean"])):
            flags.append("spike")
        cur_mark = "  <- current config" if (stop, trail, hold) == cur else ""
        print(f"  {stop*100:>4.0f}% {trail*100:>5.1f}% {hold:>4}h  {x['train']:>+8.0f} "
              f"{x['val']:>+8.0f} {x['val_worst']:>+8.0f} {x['val_pos_frac']*100:>5.0f}% "
              f"{x['rank_shift']:>+5} {s['spike']:>+8.0f} {s['neighbor_min']:>+8.0f}"
              f"{cur_mark}{('  [' + ', '.join(flags) + ']') if flags else ''}")
    print("\n  Pick a combo that wins on val$, holds up in worst$, and has |spike$| small")
    print("  (a plateau). Train-only winners with a positive rank shift are overfit.")


if __name__ == "__main__":
    asyncio.run(main())
