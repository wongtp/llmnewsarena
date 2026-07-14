"""Edge report: WHERE the entry edge actually is — slices a saved backtest report by source,
confidence (calibration), market type, and ticker. The HTML report shows every trade; this
answers "which sources make money?" and "is the confidence gate in the right place?".

Reads only the saved report (uses the realized PnL of traded rows + the precomputed would-be
PnL of skipped reads), so it's FREE and offline — but run a backtest WITHOUT --no-would-pnl
first so skipped reads carry would_pnl (that's the default).

    python scripts/edge_report.py
    python scripts/edge_report.py --report data/backtest_report_sonnet_telegram.html
    python scripts/edge_report.py --min-conf 0.6 --bucket 0.05
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.backtest.edge import (  # noqa: E402
    calibration,
    excursion_stats,
    gate_sweep_oos,
    group_by,
    kelly_table,
    load_report,
    outcomes,
    suggest_gate,
    suggest_gate_oos,
    summarize,
)


def _ts(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%m-%d %H:%M")


def _pf(s: dict) -> str:
    pf = s.get("pf", 0.0)
    return " inf" if pf == float("inf") else f"{pf:4.2f}"


def _print_group(title: str, rows, label_w: int = 28) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'':{label_w}} {'n':>4} {'trd':>4} {'win%':>5} {'pf':>4} {'avg$':>9} {'total$':>11}")
    print("  " + "-" * (label_w + 41))
    for label, s in rows:
        print(f"  {str(label)[:label_w]:{label_w}} {s['n']:>4} {s['traded']:>4} "
              f"{s['win_rate']*100:>4.0f}% {_pf(s)} {s['avg']:>+9.1f} {s['total']:>+11.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="data/backtest_report_sonnet_telegram.html")
    ap.add_argument("--min-conf", type=float, default=0.50,
                    help="floor for inclusion (would-be PnL is only computed >=0.50)")
    ap.add_argument("--bucket", type=float, default=0.05, help="calibration bucket width")
    ap.add_argument("--top", type=int, default=12, help="rows to show in by-ticker tables")
    args = ap.parse_args()

    try:
        data = load_report(args.report)
    except FileNotFoundError:
        print(f"Report not found: {args.report}\nRun scripts/backtest.py first.")
        return
    rows = data.get("rows", [])
    gate = data.get("threshold", 0.0) or 0.0
    items = outcomes(rows, min_conf=args.min_conf)

    w = data.get("window", [0, 0])
    print("=" * 72)
    print(f" EDGE REPORT  ·  {args.report}")
    print(f" {_ts(w[0])} -> {_ts(w[1])} UTC  ·  model {data.get('model','?')}  ·  gate {gate:.0%}")
    print("=" * 72)
    if not items:
        print("\n No scored outcomes. Re-run scripts/backtest.py WITHOUT --no-would-pnl so "
              "skipped reads carry would-be PnL (it's on by default).")
        return
    n_real = sum(1 for o in items if o["traded"])
    print(f" {len(items)} directional reads scored (conf>={args.min_conf:.0%}): "
          f"{n_real} actually traded (real PnL) + {len(items)-n_real} skipped (would-be PnL).")
    print(" 'win%' = share with positive PnL; 'total$' sums real+would-be PnL. Would-be PnL is a")
    print(" standalone hypothetical (ignores caps/cooldown/exposure), so read it as signal quality.")

    # ---- Confidence calibration ------------------------------------------
    print(f"\n=== CONFIDENCE CALIBRATION (bucket {args.bucket:.2f}) — is conf monotonic with edge? ===")
    print(f"  {'bucket':>11} {'n':>4} {'trd':>4} {'win%':>5} {'avg$':>9} {'total$':>11}")
    print("  " + "-" * 49)
    for (a, b), s in calibration(items, width=args.bucket, lo=0.5):
        mark = "  <- gate" if a <= gate < b else ""
        print(f"  {a:.0%}-{b:.0%}".rjust(13) + f" {s['n']:>4} {s['traded']:>4} "
              f"{s['win_rate']*100:>4.0f}% {s['avg']:>+9.1f} {s['total']:>+11.1f}{mark}")

    above = summarize([o for o in items if o["conf"] >= gate])
    below = summarize([o for o in items if o["conf"] < gate])
    print(f"\n  at/above gate ({gate:.0%}): {above['n']:>3} reads · "
          f"win {above['win_rate']*100:.0f}% · total ${above['total']:+.0f}")
    print(f"  below gate      : {below['n']:>3} reads · "
          f"win {below['win_rate']*100:.0f}% · total ${below['total']:+.0f}")
    best = suggest_gate(items)
    if best:
        print(f"  [in-sample] PnL-max gate on this whole window: {best['gate']:.0%} "
              f"({best['n']} reads, total ${best['total']:+.0f}) — overfits; see OOS below.")
    oos = suggest_gate_oos(items)
    if oos:
        tr, va = oos["train"], oos["val"]
        print(f"\n=== OUT-OF-SAMPLE GATE (fit on first {oos['n_train']} reads, "
              f"validated on the last {oos['n_val']}) ===")
        print(f"  gate fit on train: {oos['gate']:.0%}  ->  train ${tr['total']:+.0f} "
              f"({tr['n']} reads, win {tr['win_rate']*100:.0f}%)  |  "
              f"validation ${va['total']:+.0f} ({va['n']} reads, win {va['win_rate']*100:.0f}%)")
        gates = [round(0.78 + 0.02 * i, 2) for i in range(8)]   # 0.78 .. 0.92
        print(f"  {'gate':>6} {'trainN':>7} {'train$':>9}   {'valN':>5} {'val$':>9}")
        for g, ts_, vs in gate_sweep_oos(items, gates):
            print(f"  {g:>5.0%} {ts_['n']:>7} {ts_['total']:>+9.0f}   "
                  f"{vs['n']:>5} {vs['total']:>+9.0f}")
        print("  A gate whose val$ tracks its train$ generalizes; one that flips sign doesn't.")

    exc = excursion_stats(items)
    if exc.get("n_scored"):
        def fr(d: dict) -> str:
            return (f"p50 {d['p50']*100:.1f}% / p75 {d['p75']*100:.1f}% / "
                    f"p90 {d['p90']*100:.1f}% (n={d['n']})") if d else "(none)"
        print(f"\n=== STOP/TRAIL PLACEMENT (MAE/MFE of {exc['n_scored']} traded rows) ===")
        print(f"  winners' drawdown before winning (MAE): {fr(exc['winners_mae'])}")
        print(f"     -> a stop tighter than the p75/p90 MAE clips winners before they run")
        print(f"  winners' best excursion (MFE):          {fr(exc['winners_mfe'])}")
        print(f"  losers' best excursion before dying:    {fr(exc['losers_mfe'])}")
        print(f"     -> a breakeven move armed below the losers' p50 MFE rescues real losses")
        print(f"  losers' final drawdown (MAE):           {fr(exc['losers_mae'])}")

    # ---- Quarter-Kelly tier check (report-only; never auto-applied) ------
    from hlbot.config import Config
    r_cfg = Config().app.risk
    kt = kelly_table(data["rows"], r_cfg.size_tiers, r_cfg.account_size_usd,
                     r_cfg.max_notional_usd)
    print("\n=== QUARTER-KELLY TIER CHECK (traded rows only; report-only — edit size_tiers "
          "by hand, n>=20 to act) ===")
    print(f"  {'tier':>12} {'n':>4} {'win%':>5} {'kelly f*':>9} {'q-kelly $':>10} {'current $':>10}")
    print("  " + "-" * 56)
    for k in kt:
        win = f"{k['win']*100:>4.0f}%" if k["win"] is not None else "    -"
        kf = f"{k['kelly']:>+9.2f}" if k["kelly"] is not None else "        -"
        sg = f"{k['suggest']:>10.0f}" if k["suggest"] is not None else f"{'n<20':>10}"
        print(f"  {k['lo']:.2f}-{min(k['hi'], 1.0):.2f}  {k['n']:>4} {win} {kf} {sg} "
              f"{k['cur_notional']:>10.0f}")
    print("  A tier whose q-kelly$ sits far below current$ at n>=20 is oversized for its "
          "realized edge.")

    # ---- Slices ----------------------------------------------------------
    measured = [o for o in items if o.get("pre_move") is not None]
    if measured:
        _print_group("BY PRE-MOVE AT ENTRY (did chasing a completed repricing lose?)",
                     group_by(measured, "pre_move_bucket"), label_w=22)
        print("  If the '+1.25%'/'>= +3%' buckets are net losers, enable the move guard "
              "(risk.pre_move_*).")
    _print_group("BY SOURCE (whitelist candidates — winners up top)", group_by(items, "source"))
    _print_group("BY TIME SENSITIVITY", group_by(items, "time_sensitivity"), label_w=14)
    _print_group("BY MARKET TYPE", group_by(items, "market_type"), label_w=24)

    by_tkr = group_by(items, "ticker")
    _print_group(f"BY TICKER (top {args.top})", by_tkr[:args.top], label_w=10)
    losers = [r for r in by_tkr if r[1]["total"] < 0][-args.top:]
    if losers:
        _print_group(f"BY TICKER (worst {len(losers)})", list(reversed(losers)), label_w=10)


if __name__ == "__main__":
    main()
