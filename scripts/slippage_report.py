"""Live fill-quality + latency attribution from the bot's own SQLite audit log.

Answers "is the backtest's fill model honest?" with real fills:
  - ENTRY slippage: decision mid (decisions.entry_px) vs actual fill (positions.entry_px),
    in signed bps where positive = adverse. Includes half-spread by construction — the
    honest comparison against a backtest that enters at a printed candle open.
  - EXIT slippage: trigger mid (Position.exit_decision_px, recorded since this script
    shipped) vs the close fill. Exchange-stop reconciled closes are EXCLUDED: their
    exit_px is an estimate recorded at the stop price (position_manager), not a fill.
  - LATENCY: news -> decision (includes the Claude call) and news -> order, per trade.

Compare the printed percentiles against risk.dry_run_slippage_pct (5 bps default) and
risk.backtest_slippage_pct, then calibrate those knobs. Slices with n < 10 are noise.

    python scripts/slippage_report.py                    # live fills only
    python scripts/slippage_report.py --dry-run          # paper fills (sanity: ~= configured slip)
    python scripts/slippage_report.py --vs-backtest      # also compare vs next-1m-bar-open entry

--vs-backtest needs network + .env (fetches the 1m candle at each news time) and prints
the premium/discount of real fills vs the backtest's pick_entry assumption — the number
that says whether backtest PnL is trustworthy at all.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sqlite3
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

DB = "data/hlbot.sqlite"
EXCHANGE_CLOSE_REASONS = ("stop loss (exchange)",)


def fetch_trades(db_path: str, dry_run: bool) -> list[dict]:
    """positions ⨝ decisions(action=enter) ⨝ news, closed or open, oldest first."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT p.id, p.symbol, p.dex, p.side, p.entry_px AS fill_px, p.exit_px,
               p.exit_reason, p.opened_ms, p.closed_ms, p.notional, p.status, p.json AS pjson,
               d.entry_px AS decision_px, d.ts AS decision_ms,
               n.ts AS news_ms
        FROM positions p
        JOIN decisions d ON d.news_id = p.news_id AND d.action = 'enter'
        JOIN news n ON n.id = p.news_id
        WHERE p.dry_run = ?
        ORDER BY p.opened_ms
        """, (int(dry_run),)).fetchall()
    con.close()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["exit_decision_px"] = float(json.loads(d.pop("pjson") or "{}")
                                          .get("exit_decision_px") or 0.0)
        except Exception:  # noqa: BLE001
            d["exit_decision_px"] = 0.0
        out.append(d)
    return out


def signed_bps(side: str, ref: float, actual: float, *, is_exit: bool) -> float:
    """Adverse-positive slippage in bps. Entry: paid more (long) / received less (short).
    Exit: received less (long) / paid more (short)."""
    raw = (actual - ref) / ref * 10_000
    adverse_sign = (1 if side == "long" else -1) * (-1 if is_exit else 1)
    return raw * adverse_sign


def pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def show_slice(title: str, groups: dict[str, list[float]]) -> None:
    print(f"\n=== {title} (signed bps, + = adverse) ===")
    print(f"  {'':18} {'n':>4} {'median':>8} {'p90':>8} {'worst':>8}")
    print("  " + "-" * 50)
    for label, vals in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if not vals:
            continue
        print(f"  {label[:18]:18} {len(vals):>4} {pct(vals, 0.5):>8.1f} {pct(vals, 0.9):>8.1f} "
              f"{max(vals):>8.1f}{'   (n<10: noise)' if len(vals) < 10 else ''}")


def bucket(groups: dict, key: str, val: float) -> None:
    groups.setdefault(key, []).append(val)


async def vs_backtest(trades: list[dict]) -> None:
    """Real entry fill vs the backtest's assumption (open of first 1m candle at/after the
    news). Positive bps = live paid worse than the backtest assumes."""
    from hlbot.backtest.engine import BACKTEST_MAX_ENTRY_DELAY_S, fetch_candles, pick_entry
    from hlbot.config import Config
    from hlbot.trading.hl_client import HLClient

    cfg = Config()
    hl = HLClient(cfg)
    await hl.connect()
    diffs: dict[str, list[float]] = {}
    misses = 0
    for t in trades:
        name = f"{t['dex']}:{t['symbol']}" if t["dex"] else t["symbol"]
        candles = await fetch_candles(hl, name, t["news_ms"] - 60_000,
                                      t["news_ms"] + 7_200_000, "1m")
        idx, bt_px = (None, None)
        if candles:
            idx, bt_px = pick_entry(candles, t["news_ms"],
                                    max_delay_ms=BACKTEST_MAX_ENTRY_DELAY_S * 1000)
        if not bt_px:
            misses += 1
            continue
        bps = signed_bps(t["side"], bt_px, t["fill_px"], is_exit=False)
        bucket(diffs, "ALL", bps)
        bucket(diffs, f"dex:{t['dex'] or 'crypto'}", bps)
    show_slice("LIVE FILL vs BACKTEST next-1m-open ASSUMPTION", diffs)
    if misses:
        print(f"  ({misses} trades skipped: no 1m candle at news time)")
    print("  -> if ALL.median is materially positive, raise risk.backtest_slippage_pct to match.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--dry-run", action="store_true",
                    help="analyze PAPER fills instead of live (sanity-check the slip model)")
    ap.add_argument("--vs-backtest", action="store_true",
                    help="also fetch 1m candles and compare fills vs the backtest entry model")
    args = ap.parse_args()

    try:
        trades = fetch_trades(args.db, args.dry_run)
    except sqlite3.OperationalError as exc:
        print(f"Cannot read {args.db}: {exc}")
        return
    mode = "PAPER" if args.dry_run else "LIVE"
    if not trades:
        print(f"No {mode} positions with a matching enter-decision in {args.db}.")
        return

    fmt = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%m-%d")
    print("=" * 72)
    print(f" SLIPPAGE + LATENCY REPORT · {mode} · {len(trades)} fills · "
          f"{fmt(trades[0]['opened_ms'])} -> {fmt(trades[-1]['opened_ms'])}")
    print("=" * 72)

    # ---- entry slippage --------------------------------------------------
    ent: dict[str, list[float]] = {}
    for t in trades:
        if not (t["decision_px"] and t["fill_px"]):
            continue
        bps = signed_bps(t["side"], t["decision_px"], t["fill_px"], is_exit=False)
        bucket(ent, "ALL", bps)
        bucket(ent, f"dex:{t['dex'] or 'crypto'}", bps)
        bucket(ent, t["symbol"], bps)
        bucket(ent, f"h{dt.datetime.fromtimestamp(t['opened_ms']/1000, dt.timezone.utc).hour:02d} UTC",
               bps)
        nb = ("<$3k" if t["notional"] < 3000 else "$3-7k" if t["notional"] < 7000 else ">$7k")
        bucket(ent, f"size {nb}", bps)
    show_slice("ENTRY: decision mid -> fill", ent)
    print("  -> compare median vs dry_run_slippage_pct "
          "(5 bps default) and backtest_slippage_pct.")

    # ---- exit slippage ---------------------------------------------------
    ext: dict[str, list[float]] = {}
    n_est = n_missing = 0
    for t in trades:
        if t["status"] != "closed" or not t["exit_px"]:
            continue
        if t["exit_reason"] in EXCHANGE_CLOSE_REASONS:
            n_est += 1   # exit_px is the estimated stop price, not a fill — excluded
            continue
        if not t["exit_decision_px"]:
            n_missing += 1   # closed before exit_decision_px existed
            continue
        bps = signed_bps(t["side"], t["exit_decision_px"], t["exit_px"], is_exit=True)
        bucket(ext, "ALL", bps)
        bucket(ext, f"dex:{t['dex'] or 'crypto'}", bps)
        bucket(ext, (t["exit_reason"] or "?")[:18], bps)
    if ext:
        show_slice("EXIT: trigger mid -> fill", ext)
    if n_est:
        print(f"\n  {n_est} exchange-stop close(s) excluded (recorded at the estimated stop "
              f"price, not a real fill) — calibrate backtest_stop_slippage_pct only from "
              f"bot-side stop closes above.")
    if n_missing:
        print(f"  {n_missing} close(s) predate exit_decision_px capture (no exit slippage data).")

    # ---- latency ---------------------------------------------------------
    lat_dec = [(t["decision_ms"] - t["news_ms"]) / 1000 for t in trades
               if t["decision_ms"] and t["news_ms"]]
    lat_ord = [(t["opened_ms"] - t["news_ms"]) / 1000 for t in trades
               if t["opened_ms"] and t["news_ms"]]
    print("\n=== LATENCY (seconds) ===")
    if lat_dec:
        print(f"  news -> decision (incl. Claude): median {pct(lat_dec, 0.5):.1f}s · "
              f"p90 {pct(lat_dec, 0.9):.1f}s · worst {max(lat_dec):.1f}s")
    if lat_ord:
        print(f"  news -> order:                   median {pct(lat_ord, 0.5):.1f}s · "
              f"p90 {pct(lat_ord, 0.9):.1f}s · worst {max(lat_ord):.1f}s")
        print("  (opened_ms stamps just before the order RTT — true fill is slightly later)")

    if args.vs_backtest:
        asyncio.run(vs_backtest(trades))


if __name__ == "__main__":
    main()
