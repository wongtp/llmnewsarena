"""List and compare archived backtest runs (snapshotted into data/bt_history/ by
backtest.py). Free — just reads the saved reports.

    python scripts/compare_runs.py                  # list recent runs (summary table)
    python scripts/compare_runs.py latest prev      # diff the two most recent runs
    python scripts/compare_runs.py 20260607 20260605  # diff by timestamp prefix
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HIST = pathlib.Path("data/bt_history")


def _runs() -> list[pathlib.Path]:
    return sorted(d for d in HIST.iterdir() if d.is_dir()) if HIST.exists() else []


def _meta(d: pathlib.Path) -> dict:
    p = d / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"stamp": d.name}


def _traded(d: pathlib.Path) -> dict:
    html = next(iter(d.glob("*.html")), None)
    if not html:
        return {}
    t = html.read_text(encoding="utf-8")
    data, _ = json.JSONDecoder().raw_decode(t, t.index("const DATA = ") + len("const DATA = "))
    return {(r["ticker"], r["time_ms"]): r for r in data.get("rows", []) if r.get("status") == "traded"}


def _fmt(ms) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%m-%d %H:%M") if ms else "-"


def list_runs() -> None:
    rs = _runs()
    if not rs:
        print("No archived runs yet — run scripts/backtest.py (archiving is on by default).")
        return
    print(f"{'stamp':16} {'window':14} {'trades':>6} {'win%':>5} {'PnL':>9} {'pf':>5} "
          f"{'maxDD':>7} {'cost':>7}  days-exit")
    print("-" * 92)
    for d in rs:
        m = _meta(d)
        w = m.get("window") or [0, 0]
        ec = (m.get("exit_config") or {}).get("days", {})
        exit_s = f"{ec.get('stop_pct','?')}/{ec.get('trail_pct','?')}/{ec.get('hold_h','?')}h"
        pf = m.get("profit_factor")
        pf_s = "    -" if pf is None and "profit_factor" not in m else (
            "  inf" if pf is None else f"{pf:5.2f}")
        dd = m.get("max_dd_usd")
        dd_s = "      -" if dd is None else f"${dd:>6.0f}"
        print(f"{m.get('stamp',''):16} {_fmt(w[0])[:5]}->{_fmt(w[1])[:5]} {m.get('trades',0):>6} "
              f"{m.get('win_rate',0)*100:>4.0f}% ${m.get('total_pnl',0):>8.0f} {pf_s} "
              f"{dd_s} ${m.get('cost_usd',0):>6.2f}  {exit_s}")


def diff(d1: pathlib.Path, d2: pathlib.Path) -> None:
    older, newer = sorted([d1, d2])           # A = older, B = newer
    ta, tb = _traded(older), _traded(newer)
    ma, mb = _meta(older), _meta(newer)
    added = [tb[k] for k in tb if k not in ta]      # trades only in the newer run
    dropped = [ta[k] for k in ta if k not in tb]    # trades only in the older run
    print(f"A (older) {ma.get('stamp')}: {ma.get('trades')} trades, ${ma.get('total_pnl')}")
    print(f"B (newer) {mb.get('stamp')}: {mb.get('trades')} trades, ${mb.get('total_pnl')}")
    print(f"PnL delta (B - A): ${(mb.get('total_pnl',0) - ma.get('total_pnl',0)):+.0f}\n")

    def show(label, rows):
        print(f"{label} ({len(rows)}, sum ${sum((r.get('pnl') or 0) for r in rows):+.0f}):")
        for r in sorted(rows, key=lambda x: x["time_ms"]):
            print(f"  {_fmt(r['time_ms'])} {r['ticker']:6} {r['side']:5} "
                  f"conf={r.get('confidence',0):.0%} ${ (r.get('pnl') or 0):+8.1f} ({r.get('exit_reason')})")

    show("ADDED in B", added)
    print()
    show("DROPPED from A", dropped)


def main() -> None:
    args = sys.argv[1:]
    rs = _runs()
    if len(args) < 2:
        list_runs()
        return
    if len(rs) < 2:
        print("Need at least 2 archived runs to diff.")
        return

    def resolve(tok: str) -> pathlib.Path:
        if tok == "latest":
            return rs[-1]
        if tok == "prev":
            return rs[-2]
        for d in rs:
            if d.name.startswith(tok):
                return d
        raise SystemExit(f"run not found: {tok}")

    diff(resolve(args[0]), resolve(args[1]))


if __name__ == "__main__":
    main()
