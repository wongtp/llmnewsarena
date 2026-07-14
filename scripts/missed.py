"""'Would-be PnL' on news we SKIPPED with confidence >= a threshold — i.e. what we'd
have made (or lost) had we traded them. Excludes blacklisted symbols and assets with no
price at news time (not listed yet / data gap). Reads a saved report; free (candles only).

    python scripts/missed.py                       # latest report, conf >= 0.50
    python scripts/missed.py --min-conf 0.6
    python scripts/missed.py --report data/bt_history/<stamp>/backtest_report_sonnet_telegram.html
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import json
import sys

sys.path.insert(0, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.backtest.engine import simulate  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

fmt = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%m-%d %H:%M")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="data/backtest_report_sonnet_telegram.html")
    ap.add_argument("--cache", default="data/bt_cache_replay_sonnet.json")
    ap.add_argument("--min-conf", type=float, default=0.50)
    args = ap.parse_args()

    cfg = Config()
    r = cfg.app.risk
    blacklist = {b.upper() for b in cfg.app.filters.market_blacklist}
    hl = HLClient(cfg); await hl.connect()
    u = Universe(hl, cfg.app.filters.allowed_dexes); await u.refresh()

    html = open(args.report, encoding="utf-8").read()
    data, _ = json.JSONDecoder().raw_decode(html, html.index("const DATA = ") + len("const DATA = "))
    try:
        acache = json.load(open(args.cache, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        acache = {}

    skipped = [row for row in data["rows"]
               if row.get("status") == "rejected" and row.get("direction") in ("long", "short")
               and (row.get("confidence") or 0) >= args.min_conf
               and (row.get("ticker") or "").upper() not in blacklist]

    results = []
    for row in skipped:
        ac = (acache.get(row["id"]) or {}).get("asset_class")
        mkt = u.resolve(row["ticker"], ac)
        if not mkt or mkt.symbol.upper() in blacklist:
            continue
        tr = await simulate(hl, mkt, row["direction"], row["time_ms"], row["confidence"],
                            row["time_sensitivity"], r, news_id=row["id"])
        if not tr:               # no price at news time (not listed yet / data gap) -> excluded
            continue
        results.append((row, tr))

    results.sort(key=lambda x: x[1].pnl)
    total = sum(t.pnl for _, t in results)
    wins = sum(1 for _, t in results if t.pnl > 0)
    print(f"\nSkipped, conf>={args.min_conf:.0%}, tradable (had a price): {len(results)} "
          f"of {len(skipped)} candidates\n")
    print(f"  {'time':12} {'tkr':6} {'side':5} {'cf':>3} {'would-PnL':>10}  skip-reason -> exit")
    print("  " + "-" * 78)
    for row, t in results:
        print(f"  {fmt(row['time_ms']):12} {row['ticker']:6} {row['direction']:5} "
              f"{row['confidence']*100:>3.0f} ${t.pnl:>+8.1f}  [{row['reason'][:30]}] {t.reason}")
    print("  " + "-" * 78)
    print(f"  TOTAL would-be PnL left on the table: ${total:+.0f}   ({wins}/{len(results)} winners)\n")

    byr = collections.defaultdict(lambda: [0, 0.0])
    for row, t in results:
        key = row["reason"].split("(")[0].strip()[:26]
        byr[key][0] += 1
        byr[key][1] += t.pnl
    print("  by skip reason:")
    for k, (n, p) in sorted(byr.items(), key=lambda x: x[1][1]):
        print(f"    {k:28} {n:>3}x  ${p:>+8.0f}")


if __name__ == "__main__":
    asyncio.run(main())
