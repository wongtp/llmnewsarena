"""Per-signal confidence-gate sweep from a replay analysis cache.

Simulates EVERY directional analysis at/above the lowest gate through the production
exit engine — no portfolio caps or cooldown sequencing (a per-symbol duplicate window
approximates dedup), so two models' caches can be compared at *their own best gate*
without portfolio-interaction confounds. Candle fetches hit the bt_candles disk cache;
new sub-gate signals fetch paced.

Also prints each cache's analyzer latency/cost stats (cached entries carry the real
per-call latency_ms / cost_usd from when they were produced).

Usage:
  .venv/bin/python scripts/gate_sweep.py --cache data/bt_cache_replay_sonnet.json
  .venv/bin/python scripts/gate_sweep.py --cache data/bt_cache_replay_haiku.json
"""
import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, "src")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.backtest.engine import simulate  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

GATES = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--history-file", default="data/bt_tg_history_270d.json")
    ap.add_argument("--days", type=float, default=250.0)
    ap.add_argument("--dup-window", type=float, default=7200.0,
                    help="per-symbol duplicate window seconds (approximates dedup/cooldown)")
    args = ap.parse_args()

    cfg = Config()
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing()))
        return
    r = cfg.app.risk

    cache = json.loads(pathlib.Path(args.cache).read_text(encoding="utf-8"))
    time_by_id = {d["id"]: d["time_ms"]
                  for d in json.loads(pathlib.Path(args.history_file).read_text(
                      encoding="utf-8"))}
    cutoff = (time.time() - args.days * 86400) * 1000

    lat = [a["latency_ms"] for a in cache.values() if a.get("latency_ms")]
    cost = [a["cost_usd"] for a in cache.values() if a.get("cost_usd")]
    print(f"cache: {args.cache}  ({len(cache)} analyses)")
    if len(lat) >= 2:   # quantiles() raises on a single sample
        print(f"  analyzer latency ms  p50 {statistics.median(lat):,.0f}  "
              f"p90 {statistics.quantiles(lat, n=10)[8]:,.0f}")
    if cost:
        print(f"  cost/analysis  mean ${statistics.mean(cost):.4f}  "
              f"total ${sum(cost):.2f}")

    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()

    sigs = []
    for nid, a in cache.items():
        t_ms = time_by_id.get(nid)
        if t_ms is None or t_ms < cutoff:
            continue
        if (a.get("direction") or "none") not in ("long", "short"):
            continue
        conf = a.get("confidence") or 0.0
        if conf < min(GATES):
            continue
        if a.get("is_stale"):
            continue
        market = universe.resolve(a.get("ticker"), a.get("asset_class"))
        if market is None:
            continue
        sigs.append((t_ms, nid, a, market, conf))
    sigs.sort()
    print(f"  directional signals >= {min(GATES):.2f} in window, resolvable: {len(sigs)}")

    sims: dict[str, object] = {}
    n_no_data = 0
    for t_ms, nid, a, market, conf in sigs:
        tr = await simulate(hl, market, a["direction"], t_ms, conf,
                            a.get("time_sensitivity") or "none", r, news_id=nid)
        if tr is None:
            n_no_data += 1
        sims[nid] = tr
    print(f"  simulated {len(sims) - n_no_data}, no-candle-data {n_no_data}")

    print(f"\n{'gate':>5} {'n':>4} {'win%':>5} {'total PnL':>10} {'PF':>6} "
          f"{'avg/trade':>9}  per-sensitivity n")
    for g in GATES:
        last_by_sym: dict[str, float] = {}
        picked = []
        for t_ms, nid, a, market, conf in sigs:
            if conf < g:
                continue
            prev = last_by_sym.get(market.symbol)
            if prev is not None and (t_ms - prev) < args.dup_window * 1000:
                continue
            last_by_sym[market.symbol] = t_ms
            tr = sims.get(nid)
            if tr is not None:
                picked.append((tr, a.get("time_sensitivity") or "none"))
        if not picked:
            print(f"{g:>5.2f} {0:>4}")
            continue
        pnls = [t.pnl for t, _ in picked]
        gw = sum(p for p in pnls if p > 0)
        gl = -sum(p for p in pnls if p < 0)
        pf = gw / gl if gl > 0 else float("inf")
        sens_n: dict[str, int] = {}
        for _, s in picked:
            sens_n[s] = sens_n.get(s, 0) + 1
        wins = sum(1 for p in pnls if p > 0)
        print(f"{g:>5.2f} {len(picked):>4} {wins / len(picked) * 100:>4.0f}% "
              f"{sum(pnls):>+10,.0f} {pf:>6.2f} {sum(pnls) / len(picked):>+9,.0f}  "
              + " ".join(f"{k}:{v}" for k, v in sorted(sens_n.items())))


if __name__ == "__main__":
    asyncio.run(main())
