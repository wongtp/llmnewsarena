"""Control arms for the arena training results — is the leaderboard skill or luck?

Each model's 250d leaderboard PnL is a handful of tail trades. Before trusting a ranking
(or funding wallets) we place each model's REAL replay PnL inside the distribution of a
NULL strategy that runs the *identical* production machinery — regime, catalyst memory,
haircuts, similarity dedup, confidence-tiered sizing, caps, cooldown, exposure budget,
contrary-exit, the production exit engine — and randomizes ONLY the one thing the model
claims edge on:

  RANDOM-DIRECTION  — every directional verdict's long/short is replaced by a coin flip,
                      same events, same gate, same everything else. Null = "right machinery,
                      random side". If the model's real PnL doesn't clear the ~95th
                      percentile of this null, its direction calls aren't beating chance.

This drives the REAL run_live_replay (the same function that produced the leaderboard) via
a default-off `analysis_transform` hook, so the null is leaderboard-comparable by
construction — NOT the stripped per-signal sweep (which CLAUDE.md warns understates PnL by
~the $4k the risk layer is worth). Analyses/candles/regime are all disk-cached, so a full
replay is ~0.2s and thousands of Monte-Carlo runs cost $0 and no Claude calls.

References (opportunity cost, not nulls): buy-and-hold BTC, buy-and-hold SPY, cash ($0).

    .venv/bin/python scripts/control_arms.py --arena                 # all 5 at their gates
    .venv/bin/python scripts/control_arms.py --arena --seeds 2000
    .venv/bin/python scripts/control_arms.py --cache data/bt_cache_replay_gpt_arena.json --gate 0.80
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import pathlib
import random
import statistics
import sys
import urllib.request
from dataclasses import replace

sys.path.insert(0, "src")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time  # noqa: E402

from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.backtest.engine import fetch_candles, pick_entry, run_live_replay  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.models import NewsItem  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

# model -> (replay cache, Phase-2 gate). Mirrors the live arena entrants.
ARENA = [
    ("sonnet",   "data/bt_cache_replay_sonnet.json",        0.80),
    ("gpt",      "data/bt_cache_replay_gpt_arena.json",      0.80),
    ("gemini",   "data/bt_cache_replay_gemini_arena.json",   0.85),
    ("deepseek", "data/bt_cache_replay_deepseek_arena.json", 0.85),
    ("grok",     "data/bt_cache_replay_grok_arena.json",     0.85),
]
OPP = {"long": "short", "short": "long"}


def make_flip(seed: int):
    """RANDOM-DIRECTION analysis_transform: replace every directional verdict's side with a
    coin flip (same events, same sizing). Returns a NEW Analysis (cached object is reused)."""
    rng = random.Random(seed)
    def transform(a):
        if a.direction in ("long", "short") and rng.random() < 0.5:
            return replace(a, direction=OPP[a.direction])
        return a
    return transform


def make_select(selected: set, conf: float):
    """RANDOM-SELECTION analysis_transform: trade ONLY `selected` ids, at a UNIFORM confidence
    (so confidence-tiered sizing isn't a confound — this isolates *which events*, holding size
    + own direction fixed). Non-selected directional reads are nulled to 'none' (skipped)."""
    def transform(a):
        if a.news_id in selected:
            return replace(a, confidence=conf)
        if a.direction in ("long", "short"):
            return replace(a, direction="none")
        return a
    return transform


def build_pool(cache_path, time_by_id, cutoff, universe, blacklist, floor):
    """The model's opportunity set for the selection control: every directional, non-stale,
    resolvable, non-blacklisted read with raw confidence >= floor, in window."""
    cache = json.loads(pathlib.Path(cache_path).read_text(encoding="utf-8"))
    pool = []
    for nid, a in cache.items():
        t = time_by_id.get(nid)
        if t is None or t < cutoff:
            continue
        if (a.get("direction") or "none") not in ("long", "short") or a.get("is_stale"):
            continue
        if (a.get("confidence") or 0.0) < floor:
            continue
        m = universe.resolve(a.get("ticker"), a.get("asset_class"))
        if m is None or m.symbol.upper() in blacklist:
            continue
        pool.append(nid)
    return pool


async def _replay(cfg, hl, universe, raw_items, model_full, cache_path, env, transform=None):
    res = await run_live_replay(
        cfg, hl, universe, raw_items=raw_items, days=env["days"], model=model_full,
        cache_path=cache_path,
        regime_refresh_seconds=cfg.app.regime_refresh_seconds,
        regime_lookback_days=cfg.app.regime_live_lookback_days,
        catalyst_memory_days=cfg.app.catalyst_memory_days,
        regime_key=env["regime_key"], regime_model=env["regime_model"],
        analysis_transform=transform, offline_only=True)   # never hit the live API on a cache miss
    trades = res.get("trades", [])
    return sum(t.pnl for t in trades), len(trades), {t.news_id for t in trades}


def _pct_rank(value: float, dist: list[float]) -> float:
    return sum(1 for d in dist if d <= value) / len(dist) if dist else float("nan")


def _verdict(pct: float) -> str:
    if pct != pct:
        return "n/a"
    return "SKILL" if pct >= 0.95 else "lean" if pct >= 0.80 else "~luck" if pct >= 0.50 else "WORSE"


async def btc_buy_hold(hl, start_ms: int, end_ms: int) -> float | None:
    candles = await fetch_candles(hl, "BTC", start_ms - 3600_000, end_ms + 3600_000, "1h")
    if not candles:
        return None
    _, base = pick_entry(candles, start_ms)
    return ((float(candles[-1]["c"]) - base) / base * 100.0) if base else None


def spy_buy_hold(start_ms: int, end_ms: int, key: str, secret: str) -> float | None:
    """SPY daily buy-and-hold % over the window (Alpaca IEX, free tier)."""
    if not (key and secret):
        return None
    s = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).date().isoformat()
    e = dt.datetime.fromtimestamp(end_ms / 1000, dt.timezone.utc).date().isoformat()
    url = ("https://data.alpaca.markets/v2/stocks/SPY/bars"
           f"?timeframe=1Day&adjustment=split&feed=iex&limit=10000&start={s}&end={e}")
    try:
        req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key,
                                                   "APCA-API-SECRET-KEY": secret})
        with urllib.request.urlopen(req, timeout=20) as f:
            bars = (json.load(f).get("bars")) or []
        if len(bars) < 2:
            return None
        return (bars[-1]["c"] - bars[0]["o"]) / bars[0]["o"] * 100.0
    except Exception:  # noqa: BLE001
        return None


async def run_one(cfg, hl, universe, raw_items, name, cache_path, gate, seeds, env):
    if not pathlib.Path(cache_path).exists():
        print(f"  [{name}] cache not found: {cache_path}")
        return None
    cfg.app.risk.confidence_threshold = gate           # replay at THIS model's gate
    model_full = {"sonnet": "claude-sonnet-4-6", "gpt": "openai:gpt-5.4",
                  "gemini": "google:gemini-3.5-flash", "deepseek": "deepseek:deepseek-v4-pro",
                  "grok": "xai:grok-4.3"}.get(name, "claude-sonnet-4-6")

    # --- baseline real replay -> leaderboard PnL + the actually-traded id set ----------------
    actual, n, traded = await _replay(cfg, hl, universe, raw_items, model_full, cache_path, env)

    # --- RANDOM-DIRECTION null: same events, coin-flip side (own sizing) ---------------------
    dir_dist = []
    for s in range(seeds):
        pnl, _, _ = await _replay(cfg, hl, universe, raw_items, model_full, cache_path, env,
                                  transform=make_flip(s))
        dir_dist.append(pnl)
        if (s + 1) % 250 == 0:
            print(f"  [{name}] direction null {s+1}/{seeds} ...", flush=True)
    dir_pct = _pct_rank(actual, dir_dist)

    # --- RANDOM-SELECTION null: random K of the >=floor pool, uniform size, own direction ----
    pool = build_pool(cache_path, env["time_by_id"], env["cutoff"], universe,
                      env["blacklist"], env["sel_floor"])
    K = len(traded)
    sel_actual, sel_dist, sel_pct = None, [], float("nan")
    if K and len(pool) > K:
        # 'actual' selection arm at the SAME uniform size as the null -> comparable.
        sel_actual, _, _ = await _replay(cfg, hl, universe, raw_items, model_full, cache_path,
                                         env, transform=make_select(traded, env["sel_conf"]))
        rng = random.Random(env["sel_seed"])
        for s in range(seeds):
            pick = set(rng.sample(pool, K))
            pnl, _, _ = await _replay(cfg, hl, universe, raw_items, model_full, cache_path, env,
                                      transform=make_select(pick, env["sel_conf"]))
            sel_dist.append(pnl)
            if (s + 1) % 250 == 0:
                print(f"  [{name}] selection null {s+1}/{seeds} ...", flush=True)
        sel_pct = _pct_rank(sel_actual, sel_dist)
    else:
        print(f"  [{name}] selection control skipped (K={K}, pool={len(pool)})", flush=True)

    print(f"  [{name}] actual=${actual:+,.0f} ({n} tr) | DIR pct={dir_pct*100:.0f}% {_verdict(dir_pct)}"
          f" | SEL pct={(sel_pct*100 if sel_pct==sel_pct else 0):.0f}% {_verdict(sel_pct)}"
          f" (uniform-size actual ${(sel_actual or 0):+,.0f}, pool {len(pool)})", flush=True)
    return {"name": name, "gate": gate, "trades": n, "actual": actual,
            "dir_dist": dir_dist, "dir_pct": dir_pct,
            "sel_actual": sel_actual, "sel_dist": sel_dist, "sel_pct": sel_pct, "pool": len(pool)}


def _pctf(p):
    return f"{p*100:>3.0f}%" if p == p else "  -"


def report(rows, btc_ref, spy_ref):
    def q(d, p):
        return statistics.quantiles(d, n=100)[p - 1] if len(d) >= 100 else (
            statistics.median(d) if d else float("nan"))
    print("\n" + "=" * 100)
    print(" CONTROL ARMS — real replay PnL vs two NULL strategies through the SAME production machinery")
    print("=" * 100)
    refs = [f"buy-hold BTC {btc_ref:+.1f}%" if btc_ref is not None else "buy-hold BTC n/a",
            f"buy-hold SPY {spy_ref:+.1f}%" if spy_ref is not None else "buy-hold SPY n/a",
            "cash $0"]
    print(" reference (opportunity cost): " + "  |  ".join(refs) + "\n")
    hdr = (f"{'model':9} {'gate':>4} {'trades':>6} {'actual':>9}   "
           f"{'DIR p50':>8} {'p95':>8} {'pct':>4} {'verdict':>7}   "
           f"{'SEL p50':>8} {'p95':>8} {'pct':>4} {'verdict':>7}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["actual"]):
        d, s = r["dir_dist"], r["sel_dist"]
        sel_a = r["sel_actual"]
        sel_seg = (f"{q(s,50):>+8,.0f} {q(s,95):>+8,.0f} {_pctf(r['sel_pct'])} "
                   f"{_verdict(r['sel_pct']):>7}" if s else f"{'(no pool)':>33}")
        print(f"{r['name']:9} {r['gate']:>4.2f} {r['trades']:>6} {r['actual']:>+9,.0f}   "
              f"{q(d,50):>+8,.0f} {q(d,95):>+8,.0f} {_pctf(r['dir_pct'])} {_verdict(r['dir_pct']):>7}   "
              f"{sel_seg}")
    print("\n  DIR (random-direction): same events, coin-flip side, OWN sizing — actual = leaderboard.")
    print("       pct = % of coin-flip runs the real PnL beats. High = the model's DIRECTION calls add edge.")
    print("  SEL (random-selection): random K of the model's >=floor reads, UNIFORM size, OWN direction")
    print("       (actual arm is the model's own K at the same uniform size, so it's comparable — NOT the")
    print("       leaderboard $). High pct = the GATE picks better events than random plausible reads.")
    print("  verdict: SKILL >=95 · lean >=80 · ~luck >=50 · WORSE <50")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", action="store_true", help="run all 5 arena models at their gates")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--gate", type=float, default=0.80)
    ap.add_argument("--name", default="model", help="label/model for --cache mode")
    ap.add_argument("--history-file", default="data/bt_tg_history_270d.json")
    ap.add_argument("--days", type=float, default=250.0)
    ap.add_argument("--seeds", type=int, default=1000, help="Monte Carlo runs per null per model "
                    "(warm ~0.06s/run; 1000 = ~12min for the full arena's two controls)")
    ap.add_argument("--sel-floor", type=float, default=0.70,
                    help="min raw confidence for the selection-control opportunity pool")
    ap.add_argument("--sel-conf", type=float, default=0.95,
                    help="uniform confidence forced on selected events (removes the sizing confound)")
    ap.add_argument("--regime-key", default="sonnet_250d")
    args = ap.parse_args()

    logging.getLogger("hlbot").setLevel(logging.ERROR)   # silence per-run replay chatter

    cfg = Config()
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing()))
        return

    _NI = {f.name for f in __import__("dataclasses").fields(NewsItem)}
    raw_items = [NewsItem(**{k: v for k, v in d.items() if k in _NI})
                 for d in json.loads(pathlib.Path(args.history_file).read_text(encoding="utf-8"))]
    time_by_id = {i.id: i.time_ms for i in raw_items}
    cutoff = int((time.time() - args.days * 86400) * 1000)

    # shared regime brief (matches the arena replays — every model saw the same Sonnet brief)
    env = {"days": args.days, "regime_key": args.regime_key, "regime_model": "claude-sonnet-4-6",
           "time_by_id": time_by_id, "cutoff": cutoff,
           "blacklist": {b.upper() for b in cfg.app.filters.market_blacklist},
           "sel_floor": args.sel_floor, "sel_conf": args.sel_conf, "sel_seed": 12345}

    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()

    targets = ARENA if args.arena else [(args.name, args.cache, args.gate)]
    if not args.arena and not args.cache:
        print("Pass --arena or --cache <path> [--gate] [--name]."); return

    t0 = time.perf_counter()
    rows = []
    for name, cache_path, gate in targets:
        print(f"[{name}] {cache_path} @ gate {gate:.2f}  ({args.seeds} null runs) ...", flush=True)
        row = await run_one(cfg, hl, universe, raw_items, name, cache_path, gate, args.seeds, env)
        if row:
            rows.append(row)
    if not rows:
        print("No model produced a trade set."); return

    end_ms = int(max(i.time_ms for i in raw_items))
    btc_ref = await btc_buy_hold(hl, cutoff, end_ms)
    spy_ref = spy_buy_hold(cutoff, end_ms, cfg.secrets.alpaca_api_key, cfg.secrets.alpaca_api_secret)
    report(rows, btc_ref, spy_ref)
    print(f"\n  ({time.perf_counter()-t0:.0f}s, {args.seeds} seeds/null/model)")

    # Persist for the arena UI / later inspection.
    def qp(d, p):
        return statistics.quantiles(d, n=100)[p - 1] if len(d) >= 100 else None
    out = pathlib.Path("data/control_arms.json")
    out.write_text(json.dumps({
        "generated_ms": int(time.time() * 1000), "days": args.days, "seeds": args.seeds,
        "btc_buy_hold_pct": btc_ref, "spy_buy_hold_pct": spy_ref,
        "models": [{"name": r["name"], "gate": r["gate"], "trades": r["trades"],
                    "actual": r["actual"], "pool": r["pool"],
                    "dir_pct": r["dir_pct"], "dir_p50": statistics.median(r["dir_dist"]),
                    "dir_p5": qp(r["dir_dist"], 5), "dir_p95": qp(r["dir_dist"], 95),
                    "sel_pct": (r["sel_pct"] if r["sel_pct"] == r["sel_pct"] else None),
                    "sel_actual": r["sel_actual"],
                    "sel_p50": (statistics.median(r["sel_dist"]) if r["sel_dist"] else None),
                    "sel_p95": qp(r["sel_dist"], 95)} for r in rows],
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
