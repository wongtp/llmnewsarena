"""Part 2 (measurement-only): how much do each model's verdicts WOBBLE across samples?

Does NOT touch trading. For the decision-relevant events (directional, near-gate), re-analyze
each one N times at temperature>0 — holding the point-in-time regime fixed — and measure the
spread. This is the cheap first step before deciding whether ensembling (median-of-N) is worth
building:

  * conf std / range  — how far the confidence moves sample-to-sample.
  * dir-flip rate     — fraction of events where long<->short flips across samples (true
                        disagreement, not just none<->dir).
  * GATE-STRADDLE rate — fraction where some samples clear the gate and some don't: the
                        marginal-trade instability behind the SPCX boundary-variance blind
                        spot. THIS is the number that decides if ensembling can help.

If verdicts barely move, ensembling can't add anything. If they straddle the gate a lot, a
single draw is a coin toss on the trade and median-of-N would stabilize it — but per the
codebase's tail lesson, that must then be A/B'd on dVal before flipping anything.

    .venv/bin/python scripts/multisample.py --arena --samples 5 --temp 0.7 --max-events 60
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import logging
import pathlib
import random
import statistics
import sys

sys.path.insert(0, "src")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time  # noqa: E402

from hlbot.analysis.analyzer import Analyzer  # noqa: E402
from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.backtest.engine import BACKTEST_FRESH_AGE_S  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.models import NewsItem  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

ARENA = [
    ("sonnet",   "data/bt_cache_replay_sonnet.json",        0.80, "claude-sonnet-4-6"),
    ("gpt",      "data/bt_cache_replay_gpt_arena.json",      0.80, "openai:gpt-5.4"),
    ("gemini",   "data/bt_cache_replay_gemini_arena.json",   0.85, "google:gemini-3.5-flash"),
    ("deepseek", "data/bt_cache_replay_deepseek_arena.json", 0.85, "deepseek:deepseek-v4-pro"),
    ("grok",     "data/bt_cache_replay_grok_arena.json",     0.85, "xai:grok-4.3"),
]


def load_regime(regime_key: str):
    p = pathlib.Path(f"data/bt_regime_timeline_{regime_key}.json")
    tl = {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()} if p.exists() else {}
    bnds = sorted(tl)
    def regime_at(t: int) -> str:
        i = bisect.bisect_right(bnds, t) - 1
        return tl.get(bnds[i], "") if i >= 0 else ""
    return regime_at, len(bnds)


def pick_events(cache_path, items_by_id, cutoff, universe, gate, band, max_events, rng):
    """Directional, non-stale, resolvable reads with raw conf in [gate-band, 1.0], in window."""
    cache = json.loads(pathlib.Path(cache_path).read_text(encoding="utf-8"))
    evs = []
    for nid, a in cache.items():
        it = items_by_id.get(nid)
        if it is None or it.time_ms < cutoff:
            continue
        if (a.get("direction") or "none") not in ("long", "short") or a.get("is_stale"):
            continue
        if (a.get("confidence") or 0.0) < gate - band:
            continue
        if universe.resolve(a.get("ticker"), a.get("asset_class")) is None:
            continue
        evs.append((it, a))
    if len(evs) > max_events:
        evs = rng.sample(evs, max_events)
    return evs


async def run_model(analyzer, universe, name, model_full, gate, events, regime_at, samples):
    rows = []
    for it, base in events:
        analyzer.regime_context = regime_at(it.time_ms)
        analyzer.recent_catalysts = ""        # held constant across the N samples
        # N samples of ONE event are independent -> fire concurrently (cuts wall-time ~Nx,
        # matters for the slow thinking models). regime/catalysts are set above, shared.
        outs = await asyncio.gather(*(
            analyzer.analyze(it, universe, age_seconds=BACKTEST_FRESH_AGE_S, model=model_full)
            for _ in range(samples)))
        confs = [o.confidence for o in outs]
        dirs = [o.direction for o in outs]
        tks = [(o.ticker or "").upper() for o in outs]
        dir_set = {d for d in dirs if d in ("long", "short")}
        passes = [c >= gate for c in confs]
        rows.append({
            "id": it.id, "base_dir": base.get("direction"), "base_conf": base.get("confidence"),
            "conf_mean": statistics.mean(confs), "conf_std": statistics.pstdev(confs),
            "conf_min": min(confs), "conf_max": max(confs),
            "dir_flip": len(dir_set) > 1,                    # long<->short within the samples
            "dir_agree": max(dirs.count(d) for d in set(dirs)) / len(dirs),
            "gate_straddle": any(passes) and not all(passes),
            "ticker_agree": max(tks.count(t) for t in set(tks)) / len(tks),
            "cost": sum(o.cost_usd for o in outs),
        })
        print(f"  [{name}] {len(rows)}/{len(events)} "
              f"conf {statistics.mean(confs):.2f}±{statistics.pstdev(confs):.2f}"
              f"{'  FLIP' if len(dir_set)>1 else ''}{'  STRADDLE' if rows[-1]['gate_straddle'] else ''}",
              flush=True)
    return rows


def report(per_model, samples, temp):
    print("\n" + "=" * 92)
    print(f" MULTI-SAMPLE DISPERSION  ({samples} samples/event @ temp {temp}; near-gate directional reads)")
    print("=" * 92)
    hdr = (f"{'model':9} {'events':>6} {'conf std':>8} {'dir-flip%':>9} "
           f"{'GATE-STRADDLE%':>14} {'dir-agree':>9} {'tkr-agree':>9} {'cost$':>7}")
    print(hdr); print("-" * len(hdr))
    for name, rows in per_model.items():
        if not rows:
            print(f"{name:9}  (no events)"); continue
        n = len(rows)
        print(f"{name:9} {n:>6} {statistics.mean(r['conf_std'] for r in rows):>8.3f} "
              f"{sum(r['dir_flip'] for r in rows)/n*100:>8.0f}% "
              f"{sum(r['gate_straddle'] for r in rows)/n*100:>13.0f}% "
              f"{statistics.mean(r['dir_agree'] for r in rows):>9.2f} "
              f"{statistics.mean(r['ticker_agree'] for r in rows):>9.2f} "
              f"{sum(r['cost'] for r in rows):>7.2f}")
    print("\n  conf std       = avg sample-to-sample confidence spread per event.")
    print("  dir-flip%      = events where long<->short flipped across samples (true reversal).")
    print("  GATE-STRADDLE% = events where some samples trade and some don't -> a single draw is")
    print("                   a coin toss on the trade. HIGH => ensembling (median-of-N) could")
    print("                   stabilize the marginal-trade set (then A/B on dVal before shipping).")
    print("  dir/tkr-agree  = modal-sample agreement fraction (1.0 = all samples identical).")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--gate", type=float, default=0.80)
    ap.add_argument("--name", default="model")
    ap.add_argument("--model", default="claude-sonnet-4-6", help="full routing id for --cache mode")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7,
                    help="sampling temperature (live default is 0.0; >0 reveals ensemble spread). "
                         "Models that reject custom temp still vary via inherent nondeterminism.")
    ap.add_argument("--band", type=float, default=0.15,
                    help="include reads with raw conf >= gate-band (the decision-relevant zone)")
    ap.add_argument("--max-events", type=int, default=60, help="cap events/model (bounds cost)")
    ap.add_argument("--history-file", default="data/bt_tg_history_270d.json")
    ap.add_argument("--days", type=float, default=250.0)
    ap.add_argument("--regime-key", default="sonnet_250d")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    logging.getLogger("hlbot").setLevel(logging.ERROR)
    cfg = Config()
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing())); return
    cfg.app.analyzer.temperature = args.temp        # sample at temp>0 (live default 0.0)

    _NI = {f.name for f in __import__("dataclasses").fields(NewsItem)}
    raw = [NewsItem(**{k: v for k, v in d.items() if k in _NI})
           for d in json.loads(pathlib.Path(args.history_file).read_text(encoding="utf-8"))]
    items_by_id = {i.id: i for i in raw}
    cutoff = int((time.time() - args.days * 86400) * 1000)

    hl = HLClient(cfg); await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes); await universe.refresh()
    analyzer = Analyzer(cfg)        # ledger off; no result caching at this layer
    regime_at, n_bnd = load_regime(args.regime_key)
    print(f"regime timeline: {n_bnd} briefs ({args.regime_key})")
    rng = random.Random(args.seed)

    targets = ARENA if args.arena else [(args.name, args.cache, args.gate, args.model)]
    if not args.arena and not args.cache:
        print("Pass --arena or --cache <path> [--gate] [--name] [--model]."); return

    est = sum(min(args.max_events, 999) for _ in targets) * args.samples
    print(f"~{est} analyses ({len(targets)} models x <={args.max_events} events x {args.samples} samples)\n")

    per_model = {}
    for name, cache_path, gate, model_full in targets:
        if not pathlib.Path(cache_path).exists():
            print(f"[{name}] cache not found: {cache_path}"); continue
        events = pick_events(cache_path, items_by_id, cutoff, universe, gate, args.band,
                             args.max_events, rng)
        print(f"[{name}] {len(events)} near-gate events @ gate {gate:.2f} (band {args.band}) ...",
              flush=True)
        per_model[name] = await run_model(analyzer, universe, name, model_full, gate, events,
                                          regime_at, args.samples)

    report(per_model, args.samples, args.temp)
    out = pathlib.Path("data/multisample.json")
    out.write_text(json.dumps({"generated_ms": int(time.time() * 1000), "samples": args.samples,
                               "temp": args.temp, "band": args.band, "per_model": per_model},
                              indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
