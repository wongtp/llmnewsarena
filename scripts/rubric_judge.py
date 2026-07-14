"""Part 3 (diagnostic): does an expert-LLM RUBRIC grade of an analysis predict its PnL?

The codebase's hard lesson is that reasoning which READS better has lost money every time
(exemplars, knowledge bridge). So before letting any judge near a gate, we test the only
thing that matters: out-of-sample, does a frontier judge's reasoning-quality score
correlate with realized/would-be PnL?

For a pooled sample of directional, resolvable reads (across all 5 arena models), we:
  1. compute would-be PnL through the PRODUCTION exit engine (simulate(); free, cached candles),
  2. have a frontier judge (Opus) grade the analysis on a rubric — WITHOUT seeing the outcome —
     on four sub-scores: catalyst_significance (selection), direction_justification (direction),
     reasoning_soundness, confidence_appropriateness, plus a composite,
  3. rank-correlate (Spearman) each score vs PnL, and bucket top- vs bottom-tercile mean PnL.

Reading it against the control arms: if anything predicts PnL it should be
catalyst_significance (the SELECTION edge), NOT direction_justification (direction = ~luck).
A composite that DOESN'T predict PnL = judge is a narrative/diagnostic tool, not a gate.

    .venv/bin/python scripts/rubric_judge.py --n 120 --judge claude-opus-4-8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import random
import re
import statistics
import sys

sys.path.insert(0, "src")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time  # noqa: E402

from anthropic import AsyncAnthropic  # noqa: E402

from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.backtest.engine import simulate  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.models import NewsItem  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

CACHES = {
    "sonnet": "data/bt_cache_replay_sonnet.json", "gpt": "data/bt_cache_replay_gpt_arena.json",
    "gemini": "data/bt_cache_replay_gemini_arena.json",
    "deepseek": "data/bt_cache_replay_deepseek_arena.json", "grok": "data/bt_cache_replay_grok_arena.json",
}
SUBSCORES = ["catalyst_significance", "direction_justification", "reasoning_soundness",
             "confidence_appropriateness", "composite"]

RUBRIC = """You are an expert financial-news trading analyst grading the QUALITY OF REASONING in a
single trade analysis. You are NOT told what the price did afterward — grade ONLY the analysis as a
PROCESS, given the news available at the time. Score each dimension 0-10 (10 = excellent):

- catalyst_significance: how genuinely regime-changing / materially mispricing is this news for the
  named asset on an hours-to-days horizon? (vs noise / already-priced / irrelevant)
- direction_justification: how well does the news justify the specific long/short call?
- reasoning_soundness: is the rationale logical, evidence-grounded, free of hallucination/overreach?
- confidence_appropriateness: is the stated confidence well-calibrated to the strength of evidence?
- composite: your overall 0-10 judgment of whether this is a high-quality, tradeable analysis.

Respond with ONLY a JSON object, no prose:
{"catalyst_significance":N,"direction_justification":N,"reasoning_soundness":N,"confidence_appropriateness":N,"composite":N}"""


def spearman(xs, ys):
    """Spearman rank correlation (Pearson on ranks); no scipy dependency."""
    n = len(xs)
    if n < 3:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:               # average ranks for ties
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def load_pool(items_by_id, cutoff, universe, floor):
    """Pooled (model, news_id, analysis) for directional, resolvable, non-stale reads >= floor."""
    pool = []
    for model, path in CACHES.items():
        p = pathlib.Path(path)
        if not p.exists():
            continue
        for nid, a in json.loads(p.read_text(encoding="utf-8")).items():
            it = items_by_id.get(nid)
            if it is None or it.time_ms < cutoff:
                continue
            if (a.get("direction") or "none") not in ("long", "short") or a.get("is_stale"):
                continue
            if (a.get("confidence") or 0.0) < floor:
                continue
            if universe.resolve(a.get("ticker"), a.get("asset_class")) is None:
                continue
            pool.append((model, it, a))
    return pool


async def judge_one(client, judge_model, it, a) -> dict | None:
    body = f"{it.title}\n{it.body}".strip()[:1500]
    user = (f"{RUBRIC}\n\nNEWS:\n{body}\n\nANALYSIS:\n"
            f"ticker={a.get('ticker')}  direction={a.get('direction')}  "
            f"confidence={a.get('confidence')}\nrationale: {a.get('rationale','')[:800]}")
    try:
        resp = await client.messages.create(model=judge_model, max_tokens=200,
                                             messages=[{"role": "user", "content": user}])
        txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None
        d = json.loads(m.group(0))
        return {k: float(d[k]) for k in SUBSCORES if k in d}
    except Exception as e:  # noqa: BLE001
        logging.getLogger("rubric").debug("judge failed: %r", e)
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="pooled analyses to judge (bounds cost)")
    ap.add_argument("--judge", default="claude-opus-4-8")
    ap.add_argument("--floor", type=float, default=0.70, help="min raw confidence (candles cached >=0.70)")
    ap.add_argument("--history-file", default="data/bt_tg_history_270d.json")
    ap.add_argument("--days", type=float, default=250.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    logging.getLogger("hlbot").setLevel(logging.ERROR)
    cfg = Config()
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing())); return
    r = cfg.app.risk

    _NI = {f.name for f in __import__("dataclasses").fields(NewsItem)}
    raw = [NewsItem(**{k: v for k, v in d.items() if k in _NI})
           for d in json.loads(pathlib.Path(args.history_file).read_text(encoding="utf-8"))]
    items_by_id = {i.id: i for i in raw}
    cutoff = int((time.time() - args.days * 86400) * 1000)

    hl = HLClient(cfg); await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes); await universe.refresh()

    pool = load_pool(items_by_id, cutoff, universe, args.floor)
    rng = random.Random(args.seed)
    if len(pool) > args.n:
        pool = rng.sample(pool, args.n)
    print(f"pool: {len(pool)} analyses (>= {args.floor} conf, directional, resolvable)")

    # would-be PnL through the production exit engine (free; cached candles) ------------------
    rows = []
    for model, it, a in pool:
        mkt = universe.resolve(a.get("ticker"), a.get("asset_class"))
        tr = await simulate(hl, mkt, a["direction"], it.time_ms, a["confidence"],
                            a.get("time_sensitivity") or "none", r, news_id=it.id)
        if tr is None:
            continue
        rows.append({"model": model, "it": it, "a": a, "pnl": tr.pnl, "conf": a["confidence"]})
    print(f"with candle data (PnL computable): {len(rows)}")

    # judge (no outcome shown), bounded concurrency -----------------------------------------
    client = AsyncAnthropic(api_key=cfg.secrets.anthropic_api_key)
    sem = asyncio.Semaphore(args.concurrency)
    done = [0]
    async def grade(row):
        async with sem:
            sc = await judge_one(client, args.judge, row["it"], row["a"])
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  judged {done[0]}/{len(rows)} ...", flush=True)
        if sc:
            row.update(sc)
        return row
    rows = await asyncio.gather(*(grade(r) for r in rows))
    rows = [r for r in rows if "composite" in r]
    print(f"judged OK: {len(rows)}\n")
    if len(rows) < 10:
        print("Too few judged rows for correlation."); return

    pnls = [r["pnl"] for r in rows]
    print("=" * 78)
    print(f" RUBRIC vs PnL  (judge={args.judge}, n={len(rows)} analyses, would-be PnL)")
    print("=" * 78)
    print(f" PnL: mean ${statistics.mean(pnls):+,.0f}  median ${statistics.median(pnls):+,.0f}  "
          f"win {sum(1 for p in pnls if p>0)/len(pnls)*100:.0f}%\n")
    print(f" {'score':28} {'Spearman vs PnL':>16} {'top-T $':>9} {'bot-T $':>9}  read")
    print(" " + "-" * 76)

    def tercile(key):
        s = sorted(rows, key=lambda r: r[key]); k = max(1, len(s) // 3)
        top = [r["pnl"] for r in s[-k:]]; bot = [r["pnl"] for r in s[:k]]
        return statistics.mean(top), statistics.mean(bot)

    def read(rho):
        if rho != rho:
            return "n/a"
        ar = abs(rho)
        sign = "+" if rho >= 0 else "−"
        return (f"{sign}strong" if ar >= .3 else f"{sign}weak" if ar >= .12 else "~none")

    for key in SUBSCORES + ["conf"]:
        if not all(key in r for r in rows):
            continue
        rho = spearman([r[key] for r in rows], pnls)
        top, bot = tercile(key)
        label = key + (" (model's own)" if key == "conf" else "")
        print(f" {label:28} {rho:>+16.3f} {top:>+9,.0f} {bot:>+9,.0f}  {read(rho)}")

    print("\n  Spearman: rank-corr of score vs PnL (+strong>=.30, +weak>=.12, ~none<.12).")
    print("  top-T/bot-T $ = mean PnL of the top vs bottom score tercile.")
    print("  If composite is ~none but catalyst_significance is +: the predictive part is")
    print("  SELECTION (matches control arms), and a rubric GATE should weight that, not 'reads well'.")
    print("  If nothing predicts: rubric is diagnostic/narrative only — keep it away from the gate.")

    out = pathlib.Path("data/rubric_judge.json")
    out.write_text(json.dumps({
        "generated_ms": int(time.time() * 1000), "judge": args.judge, "n": len(rows),
        "spearman": {k: spearman([r[k] for r in rows if k in r], [r["pnl"] for r in rows if k in r])
                     for k in SUBSCORES + ["conf"]},
        "rows": [{"model": r["model"], "id": r["it"].id, "pnl": r["pnl"], "conf": r["conf"],
                  **{k: r[k] for k in SUBSCORES if k in r}} for r in rows],
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
