"""A/B the analyzer across Claude models on real historical news items.

    python scripts/compare_models.py --query "32 bitcoin" --n 3
    python scripts/compare_models.py --query saylor --models haiku,sonnet,opus

Runs each matching news item through each model (presented as fresh) and prints the
verdicts side by side. Useful for deciding whether a stronger model is worth it.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.analyzer import Analyzer  # noqa: E402
from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.backtest.engine import BACKTEST_FRESH_AGE_S, fetch_history  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="32 bitcoin")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--models", default="haiku,sonnet,opus")
    ap.add_argument("--days", type=float, default=7.0)
    args = ap.parse_args()

    cfg = Config()
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing()))
        return

    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()
    analyzer = Analyzer(cfg)

    items = await fetch_history(5000)
    q = args.query.lower()
    matches = [i for i in items if q in i.text.lower()][:args.n]
    if not matches:
        print(f"No items matched '{args.query}'.")
        return

    models = [m.strip() for m in args.models.split(",")]
    print(f"Comparing {models} on {len(matches)} item(s) matching '{args.query}'\n")
    for it in matches:
        print("=" * 80)
        print(f"{it.source or 'TW'} | {it.title[:100]}")
        if it.body:
            print(f"  {it.body[:160]}")
        print("-" * 80)
        results = await asyncio.gather(*[
            analyzer.analyze(it, universe, age_seconds=BACKTEST_FRESH_AGE_S, model=MODELS.get(m, m))
            for m in models
        ])
        for m, a in zip(models, results):
            flag = " STALE" if a.is_stale else ""
            err = f"  ERROR: {a.error}" if a.error else ""
            print(f"  {m:7} -> {str(a.ticker):5} {a.direction:5} conf={a.confidence:.0%} "
                  f"[{a.time_sensitivity}]{flag}{err}")
            print(f"            {a.rationale[:150]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
