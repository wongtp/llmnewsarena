"""Paper smoke for the arena: build the 5-lane pipeline, inject one synthetic news item,
and report each lane's verdict -> decision -> paper position. Validates the fan-out, per-lane
gates, dry-run execution, and the model_id partition end-to-end. Needs .env (HL + provider keys).

    .venv/bin/python scripts/arena_smoke.py
"""
import asyncio
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - Windows console chokes on emoji otherwise
    pass

import logging  # noqa: E402

logging.disable(logging.INFO)

from hlbot.analysis.analyzer import Analyzer  # noqa: E402
from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.arena.lane import TradingLane  # noqa: E402
from hlbot.arena.pipeline import ArenaPipeline  # noqa: E402
from hlbot.bus import EventBus  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.models import NewsItem, now_ms  # noqa: E402
from hlbot.news.dedup import Dedup  # noqa: E402
from hlbot.store.db import Store  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

HEADLINE = ("NVIDIA reports Q3 revenue of $35.1B, up 94% YoY, beating consensus of $33.2B; "
            "guides Q4 revenue to ~$37.5B vs ~$35.9B expected.")


async def main() -> None:
    cfg = Config()
    cfg.runtime.dry_run = True
    cfg.app.analyzer.max_tokens = max(cfg.app.analyzer.max_tokens, cfg.app.arena.max_tokens)

    store = await Store("data/arena_smoke.sqlite").init()
    bus = EventBus()
    hl = HLClient(cfg)
    await hl.connect()
    uni = Universe(hl, cfg.app.filters.allowed_dexes)
    await uni.refresh()
    az = Analyzer(cfg)
    dedup = Dedup(cfg.app.filters)

    lanes = [TradingLane(key=e.key, model=e.model, gate=e.gate,
                         capital_usd=cfg.app.arena.capital_per_model_usd, config=cfg, hl=hl,
                         store=store, bus=bus, universe=uni) for e in cfg.app.arena.entrants]
    print(f"lanes: {', '.join(f'{l.key}@{l.gate}' for l in lanes)}")
    pipeline = ArenaPipeline(bus=bus, store=store, dedup=dedup, analyzer=az,
                             universe=uni, lanes=lanes)

    item = NewsItem(id="smoke:nvda", title="NVDA earnings", body=HEADLINE,
                    source="Telegram:trad_fin", link=None, time_ms=now_ms(), received_ms=now_ms())
    await pipeline.on_news(item)
    await pipeline.drain()

    snap = await store.snapshot()
    analyses = {a["model"]: a for a in snap["analyses"]}
    decisions = {}  # model not on Decision; correlate by news_id+symbol via the lane verdicts
    positions = snap["positions"]
    print("\n%-9s %-26s %-6s %5s  %-7s  %s" % ("lane", "model", "dir", "conf", "decision", "position"))
    print("-" * 86)
    for l in lanes:
        a = analyses.get(l.model, {})
        pos = next((p for p in positions if p.get("model_id") == l.model), None)
        posdesc = (f"{pos['side']} {pos['symbol']} ${pos['notional']:.0f} @{pos['entry_px']:.2f}"
                   if pos else "-")
        dec = "ENTER" if pos else ("gate" if a.get("direction") != "none" else "none")
        print("%-9s %-26s %-6s %4.0f%%  %-7s  %s" % (
            l.key, l.model, a.get("direction", "?"), 100 * a.get("confidence", 0), dec, posdesc))
    print(f"\nanalyses={len(snap['analyses'])} decisions={len(snap['decisions'])} "
          f"paper_positions={len(positions)}")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
