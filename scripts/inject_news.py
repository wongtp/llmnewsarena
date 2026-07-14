"""End-to-end acceptance test (always DRY-RUN, never places real orders).

Builds the full pipeline against live Hyperliquid + Claude and feeds a synthetic
news item so you can confirm: detect -> analyze -> resolve market -> paper trade.

    python scripts/inject_news.py
    python scripts/inject_news.py "TF (@tradfi)" "BREAKING: $TSLA crushes earnings..."
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

try:  # robust output on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.analyzer import Analyzer  # noqa: E402
from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.bus import EventBus  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.models import NewsItem, now_ms  # noqa: E402
from hlbot.news.dedup import Dedup  # noqa: E402
from hlbot.pipeline import Pipeline  # noqa: E402
from hlbot.store.db import Store  # noqa: E402
from hlbot.trading.executor import Executor  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402
from hlbot.trading.position_manager import PositionManager  # noqa: E402
from hlbot.trading.risk import RiskEngine  # noqa: E402

# Representative reconstruction of the bullish MRVL example (edit freely).
DEFAULT_TITLE = "TF (@tradfi)"
DEFAULT_BODY = (
    "BREAKING: $MRVL Marvell Technology signs a multi-year custom AI silicon "
    "supply agreement with a major hyperscaler, expected to add billions in "
    "annual revenue. Analysts calling it a game changer for the company."
)
DEFAULT_LINK = "https://x.com/tradfi/status/2061650323496976699"


async def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TITLE
    body = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BODY

    cfg = Config()
    cfg.runtime.dry_run = True  # hard safety: this script never trades live
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing()))
        return

    store = await Store(path="data/inject_test.sqlite").init()
    bus = EventBus()
    q = bus.subscribe()

    async def printer():
        while True:
            ev = await q.get()
            p = ev.payload
            if ev.topic == "analysis":
                print(f"  ANALYSIS  ticker={p.ticker} dir={p.direction} "
                      f"conf={p.confidence:.2f} stale={p.is_stale} :: {p.rationale}")
            elif ev.topic == "decision":
                if p.action == "enter":
                    print(f"  DECISION  ENTER {p.market.symbol} {p.side} ${p.notional_usd:.0f} "
                          f"@ {p.entry_px:.4f} x{p.leverage} SL {p.stop_loss:.4f} TP {p.take_profit:.4f}")
                else:
                    print(f"  DECISION  REJECT :: {p.reason}")
            elif ev.topic == "trade.open":
                print(f"  PAPER FILL  {p.side} {p.symbol} size={p.size} @ {p.entry_px:.4f}")
            elif ev.topic == "news.skipped":
                print(f"  SKIPPED   :: {p['reason']}")

    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()
    print(f"Universe: {len(universe.equity_symbols())} equities, "
          f"{len(universe.crypto_symbols())} crypto")

    executor = Executor(cfg, hl, store, bus)
    pm = PositionManager(cfg, hl, executor)
    risk = RiskEngine(cfg, hl, store, pm)
    pipeline = Pipeline(bus=bus, store=store, dedup=Dedup(cfg.app.filters),
                        analyzer=Analyzer(cfg), universe=universe, risk=risk,
                        executor=executor, position_manager=pm)

    item = NewsItem(id=str(now_ms()), title=title, body=body, source="Twitter",
                    link=DEFAULT_LINK, time_ms=now_ms(), received_ms=now_ms())
    print(f"\nInjecting: {title} :: {body[:90]}…\n")

    task = asyncio.create_task(printer())
    await pipeline.on_news(item)     # fast enqueue — processing runs in a background task
    await pipeline.drain()           # wait for analyze -> decide -> (paper) trade to finish
    await asyncio.sleep(0.2)         # let the printer flush the last bus events
    task.cancel()

    print(f"\nOpen paper positions: {pm.open_count()}")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
