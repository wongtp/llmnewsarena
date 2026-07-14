"""Close every OPEN dry-run (paper) position before a live run.

A paper position left open when you switch to live gets restored on startup and occupies a
concurrency/exposure slot (and would block live entries if max_concurrent is small). Run this
once before flipping to live so the session starts clean.

Closes ONLY paper positions (dry_run=True) at the current mid — it never sends a real order. A
genuine LIVE open position is reported and left untouched (close those deliberately).

    python scripts/close_paper_positions.py            # close them
    python scripts/close_paper_positions.py --dry      # just list what would be closed
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import fields as dataclass_fields

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.bus import EventBus  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.models import Position  # noqa: E402
from hlbot.store.db import Store  # noqa: E402
from hlbot.trading.executor import Executor  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="list open positions but don't close them")
    args = ap.parse_args()

    cfg = Config()
    store = await Store().init()
    rows = await store.open_positions()
    if not rows:
        print("No open positions in the store — clean slate.")
        await store.close()
        return

    valid = {f.name for f in dataclass_fields(Position)}
    positions = [Position(**{k: v for k, v in d.items() if k in valid}) for d in rows]
    paper = [p for p in positions if p.dry_run]
    live = [p for p in positions if not p.dry_run]
    print(f"Open positions: {len(positions)}  ({len(paper)} paper, {len(live)} LIVE)")
    for p in live:
        print(f"  [LIVE] {p.symbol} {p.side} size={p.size} — left untouched; close deliberately.")

    if not paper:
        await store.close()
        return
    if args.dry:
        for p in paper:
            print(f"  [paper] {p.symbol} {p.side} size={p.size} @ {p.entry_px} — WOULD close")
        await store.close()
        return

    hl = HLClient(cfg)
    await hl.connect()
    ex = Executor(cfg, hl, store, EventBus())
    for p in paper:
        closed = await ex.close(p, "cleared before live run")   # pos.dry_run=True -> paper close
        if closed:
            print(f"  [paper] closed {p.symbol} {p.side} @ {closed.exit_px:.4f} "
                  f"pnl={closed.pnl_usd:+.2f}")
        else:
            print(f"  [paper] {p.symbol}: close did not complete — re-run or inspect.")
    await store.close()
    print("Done. Paper book is flat — safe to start the live run.")


if __name__ == "__main__":
    asyncio.run(main())
