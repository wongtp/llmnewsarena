"""Arena entrypoint: one news feed -> 5 competing models, each on its own wallet/capital.

    .venv/bin/python -m hlbot.main_arena      # paper by default; flip lanes live at go-live

Replaces the single-model production bot (one Telethon session, ingested once and fanned out).
Paper/dry-run uses a shared read-only HLClient + virtual capital; at go-live each lane gets its
own funded wallet. Arena dashboard on config.app.arena.ui_port (default 8001).
"""
from __future__ import annotations

import asyncio
import logging

import uvicorn

from .analysis.analyzer import Analyzer
from .analysis.universe import Universe
from .arena.lane import TradingLane
from .arena.pipeline import ArenaPipeline
from .bus import EventBus
from .config import Config
from .main import (
    backfill_news_and_regime,
    cache_keepwarm_loop,
    catalyst_memory_loop,
    feed_watchdog,
    periodic_refresh,
    refresh_catalyst_memory,
    regime_refresh_loop,
    setup_logging,
    supervised,
)
from .news.dedup import Dedup
from .news.telegram_source import TelegramSource
from .news.tree_client import TreeClient
from .store.db import Store
from .trading.hl_client import HLClient
from .ui.arena_server import create_arena_app

log = logging.getLogger("hlbot.arena")

ARENA_DB = "data/arena.sqlite"
# Own state file: sharing data/runtime_state.json with the production entrypoint would let
# arena saves clobber the operator's production dry/live + kill-switch toggles (and vice versa).
ARENA_RUNTIME_STATE = "data/arena_runtime_state.json"
# Own token ledger: the shared data/token_usage.json mixes pre-arena production Sonnet usage
# into the arena's per-model fairness comparison (and a future production run on this box
# would re-pollute it). Seeded at the 2026-07-13 cutover as shared-ledger minus the
# 2026-06-10 migration baseline, i.e. exactly the arena's own usage.
ARENA_TOKEN_LEDGER = "data/arena_token_usage.json"


async def arena_capital_loop(lanes: list[TradingLane], bus: EventBus, store,
                             interval: int = 30) -> None:
    """Publish each lane's live stats (equity, realized, uPnL, trades, win rate) for the
    leaderboard + the trailing point on the equity curve."""
    import time as _t
    while True:
        await asyncio.sleep(interval)
        rows = []
        for lane in lanes:
            try:
                stats = await store.lane_stats(lane.model, dry_run=not lane.live)
                n, wins, realized = stats["n"], stats["wins"], stats["realized"]
                upnl = lane.open_unrealized()
                halted, halt_why = lane.risk.config.runtime.is_daily_halted(lane.model)
                rows.append({"model": lane.model, "key": lane.key, "starting": lane.capital_usd,
                             "current": lane.capital_usd + realized + upnl, "realized": realized,
                             "unrealized": upnl, "open": lane.pm.open_count(),
                             "trades": n, "wins": wins, "win_rate": (wins / n) if n else None,
                             "halted": halted, "halt_reason": halt_why})
            except Exception:  # noqa: BLE001 - display only, but a permanently erroring lane
                log.warning("capital readout failed for %s", lane.key)   # must stay visible
        await bus.publish("arena.capital", {"lanes": rows, "ts": int(_t.time())})


async def run() -> None:
    setup_logging()
    config = Config(runtime_state_file=ARENA_RUNTIME_STATE)
    # Both the Analyzer (writer) and the dashboard's usage endpoint (reader) resolve the
    # ledger through this config field — one assignment routes them to the arena's own file.
    config.app.token_ledger_file = ARENA_TOKEN_LEDGER
    acfg = config.app.arena

    missing = config.secrets.missing(require_tree=config.app.enable_tree_feed)
    if missing:
        log.error("Missing required secrets in .env: %s — aborting.", ", ".join(missing))
        return

    # Thinking models (DeepSeek V4 Pro) truncate->empty below 2048 output tokens.
    config.app.analyzer.max_tokens = max(config.app.analyzer.max_tokens, acfg.max_tokens)

    store = await Store(ARENA_DB).init()
    bus = EventBus()
    hl = HLClient(config)   # shared read-only market data (paper); per-wallet at go-live
    await hl.connect()
    universe = Universe(hl, config.app.filters.allowed_dexes)
    if await universe.refresh() == 0:
        log.error("No tradable markets loaded; aborting.")
        return

    analyzer = Analyzer(config, ledger_path=config.app.token_ledger_file)
    await refresh_catalyst_memory(analyzer, store, config)
    await backfill_news_and_regime(config, store, analyzer)

    dedup = Dedup(config.app.filters)
    try:
        dedup.restore(await store.recent_news_ids(dedup.memory))
    except Exception:  # noqa: BLE001
        log.warning("Could not restore dedup memory")

    lanes: list[TradingLane] = []
    n_live = 0
    for e in acfg.entrants:
        # A lane goes LIVE only if config marks it live AND its wallet secrets are present;
        # otherwise it's paper on the shared read-only client. Live lanes get their own
        # per-wallet client so orders/reconciliation hit the right funded wallet.
        addr, sec = config.secrets.arena_wallet(e.wallet)
        want_live = bool(e.live and addr and sec)
        if want_live:
            hl_lane = HLClient(config, address=addr, secret_key=sec)
            await hl_lane.connect()
            n_live += 1
        else:
            hl_lane = hl
        lane = TradingLane(key=e.key, model=e.model, gate=e.gate,
                           capital_usd=acfg.capital_per_model_usd, config=config,
                           hl=hl_lane, store=store, bus=bus, universe=universe, live=want_live)
        await lane.restore()
        lanes.append(lane)
        log.info("  lane %-9s model=%-28s gate=%.2f  [%s]", e.key, e.model, e.gate,
                 "LIVE $REAL" if want_live else "paper")
    # Global flag is the UI badge only; per-lane overrides govern actual paper/live execution.
    config.runtime.dry_run = (n_live == 0)
    log.warning("ARENA: %d/%d lanes LIVE (real money), %d paper — $%s capital each",
                n_live, len(lanes), len(lanes) - n_live, f"{acfg.capital_per_model_usd:,.0f}")

    pipeline = ArenaPipeline(bus=bus, store=store, dedup=dedup, analyzer=analyzer,
                             universe=universe, lanes=lanes)
    tree = TreeClient(config, pipeline.on_news)
    tg_source = TelegramSource(config, pipeline.on_news)

    app = create_arena_app(bus, store, config, lanes, hl=hl)
    userver = uvicorn.Server(uvicorn.Config(
        app, host=config.app.ui.host, port=acfg.ui_port, log_level="warning"))
    log.info("Arena dashboard: http://%s:%d", config.app.ui.host, acfg.ui_port)
    mode = "PAPER" if n_live == 0 else f"{n_live} LIVE / {len(lanes) - n_live} paper"
    await bus.publish("status", f"Arena started ({mode}) — {len(lanes)} models")

    def _task(name, factory):
        return asyncio.create_task(supervised(name, factory), name=name)

    tasks = [
        _task("telegram_src", tg_source.run),
        _task("mids", lambda: hl.mid_cache_loop(config.app.filters.allowed_dexes)),
        _task("refresh", lambda: periodic_refresh(universe, config.app.universe_refresh_seconds)),
        _task("cache_warm", lambda: cache_keepwarm_loop(analyzer, universe, config)),
        _task("regime", lambda: regime_refresh_loop(analyzer, store, config)),
        _task("catalyst_mem", lambda: catalyst_memory_loop(analyzer, store, config)),
        _task("watchdog", lambda: feed_watchdog(pipeline, bus, config)),
        _task("capital", lambda: arena_capital_loop(lanes, bus, store)),
        _task("ui", userver.serve),
    ]
    tasks += [_task(f"pm:{lane.key}", lane.pm.run) for lane in lanes]
    # Live lanes use their own wallet client — give it a warm mid cache too (paper lanes share
    # the main client's "mids" loop above).
    tasks += [_task(f"mids:{lane.key}",
                    lambda l=lane: l.hl.mid_cache_loop(config.app.filters.allowed_dexes))
              for lane in lanes if lane.live]
    if config.app.enable_tree_feed:
        tasks.append(_task("tree", tree.run))

    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutting down arena…")
    finally:
        tree.stop()
        for lane in lanes:
            lane.pm.stop()
        await tg_source.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await pipeline.drain()
        await store.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
