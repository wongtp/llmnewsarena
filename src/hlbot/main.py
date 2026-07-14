"""Entrypoint: wires every component and runs the bot.

    python -m hlbot.main          # run the bot (mode from config.yaml dry_run)

Dashboard at http://127.0.0.1:8000 ; Telegram alerts if configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time

import uvicorn

from .analysis.analyzer import Analyzer
from .analysis.universe import Universe
from .bus import EventBus
from .config import Config
from .news.dedup import Dedup
from .news.telegram_source import TelegramSource
from .news.tree_client import TreeClient
from .notify.telegram import TelegramNotifier
from .pipeline import Pipeline
from .store.db import Store
from .trading.executor import Executor
from .trading.hl_client import HLClient
from .trading.position_manager import PositionManager
from .trading.risk import RiskEngine
from .ui.server import create_app

log = logging.getLogger("hlbot")


CAPITAL_BASELINE_FILE = "data/capital_baseline.json"


class CapitalTracker:
    """Tracks account equity for the dashboard. 'starting' is a PERSISTENT baseline — the equity
    when you first went live — saved to disk so it survives restarts (your true since-launch P&L
    reference). 'current' is refreshed periodically. Equity is the unified total (spot + perp), so
    it reflects realized + open PnL. reset() re-baselines to the current equity; use it after a
    deposit/withdrawal, when the old baseline no longer reflects deployed capital."""

    def __init__(self, hl: HLClient, path: str | None = None):
        self.hl = hl
        # Resolved at call time (not def time) so the test fixture's redirect of the module
        # constant actually isolates default-constructed trackers from real data/.
        self.path = path or CAPITAL_BASELINE_FILE
        self.current: float | None = None
        self.starting: float | None = None
        self.started_at: int | None = None
        self._load()

    def _load(self) -> None:
        try:
            p = pathlib.Path(self.path)
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                s = d.get("starting")
                self.starting = float(s) if s is not None else None
                self.started_at = d.get("started_at")
        except Exception:  # noqa: BLE001 - corrupt/missing: re-baseline on next refresh
            self.starting, self.started_at = None, None

    def _save(self) -> None:
        try:
            p = pathlib.Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"starting": self.starting, "started_at": self.started_at}),
                         encoding="utf-8")
        except Exception:  # noqa: BLE001 - display-only, never break trading
            pass

    async def refresh(self) -> float | None:
        v = await self.hl.account_value()
        if v is not None:
            self.current = v
            if self.starting is None:          # first ever reading -> persist as the baseline
                self.starting = v
                self.started_at = int(time.time() * 1000)
                self._save()
        return v

    def reset(self) -> None:
        """Re-baseline 'starting' to the current equity and persist it."""
        if self.current is not None:
            self.starting = self.current
            self.started_at = int(time.time() * 1000)
            self._save()

    def state(self) -> dict:
        return {"starting": self.starting, "current": self.current,
                "started_at": self.started_at}


async def capital_loop(tracker: "CapitalTracker", bus: EventBus, interval: int = 60) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await tracker.refresh()
            await bus.publish("capital", tracker.state())
        except Exception:  # noqa: BLE001
            log.exception("Capital refresh failed")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


async def periodic_refresh(universe: Universe, seconds: int) -> None:
    while True:
        await asyncio.sleep(seconds)
        try:
            await universe.refresh()
        except Exception:  # noqa: BLE001
            log.exception("Universe refresh failed")


async def cache_keepwarm_loop(analyzer: Analyzer, universe: Universe, config: Config) -> None:
    """Keep the analyzer's cached prompt prefix hot (see Analyzer.maybe_warm_cache): the
    first headline after a quiet stretch then never pays the cold-cache prefill/rewrite on
    the news->order critical path, and sparse calls bill at the 0.1x cached-read rate
    instead of re-writing the prefix at 2x. Warms immediately at startup (and shortly after
    each regime/universe change, picked up via the prefix fingerprint), then whenever the
    prefix has gone unused for cache_keepwarm_seconds."""
    interval = config.app.analyzer.cache_keepwarm_seconds
    if interval <= 0:
        return
    check = max(30, min(interval // 5, 300))
    while True:
        try:
            await analyzer.maybe_warm_cache(universe, interval)
        except Exception:  # noqa: BLE001 - best-effort: a failed warm only costs latency later
            log.warning("Prompt-cache keep-warm failed (will retry)", exc_info=True)
        await asyncio.sleep(check)


async def regime_refresh_loop(analyzer: Analyzer, store: Store, config: Config) -> None:
    """Periodically rebuild the regime brief from recently-seen news and hot-swap it
    into the analyzer, so mid-session events feed the context for SUBSEQUENT news."""
    from .analysis.regime import summarize_headlines

    cfg = config.app
    interval = cfg.regime_refresh_seconds
    if interval <= 0:
        return
    await asyncio.sleep(min(interval, 600))   # short warmup, then every interval
    while True:
        try:
            lines = await store.recent_news(cfg.regime_live_lookback_days)
            if len(lines) < cfg.regime_min_items:
                log.info("Regime refresh skipped: only %d recent items (< %d)",
                         len(lines), cfg.regime_min_items)
            else:
                brief = await summarize_headlines(config, lines, model=cfg.analyzer.model_fast,
                                                  ledger_path=cfg.token_ledger_file)
                if brief:
                    analyzer.regime_context = brief
                    import pathlib
                    pathlib.Path(cfg.regime_context_file).write_text(brief, encoding="utf-8")
                    log.info("Regime brief refreshed from %d recent items (%d chars)",
                             len(lines), len(brief))
        except Exception:  # noqa: BLE001
            log.exception("Regime refresh failed")
        await asyncio.sleep(interval)


async def refresh_catalyst_memory(analyzer: Analyzer, store: Store, config: Config) -> int:
    """Pull recently-entered catalysts from the store into the analyzer's prompt memory."""
    from .analysis.regime import format_recent_catalysts
    rows = await store.recent_entered_catalysts(config.app.catalyst_memory_days)
    analyzer.recent_catalysts = format_recent_catalysts(rows)
    return len(rows)


async def backfill_news_and_regime(config: Config, store: Store, analyzer: Analyzer) -> None:
    """Startup warm-up so live matches the backtest's context: pull the recent channel
    history the bot missed into the store (gap-aware -> frequent reboots stay fast), then
    rebuild the regime if the saved brief is missing/stale. Runs BEFORE the live Telegram
    source connects (separate client) so there's no session conflict. Catalyst memory
    already persists via the DB, so it isn't touched here."""
    from .analysis.regime import summarize_headlines
    from .backtest.engine import fetch_telegram_history

    app = config.app
    if not app.telegram_channels or app.regime_backfill_days <= 0:
        return
    latest = await store.latest_news_ms()
    gap_days = (min(app.regime_backfill_days, (time.time() - latest / 1000) / 86400 + 0.05)
                if latest else app.regime_backfill_days)
    if gap_days > 0.02:   # skip if the store is already current (<~30min gap)
        try:
            items = await fetch_telegram_history(config, app.telegram_channels, gap_days)
            for it in items:
                await store.save_news(it)
            log.info("Backfilled %d news items (%.1fd gap) into the store for regime context",
                     len(items), gap_days)
        except Exception:  # noqa: BLE001
            log.exception("News backfill failed (continuing)")

    p = pathlib.Path(app.regime_context_file)
    fresh = p.exists() and (time.time() - p.stat().st_mtime) <= app.regime_refresh_seconds
    if fresh:
        return   # saved brief is recent enough; the 12h loop will keep it current
    try:
        lines = await store.recent_news(app.regime_live_lookback_days)
        if len(lines) >= app.regime_min_items:
            brief = await summarize_headlines(config, lines, model=app.analyzer.model_fast,
                                              ledger_path=app.token_ledger_file)
            if brief:
                analyzer.regime_context = brief
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(brief, encoding="utf-8")
                log.info("Built startup regime from %d recent items (%d chars)", len(lines), len(brief))
        else:
            log.info("Startup regime not rebuilt (%d recent items < %d)", len(lines), app.regime_min_items)
    except Exception:  # noqa: BLE001
        log.exception("Startup regime build failed")


async def feed_watchdog(pipeline: Pipeline, bus: EventBus, config: Config) -> None:
    """Alert if the news feeds go silent. Tree/Telegram both auto-reconnect, but a silently
    dead feed (dropped MTProto session, a WS that reconnects yet emits nothing) otherwise just
    stops the bot trading with no signal. Fires once when silence crosses the threshold and
    once again on recovery; reads the pipeline's last-news heartbeat."""
    secs = config.app.feed_silence_alert_seconds
    if secs <= 0:
        return
    check = max(15, min(secs // 2, 300))
    alerted = False
    while True:
        await asyncio.sleep(check)
        silent = time.monotonic() - pipeline.last_news_at
        if silent >= secs and not alerted:
            mins = silent / 60
            log.warning("No news in %.0f min — feeds may be down", mins)
            await bus.publish("status", f"⚠️ No news in {mins:.0f} min — feeds (Tree/Telegram) "
                              f"may be down; no new trades will trigger until they recover.")
            alerted = True
        elif silent < secs and alerted:
            log.info("News feed recovered after silence")
            await bus.publish("status", "✅ News feed recovered — receiving items again.")
            alerted = False


async def catalyst_memory_loop(analyzer: Analyzer, store: Store, config: Config) -> None:
    interval = config.app.catalyst_memory_refresh_seconds
    if interval <= 0:
        return
    while True:
        await asyncio.sleep(interval)
        try:
            await refresh_catalyst_memory(analyzer, store, config)
        except Exception:  # noqa: BLE001
            log.exception("Catalyst-memory refresh failed")


async def supervised(name: str, factory, *, base_delay: float = 5.0,
                     max_delay: float = 300.0) -> None:
    """Run a long-lived component, restarting it with backoff if it CRASHES. A clean
    return (a disabled/unauthorized feature) is final — no restart. Without this, one
    component raising (Telegram connect error at boot, UI port conflict) tears down the
    whole gather — including the position monitor, leaving live positions with only the
    fixed exchange backstop stop."""
    delay = base_delay
    while True:
        started = time.monotonic()
        try:
            await factory()
            return
        except asyncio.CancelledError:
            raise
        # SystemExit included: uvicorn's bind failure calls sys.exit(1), which is a
        # BaseException and would otherwise escape the gather and kill the whole bot.
        except (Exception, SystemExit):  # noqa: BLE001
            if time.monotonic() - started > 60:
                delay = base_delay   # ran healthy for a while — don't punish a fresh crash
            log.exception("Task %r crashed; restarting in %.0fs", name, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, max_delay)


async def run() -> None:
    setup_logging()
    config = Config()

    missing = config.secrets.missing(require_tree=config.app.enable_tree_feed)
    if missing:
        log.error("Missing required secrets in .env: %s", ", ".join(missing))
        log.error("Copy .env.example to .env and fill it in. Aborting.")
        return

    store = await Store().init()
    bus = EventBus()

    hl = HLClient(config)
    await hl.connect()

    universe = Universe(hl, config.app.filters.allowed_dexes)
    n = await universe.refresh()
    if n == 0:
        log.error("No tradable markets loaded; check HL_NETWORK / allowed_dexes. Aborting.")
        return
    log.info("Equities(xyz): %d | Crypto: %d",
             len(universe.equity_symbols()), len(universe.crypto_symbols()))

    analyzer = Analyzer(config, ledger_path=config.app.token_ledger_file)
    n_cat = await refresh_catalyst_memory(analyzer, store, config)  # persists across reboots (DB)
    log.info("Loaded %d recent traded catalysts into analyzer memory (last %gd)",
             n_cat, config.app.catalyst_memory_days)
    # Warm the regime from recent history (gap-aware) before the live source connects.
    await backfill_news_and_regime(config, store, analyzer)
    executor = Executor(config, hl, store, bus)
    pm = PositionManager(config, hl, executor)
    await pm.restore(store)
    # confirmer wired here (not imported by trading/) so the skeptic entry-confirmation
    # can be flipped on via analyzer.confirm_entries without any further code change.
    risk = RiskEngine(config, hl, store, pm, universe=universe, confirmer=analyzer.confirm)
    await risk.restore()   # rebuild per-ticker cooldowns from the DB
    dedup = Dedup(config.app.filters)
    try:
        n_seen = dedup.restore(await store.recent_news_ids(dedup.memory))
        log.info("Restored dedup memory: %d seen news ids", n_seen)
    except Exception:  # noqa: BLE001
        log.warning("Could not restore dedup memory")

    pipeline = Pipeline(bus=bus, store=store, dedup=dedup, analyzer=analyzer,
                        universe=universe, risk=risk, executor=executor, position_manager=pm)

    telegram = TelegramNotifier(config, bus, pm=pm)
    tree = TreeClient(config, pipeline.on_news)
    tg_source = TelegramSource(config, pipeline.on_news)

    capital = CapitalTracker(hl)
    persisted = capital.starting is not None   # loaded a baseline from a prior run?
    try:
        await capital.refresh()   # set current equity; capture the baseline if none persisted
        if capital.starting is not None:
            log.info("Capital baseline $%s (%s) | current equity $%s",
                     f"{capital.starting:,.2f}", "persisted" if persisted else "new",
                     f"{capital.current:,.2f}" if capital.current is not None else "?")
    except Exception:  # noqa: BLE001
        log.warning("Could not read account equity")

    app = create_app(bus, store, config, capital, hl=hl, tg_source=tg_source, tree=tree, pm=pm)
    userver = uvicorn.Server(uvicorn.Config(
        app, host=config.app.ui.host, port=config.app.ui.port, log_level="warning"))

    mode = "DRY-RUN" if config.runtime.dry_run else "LIVE"
    log.info("hlbot starting in %s mode. Dashboard: http://%s:%d",
             mode, config.app.ui.host, config.app.ui.port)
    await bus.publish("status", f"hlbot started in {mode} mode")

    def _task(name: str, factory) -> asyncio.Task:
        return asyncio.create_task(supervised(name, factory), name=name)

    tasks = [
        _task("telegram_src", tg_source.run),
        _task("mids", lambda: hl.mid_cache_loop(config.app.filters.allowed_dexes)),
        _task("positions", pm.run),
        _task("telegram", telegram.run),
        _task("telegram_cmds", telegram.poll_commands),
        _task("refresh", lambda: periodic_refresh(universe, config.app.universe_refresh_seconds)),
        _task("cache_warm", lambda: cache_keepwarm_loop(analyzer, universe, config)),
        _task("regime", lambda: regime_refresh_loop(analyzer, store, config)),
        _task("catalyst_mem", lambda: catalyst_memory_loop(analyzer, store, config)),
        _task("watchdog", lambda: feed_watchdog(pipeline, bus, config)),
        _task("capital", lambda: capital_loop(capital, bus)),
        _task("ui", userver.serve),
    ]
    if config.app.enable_tree_feed:
        tasks.append(_task("tree", tree.run))
    else:
        log.info("Tree feed disabled (enable_tree_feed=false) — Telegram channels only")
    try:
        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutting down…")
    finally:
        tree.stop()
        pm.stop()
        telegram.stop()
        await tg_source.stop()   # async: cleanly disconnects the Telethon client
        for t in tasks:
            t.cancel()
        # Let the cancellations land before draining/closing the store, so no task is
        # still mid-write (or mid-restart-backoff) when the DB goes away.
        await asyncio.gather(*tasks, return_exceptions=True)
        await pipeline.drain()   # flush in-flight processing + audit writes before closing the DB
        await store.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
