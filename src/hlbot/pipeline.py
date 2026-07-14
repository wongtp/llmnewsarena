"""The news -> trade pipeline.

Ingestion (`on_news`) is a fast, non-blocking enqueue: feed callbacks (Tree WS loop,
Telegram handler/poll) await it inline, so it must never stall behind a multi-second
Claude call — during a breaking-news burst the Nth headline would otherwise wait
N x analysis-latency and could age past the staleness gate unanalyzed. Each kept item
is processed in its own task: analysis runs concurrently (bounded by a semaphore);
the risk->execute section stays strictly serialized by `_trade_lock` so concurrent
items can't both pass the portfolio caps. Each stage publishes to the bus for the
UI / Telegram / audit.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .analysis.analyzer import Analyzer
from .analysis.burst import BurstBuffer, build_burst_item
from .analysis.universe import Universe
from .bus import EventBus
from .models import Analysis, NewsItem
from .news.dedup import Dedup
from .trading.executor import Executor
from .trading.position_manager import PositionManager
from .trading.risk import RiskEngine

log = logging.getLogger("hlbot.pipeline")

# Hard ceiling on in-flight processing tasks. Reaching it means the analyzer is far
# behind (or wedged); items deep in such a backlog would be stale before analysis
# anyway, so drop loudly rather than grow without bound.
MAX_PENDING = 200


def _log_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        log.error("pipeline task %s failed: %r", task.get_name(), exc)


class Pipeline:
    def __init__(self, *, bus: EventBus, store, dedup: Dedup, analyzer: Analyzer,
                 universe: Universe, risk: RiskEngine, executor: Executor,
                 position_manager: PositionManager, analysis_concurrency: int = 4):
        self.bus = bus
        self.store = store
        self.dedup = dedup
        self.analyzer = analyzer
        self.universe = universe
        self.risk = risk
        self.executor = executor
        self.pm = position_manager
        # Monotonic time of the last news item received (any feed) — the feed-silence watchdog
        # reads this to detect a silently-dead feed. Seeded at startup so silence is measured
        # from launch, not from epoch.
        self.last_news_at = time.monotonic()
        self._bg_tasks: set = set()
        # Serializes the risk->execute->register section so concurrently-processed news
        # can't both pass the caps (max positions / exposure / daily-loss / cooldown)
        # before either registers its position.
        self._trade_lock = asyncio.Lock()
        # Bounds concurrent Claude calls during a headline burst.
        self._analysis_sem = asyncio.Semaphore(analysis_concurrency)
        self._proc_tasks: set = set()   # per-item processing tasks (analyze -> trade)
        self._critical: set = set()     # in-flight trade sections (shielded; see _process)
        # Burst aggregation (default off): combine same-ticker wire pieces (EPS line,
        # then revenue, then guidance) into one holistic re-analysis. See analysis/burst.py.
        bw = getattr(getattr(analyzer, "cfg", None), "burst_window_seconds", 0.0)
        self._burst = BurstBuffer(bw or 0.0)

    def _persist(self, coro) -> None:
        """Fire-and-forget a DB write so it stays off the news->order critical path."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        task.add_done_callback(
            lambda t: t.cancelled() or (t.exception() and log.error("persist error: %s", t.exception())))

    async def drain(self) -> None:
        """Wait for all in-flight work: per-item processing, then shielded trade sections,
        then fire-and-forget persistence writes. Ordered so each later group catches tasks
        spawned by the earlier one. Call on shutdown BEFORE closing the store."""
        for group in (self._proc_tasks, self._critical, self._bg_tasks):
            pending = list(group)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def on_news(self, item: NewsItem) -> None:
        """Fast ingest: heartbeat, persist raw, dedup, then hand the item to a background
        processing task. Must stay cheap — the feed loop that awaits this is blocked from
        reading its next message until it returns."""
        self.last_news_at = time.monotonic()   # heartbeat for the feed-silence watchdog
        self._persist(self.store.save_news(item))
        await self.bus.publish("news.raw", item)

        keep, reason = self.dedup.check(item)
        if not keep:
            await self.bus.publish("news.skipped", {"news": item.to_dict(), "reason": reason})
            log.debug("skip %s: %s", item.id, reason)
            return

        if len(self._proc_tasks) >= MAX_PENDING:
            log.error("Pipeline backlog (%d in flight) — dropping %s", len(self._proc_tasks), item.id)
            await self.bus.publish("news.skipped", {"news": item.to_dict(),
                                                    "reason": "pipeline backlog"})
            return
        task = asyncio.create_task(self._process(item), name=f"news:{item.id}")
        self._proc_tasks.add(task)
        task.add_done_callback(self._proc_tasks.discard)
        task.add_done_callback(_log_task_error)

    async def _process(self, item: NewsItem) -> None:
        t0 = time.perf_counter()
        async with self._analysis_sem:
            queued = time.perf_counter() - t0   # burst pressure: time spent waiting for a slot
            analysis = await self.analyzer.analyze(item, self.universe)
        took = time.perf_counter() - t0
        log.info("analysis %s -> %s %s conf=%.2f in %.2fs%s (%s)",
                 item.id, analysis.ticker or "-", analysis.direction, analysis.confidence,
                 took, f" [queued {queued:.2f}s]" if queued > 0.05 else "", analysis.model)
        self._persist(self.store.save_analysis(analysis))
        await self.bus.publish("analysis", analysis)

        # Burst aggregation: if earlier pieces of this story are in the window, re-analyze
        # the combined burst and act on THAT verdict (piece 1 always acts solo, instantly).
        if self._burst.enabled and analysis.ticker:
            prior = self._burst.prior(analysis.ticker, item.time_ms)
            self._burst.add(analysis.ticker, item.time_ms, item.body.strip() or item.title)
            if prior:
                combined_item = build_burst_item(item, prior)
                async with self._analysis_sem:
                    combined = await self.analyzer.analyze(combined_item, self.universe)
                log.info("burst %s (%d pieces) -> %s %s conf=%.2f (solo was %s %.2f)",
                         analysis.ticker, len(prior) + 1, combined.ticker or "-",
                         combined.direction, combined.confidence,
                         analysis.direction, analysis.confidence)
                self._persist(self.store.save_analysis(combined))
                await self.bus.publish("analysis", combined)
                analysis = combined

        # The trade section runs as a SHIELDED task: if THIS task is cancelled (shutdown,
        # or an upstream feed timeout) while an order is in flight, severing the await
        # could leave a filled exchange position that was never tracked — the order thread
        # completes regardless. The shield lets order->track->persist finish; drain()
        # awaits any stragglers.
        trade = asyncio.create_task(self._trade(item, analysis), name=f"trade:{item.id}")
        self._critical.add(trade)
        trade.add_done_callback(self._critical.discard)
        trade.add_done_callback(_log_task_error)
        try:
            await asyncio.shield(trade)
        except Exception:  # noqa: BLE001 - already logged by the task's done callback
            pass

    async def _trade(self, item: NewsItem, analysis: Analysis) -> None:
        async with self._trade_lock:
            market = self.universe.resolve(analysis.ticker, analysis.asset_class)

            # Contrary-news safeguard: bearish news on a long we hold (or bullish on a short)
            # closes the position immediately — even below the entry gate, since exiting a
            # turning position is lower-risk than entering one.
            if market and self.pm.has_open(market.name):
                pos = self.pm.position_for(market.name)
                floor = self.risk.r.contrary_exit_min_confidence
                if (pos and not analysis.is_stale and analysis.direction in ("long", "short")
                        and analysis.direction != pos.side and floor > 0
                        and analysis.confidence >= floor):
                    log.info("CONTRARY-EXIT %s (held %s) on %s news conf=%.2f: %s",
                             pos.symbol, pos.side, analysis.direction, analysis.confidence,
                             (analysis.rationale or "")[:80])
                    await self.pm.force_close(pos, "contrary news exit")
                    await self.bus.publish("status", f"⚠️ Closed {pos.symbol} {pos.side} on "
                                           f"contrary news ({analysis.direction} "
                                           f"{analysis.confidence:.0%})")
                    return

            decision = await self.risk.evaluate(item, analysis, market)
            self._persist(self.store.save_decision(decision))
            await self.bus.publish("decision", decision)

            if decision.action != "enter":
                log.info("reject %s (%s): %s", item.id, analysis.ticker, decision.reason)
                return

            log.info("ENTER %s %s conf=%.2f $%.0f", decision.side, market.symbol,
                     decision.confidence, decision.notional_usd)
            pos = await self.executor.open(decision, item, analysis)
            if pos:
                self.pm.track(pos)
                self.risk.note_entry(market.symbol)
