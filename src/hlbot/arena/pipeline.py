"""The arena news -> 5-lane trade pipeline.

Ingestion (`on_news`) stays a cheap non-blocking enqueue (the feed loop awaits it inline),
exactly like production. Each kept item is processed in its own task; inside, the item is
analyzed ONCE PER MODEL (shared Analyzer, bounded by a semaphore) and each lane trades on its
own verdict. Per-lane risk->execute is serialized by that lane's own lock; the trade section is
shielded so a cancel mid-order can't orphan an exchange position. Lanes are independent, so one
lane's slow analysis or failed order never blocks another.

Shared (identical inputs => fair competition): feed, dedup, universe, regime, Analyzer.
Per-lane (the competition): the model + its gate + an isolated wallet/risk/executor/PM.
"""
from __future__ import annotations

import asyncio
import logging
import time

log = logging.getLogger("hlbot.arena.pipeline")

MAX_PENDING = 200


def _log_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        log.error("arena task %s failed: %r", task.get_name(), exc)


class ArenaPipeline:
    def __init__(self, *, bus, store, dedup, analyzer, universe, lanes,
                 analysis_concurrency: int = 6):
        self.bus = bus
        self.store = store
        self.dedup = dedup
        self.analyzer = analyzer
        self.universe = universe
        self.lanes = lanes
        self.last_news_at = time.monotonic()
        self._bg_tasks: set = set()
        self._proc_tasks: set = set()
        self._critical: set = set()
        # Bounds TOTAL concurrent Claude/provider calls across all lanes during a burst.
        self._analysis_sem = asyncio.Semaphore(analysis_concurrency)

    def _persist(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        task.add_done_callback(
            lambda t: t.cancelled() or (t.exception() and log.error("persist error: %s", t.exception())))

    async def drain(self) -> None:
        for group in (self._proc_tasks, self._critical, self._bg_tasks):
            pending = list(group)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def on_news(self, item) -> None:
        self.last_news_at = time.monotonic()
        self._persist(self.store.save_news(item))
        await self.bus.publish("news.raw", item)

        keep, reason = self.dedup.check(item)
        if not keep:
            await self.bus.publish("news.skipped", {"news": item.to_dict(), "reason": reason})
            return
        if len(self._proc_tasks) >= MAX_PENDING:
            log.error("Arena backlog (%d in flight) — dropping %s", len(self._proc_tasks), item.id)
            await self.bus.publish("news.skipped", {"news": item.to_dict(), "reason": "backlog"})
            return
        task = asyncio.create_task(self._process(item), name=f"news:{item.id}")
        self._proc_tasks.add(task)
        task.add_done_callback(self._proc_tasks.discard)
        task.add_done_callback(_log_task_error)

    async def _process(self, item) -> None:
        # Fan out: every lane analyzes (its own model) and trades, concurrently + independently.
        results = await asyncio.gather(*(self._lane_process(lane, item) for lane in self.lanes),
                                       return_exceptions=True)
        for lane, res in zip(self.lanes, results):
            if isinstance(res, BaseException):   # gather absorbed it; don't lose the traceback
                log.error("[%s] lane processing failed for %s", lane.key, item.id,
                          exc_info=res)

    async def _lane_process(self, lane, item) -> None:
        t0 = time.perf_counter()
        async with self._analysis_sem:
            analysis = await self.analyzer.analyze(item, self.universe, model=lane.model)
        log.info("[%s] %s -> %s %s conf=%.2f in %.2fs", lane.key, item.id,
                 analysis.ticker or "-", analysis.direction, analysis.confidence,
                 time.perf_counter() - t0)
        self._persist(self.store.save_analysis(analysis))
        await self.bus.publish("analysis", analysis)   # analysis.model identifies the lane

        # Shielded trade section (per-lane): an in-flight order must finish even if this task is
        # cancelled on shutdown, or it could leave an untracked exchange position. drain() waits.
        trade = asyncio.create_task(self._lane_trade(lane, item, analysis),
                                    name=f"trade:{lane.key}:{item.id}")
        self._critical.add(trade)
        trade.add_done_callback(self._critical.discard)
        trade.add_done_callback(_log_task_error)
        try:
            await asyncio.shield(trade)
        except Exception:  # noqa: BLE001 - logged by the done callback
            pass

    async def _lane_trade(self, lane, item, analysis) -> None:
        async with lane.trade_lock:
            market = self.universe.resolve(analysis.ticker, analysis.asset_class)

            # Contrary-news safeguard (mirror production): bearish news on a long we hold (or
            # bullish on a short) closes the position — even below the gate, on RAW confidence.
            if market and lane.pm.has_open(market.name):
                pos = lane.pm.position_for(market.name)
                floor = lane.risk.r.contrary_exit_min_confidence
                if (pos and not analysis.is_stale and analysis.direction in ("long", "short")
                        and analysis.direction != pos.side and floor > 0
                        and analysis.confidence >= floor):
                    log.info("[%s] CONTRARY-EXIT %s (held %s) on %s conf=%.2f", lane.key,
                             pos.symbol, pos.side, analysis.direction, analysis.confidence)
                    await lane.pm.force_close(pos, "contrary news exit")
                    return

            decision = await lane.risk.evaluate(item, analysis, market)
            decision.model = analysis.model   # tag for the per-lane UI drill-in
            self._persist(self.store.save_decision(decision))
            await self.bus.publish("decision", decision)
            if decision.action != "enter":
                return
            log.info("[%s] ENTER %s %s conf=%.2f $%.0f", lane.key, decision.side,
                     market.symbol, decision.confidence, decision.notional_usd)
            pos = await lane.executor.open(decision, item, analysis)
            if pos:
                lane.pm.track(pos)
                lane.risk.note_entry(market.symbol)
