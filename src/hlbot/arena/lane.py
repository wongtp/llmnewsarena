"""A single arena competitor — one fully isolated trading lane.

Each lane has its own RiskEngine (the ONLY per-lane parameter is its confidence gate),
Executor, PositionManager and capital baseline, over its own wallet (a shared read-only
HLClient in paper; a per-wallet client at go-live). All lanes share the arena's news feed,
Analyzer, universe and regime — so the only differences between competitors are the model and
its gate. Mirrors the production wiring in main.py, just instantiated per model.
"""
from __future__ import annotations

import asyncio

from ..bus import EventBus
from ..config import Config
from ..trading.executor import Executor
from ..trading.hl_client import HLClient
from ..trading.position_manager import PositionManager
from ..trading.risk import RiskEngine


class TradingLane:
    def __init__(self, *, key: str, model: str, gate: float, capital_usd: float,
                 config: Config, hl: HLClient, store, bus: EventBus, universe, live: bool = False):
        self.key = key            # short id (sonnet/gpt/...) for display/logging
        self.model = model        # routing id passed to analyzer.analyze() AND the model_id partition
        #                           (executor stamps positions with analysis.model == this)
        self.gate = gate
        self.capital_usd = capital_usd
        self.live = live          # real money this lane? (paper otherwise)
        self.hl = hl
        self.store = store
        self.executor = Executor(config, hl, store, bus)
        self.pm = PositionManager(config, hl, self.executor)
        self.risk = RiskEngine(config, hl, store, self.pm, universe=universe)
        # The one varied parameter: clone the shared RiskConfig and override only the gate, so
        # every other guardrail (sizing tiers, caps, cooldowns, exits, haircuts) is identical.
        self.risk.r = config.app.risk.model_copy(update={"confidence_threshold": gate})
        # Per-lane paper/live + daily-loss accounting (its own model_id). Paper lanes force
        # dry-run regardless of the global flag so a live lane never drags others live.
        self.executor.dry_run_override = not live
        self.risk.dry_run_override = not live
        self.risk.model_id = model
        # Serializes THIS lane's risk->execute section so concurrent news can't both pass its
        # caps. Lanes are independent (separate wallets/caps), so each has its own lock.
        self.trade_lock = asyncio.Lock()

    async def restore(self) -> None:
        """Restore this lane's open positions (filtered to its model_id) so a restart resumes
        managing them; paper starts fresh."""
        await self.pm.restore(self.store, model_id=self.model)
        await self.risk.restore()   # cooldowns, filtered to this lane via risk.model_id

    def open_unrealized(self) -> float:
        # Mode must match how this lane's positions were opened, or a LIVE lane's uPnL
        # reads 0 on the leaderboard exactly when real money is on.
        return self.pm.unrealized_pnl(dry_run=not self.live)
