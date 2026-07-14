"""Shared test isolation: give every test fresh, empty persisted-state files so runtime
flags, listing-seen data, backtest caches (candles/funding), the capital baseline and the
arena DB never leak between tests or touch the real data/ directory. New persisted-state
files MUST be added here (CLAUDE.md rule)."""
import pytest

from hlbot import config as config_mod
from hlbot import main as main_mod
from hlbot import main_arena as main_arena_mod
from hlbot.analysis import universe as universe_mod
from hlbot.backtest import engine as bt_engine


@pytest.fixture(autouse=True)
def _isolate_persisted_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "RUNTIME_STATE_FILE", str(tmp_path / "runtime_state.json"))
    monkeypatch.setattr(universe_mod, "LISTING_SEEN_FILE", str(tmp_path / "listing_seen.json"))
    # Backtest caches: no test may read the real (multi-MB) funding cache or write data/bt_candles.
    monkeypatch.setattr(bt_engine, "CANDLE_CACHE_DIR", tmp_path / "bt_candles")
    monkeypatch.setattr(bt_engine, "_FUNDING",
                        bt_engine.FundingHistory(str(tmp_path / "bt_funding_cache.json")))
    monkeypatch.setattr(main_mod, "CAPITAL_BASELINE_FILE",
                        str(tmp_path / "capital_baseline.json"))
    monkeypatch.setattr(main_arena_mod, "ARENA_DB", str(tmp_path / "arena.sqlite"))
    monkeypatch.setattr(main_arena_mod, "ARENA_RUNTIME_STATE",
                        str(tmp_path / "arena_runtime_state.json"))
