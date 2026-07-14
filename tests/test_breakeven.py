"""Live-side breakeven stop: PositionManager._exit_reason/_eff_stop must mirror the
backtest's walk_candles floor semantics (initial / trailing / breakeven, binding wins)."""
import types

from hlbot.config import Config
from hlbot.models import Position, now_ms
from hlbot.trading.position_manager import PositionManager


def _pos(side="long", entry=100.0, stop=95.0, trail=0.0):
    return Position(id="p1", news_id="n", market="xyz:MRVL", symbol="MRVL", dex="xyz",
                    side=side, size=1.0, entry_px=entry, stop_loss=stop, take_profit=0.0,
                    leverage=5, notional_usd=100.0, opened_ms=now_ms(),
                    time_exit_ms=now_ms() + 3_600_000, dry_run=True, trail_pct=trail)


def _pm(arm=0.02, offset=0.0):
    cfg = Config()
    cfg.app.risk.breakeven_arm_pct = arm
    cfg.app.risk.breakeven_offset_pct = offset
    return PositionManager(cfg, types.SimpleNamespace(), types.SimpleNamespace())


def test_long_breakeven_arms_then_exits_at_entry():
    pm, pos = _pm(), _pos()
    assert pm._exit_reason(pos, 102.5) is None          # +2.5% arms (peak advanced)
    assert pm._eff_stop(pos) == 100.0                   # floor moved to entry
    assert pm._exit_reason(pos, 99.9) == "breakeven stop"


def test_long_not_armed_below_threshold():
    pm, pos = _pm(), _pos()
    assert pm._exit_reason(pos, 101.5) is None          # +1.5% < 2% arm
    assert pm._eff_stop(pos) == 95.0                    # still the initial stop
    assert pm._exit_reason(pos, 96.0) is None           # above stop -> no exit
    assert pm._exit_reason(pos, 94.9) == "stop loss"


def test_short_breakeven_with_offset_locks_profit():
    pm, pos = _pm(offset=0.002), _pos(side="short", entry=100.0, stop=105.0)
    assert pm._exit_reason(pos, 97.5) is None           # -2.5% favorable arms
    assert round(pm._eff_stop(pos), 2) == 99.8          # entry*(1-0.002)
    assert pm._exit_reason(pos, 99.85) == "breakeven stop"


def test_trailing_floor_wins_after_big_run():
    pm, pos = _pm(), _pos(trail=0.03)
    assert pm._exit_reason(pos, 110.0) is None          # peak 110; trail floor 106.7 > entry
    assert round(pm._eff_stop(pos), 2) == 106.7
    assert pm._exit_reason(pos, 106.0) == "trailing stop"


def test_feature_off_is_unchanged():
    pm, pos = _pm(arm=0.0), _pos()
    assert pm._exit_reason(pos, 102.5) is None
    assert pm._eff_stop(pos) == 95.0                    # no breakeven floor
    assert pm._exit_reason(pos, 99.0) is None           # falls through entry, no exit
    assert pm._exit_reason(pos, 94.0) == "stop loss"
