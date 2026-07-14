"""HLClient.account_value: unified equity = spot USDC + perp accountValue across dexes."""
import asyncio

from hlbot.config import Config
from hlbot.trading.hl_client import HLClient


class FakeInfo:
    def __init__(self, fail_spot=False):
        self.fail_spot = fail_spot
        self.perp = {"": "500.0", "xyz": "250.0"}

    def spot_user_state(self, addr):
        if self.fail_spot:
            raise RuntimeError("spot endpoint down")
        return {"balances": [{"coin": "USDC", "total": "13305.33"},
                             {"coin": "USDE", "total": "0.0"}]}

    def user_state(self, addr, dex=""):
        return {"marginSummary": {"accountValue": self.perp.get(dex, "0")}}


def _hl(info):
    hl = HLClient(Config())
    hl.info = info
    hl.address = "0xabc"
    hl._dexes = ["", "xyz"]
    return hl


def test_account_value_unified_sum():
    v = asyncio.run(_hl(FakeInfo()).account_value())
    assert abs(v - (13305.33 + 500.0 + 250.0)) < 1e-6   # spot USDC + both perp dexes


def test_account_value_survives_spot_failure():
    # spot read fails but perp reads succeed -> still returns the perp sum, not None
    v = asyncio.run(_hl(FakeInfo(fail_spot=True)).account_value())
    assert abs(v - (500.0 + 250.0)) < 1e-6


def test_account_value_none_when_everything_fails():
    class DeadInfo:
        def spot_user_state(self, a):
            raise RuntimeError("x")

        def user_state(self, a, d=""):
            raise RuntimeError("x")

    assert asyncio.run(_hl(DeadInfo()).account_value()) is None


# ---- CapitalTracker: baseline persists across restarts ----------------------
class _StubHL:
    def __init__(self, equity):
        self.equity = equity

    async def account_value(self):
        return self.equity


def test_capital_baseline_persists_across_restart(tmp_path):
    from hlbot.main import CapitalTracker
    path = str(tmp_path / "cap.json")

    t1 = CapitalTracker(_StubHL(1000.0), path=path)
    asyncio.run(t1.refresh())
    assert t1.starting == 1000.0 and t1.current == 1000.0 and t1.started_at

    # "restart": fresh tracker, same file, equity has since grown. Baseline must NOT move.
    t2 = CapitalTracker(_StubHL(1500.0), path=path)
    assert t2.starting == 1000.0          # loaded from disk before any refresh
    asyncio.run(t2.refresh())
    assert t2.starting == 1000.0 and t2.current == 1500.0

    # reset() re-baselines to current and persists it...
    t2.reset()
    assert t2.starting == 1500.0
    # ...so a subsequent restart sees the reset baseline.
    t3 = CapitalTracker(_StubHL(1500.0), path=path)
    assert t3.starting == 1500.0
