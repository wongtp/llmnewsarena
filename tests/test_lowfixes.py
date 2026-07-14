"""L2 (token ledger) + L4 (dry-run slippage) regression tests."""
import asyncio
import json
import os
import tempfile
import types

from hlbot.analysis import usage_ledger
from hlbot.config import Config
from hlbot.models import Analysis, Decision, Market, NewsItem
from hlbot.trading.executor import Executor


def _resp(i, o, cr=0, cc=0):
    u = types.SimpleNamespace(input_tokens=i, output_tokens=o,
                             cache_read_input_tokens=cr, cache_creation_input_tokens=cc)
    return types.SimpleNamespace(usage=u)


def test_usage_ledger_accumulates_and_is_safe():
    path = os.path.join(tempfile.mkdtemp(), "led.json")
    usage_ledger.record(path, "claude-sonnet-4-6", _resp(100, 20, 5, 0))
    usage_ledger.record(path, "claude-sonnet-4-6", _resp(50, 10, 0, 7))
    usage_ledger.record(path, "claude-haiku-4-5-20251001", _resp(10, 2))
    data = json.loads(open(path, encoding="utf-8").read())
    s = data["claude-sonnet-4-6"]
    assert s["calls"] == 2 and s["input"] == 150 and s["output"] == 30
    assert s["cache_read"] == 5 and s["cache_creation"] == 7
    assert data["claude-haiku-4-5-20251001"]["calls"] == 1
    # No path / no usage must be silent no-ops (never raise into the caller).
    usage_ledger.record(None, "x", _resp(1, 1))
    usage_ledger.record(path, "x", types.SimpleNamespace(usage=None))


class _FakeStore:
    async def upsert_position(self, p):  # noqa: ANN001
        pass


class _FakeBus:
    async def publish(self, *a, **k):
        pass


def _decision(side):
    mkt = Market(name="xyz:MRVL", symbol="MRVL", dex="xyz", asset_class="equity",
                 sz_decimals=2, max_leverage=5)
    return Decision(news_id="n", action="enter", reason="r", market=mkt, side=side,
                    notional_usd=1000.0, size=10.0, leverage=5, entry_px=100.0,
                    stop_loss=97.0, take_profit=0.0, trail_pct=0.08, time_exit_seconds=3600,
                    confidence=0.9)


def _item():
    return NewsItem(id="n", title="t", body="b", source="s", link=None, time_ms=0, received_ms=0)


def _analysis():
    return Analysis(news_id="n", ticker="MRVL", asset_class="equity", direction="long",
                    confidence=0.9, time_sensitivity="days", is_stale=False, rationale="r", model="m")


def test_dry_run_slippage_applied_adversely():
    cfg = Config()
    cfg.runtime.dry_run = True
    cfg.app.risk.dry_run_slippage_pct = 0.001   # 10 bps, obvious
    ex = Executor(cfg, None, _FakeStore(), _FakeBus())
    plong = asyncio.run(ex.open(_decision("long"), _item(), _analysis()))
    assert abs(plong.entry_px - 100.10) < 1e-6   # long pays UP
    assert abs(plong.peak_px - plong.entry_px) < 1e-9
    # SL re-anchored to the slipped fill (keeps the configured % distance).
    assert abs(plong.stop_loss - 97.0 * 1.001) < 1e-6
    pshort = asyncio.run(ex.open(_decision("short"), _item(), _analysis()))
    assert abs(pshort.entry_px - 99.90) < 1e-6   # short fills DOWN


def test_tree_key_required_only_when_tree_feed_enabled(monkeypatch):
    for k in ("HL_ACCOUNT_ADDRESS", "HL_SECRET_KEY", "ANTHROPIC_API_KEY", "TREE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from hlbot.config import Secrets
    s = Secrets(hl_account_address="0xabc", hl_secret_key="0xdef",
                anthropic_api_key="sk-test", tree_api_key="")
    assert s.missing() == ["TREE_API_KEY"]          # default: Tree feed assumed enabled
    assert s.missing(require_tree=False) == []      # Telegram-only setup needs no Tree key
