"""Restart-persistence + safe-restore regression tests (H2, H3, M1, L3)."""
import asyncio
import os
import tempfile
import time

from hlbot.config import Config, RuntimeState
from hlbot.models import Position
from hlbot.store.db import Store
from hlbot.trading.position_manager import PositionManager
from hlbot.trading.risk import RiskEngine

# RUNTIME_STATE_FILE / LISTING_SEEN_FILE are isolated per-test by conftest.py.


def _tmp(name):
    return os.path.join(tempfile.mkdtemp(), name)


def _pos(**kw) -> Position:
    base = dict(id="p1", news_id="n1", market="xyz:MRVL", symbol="MRVL", dex="xyz", side="long",
                size=10.0, entry_px=100.0, stop_loss=97.0, take_profit=0.0, leverage=5,
                notional_usd=1000.0, opened_ms=int(time.time() * 1000),
                time_exit_ms=int(time.time() * 1000) + 10**9, dry_run=True)
    base.update(kw)
    return Position(**base)


# ---- H3: runtime flags (kill switch + mode) persist across restart ----------
def test_runtime_state_persists():
    rs = RuntimeState(dry_run=True)
    rs.halt("daily loss limit")
    rs.set_dry_run(False)
    # A fresh instance (== a restart) must reload the persisted flags, NOT the config default.
    rs2 = RuntimeState(dry_run=True)
    assert rs2.dry_run is False
    assert rs2.trading_halted is True and rs2.halt_reason == "daily loss limit"
    rs2.resume()
    assert RuntimeState(dry_run=True).trading_halted is False


# ---- H2: restore tolerates schema drift, never silently drops a position -----
class _FakeStore:
    def __init__(self, rows):
        self._rows = rows

    async def open_positions(self, model_id=None):
        return self._rows


def test_restore_ignores_unknown_fields_and_drops_only_unbuildable():
    good = _pos(id="good").to_dict()
    extra = _pos(id="extra").to_dict()
    extra["bogus_legacy_field"] = 123          # unknown field -> ignored, not fatal
    broken = {"id": "broken", "side": "long"}  # missing required fields -> dropped, no crash
    pm = PositionManager(config=None, hl=None, executor=None)
    asyncio.run(pm.restore(_FakeStore([good, extra, broken])))
    assert set(pm._open) == {"good", "extra"}      # both real positions restored
    assert pm._open["extra"].symbol == "MRVL"      # extra field didn't corrupt it
    assert "broken" not in pm._open                # unbuildable dropped (and logged)


# ---- M1 + L3: cooldown rebuilt from DB; bad rows skipped ---------------------
def test_cooldown_restore_and_bad_row_skip():
    async def run():
        st = Store(_tmp("t.sqlite"))
        await st.init()
        now = int(time.time() * 1000)
        await st.upsert_position(_pos(id="a", symbol="MRVL", opened_ms=now - 60_000))
        await st.upsert_position(_pos(id="b", symbol="BTC", market="BTC", dex="", opened_ms=now - 120_000))
        await st.upsert_position(_pos(id="c", symbol="ETH", market="ETH", dex="", opened_ms=now - 10_000_000))
        # L3: an unparseable row must not abort open_positions()
        await st._db.execute("INSERT INTO positions (id,status,json) VALUES ('bad','open','{not json')")
        await st._db.commit()
        risk = RiskEngine(Config(), None, st, None)
        await risk.restore()
        opens = await st.open_positions()
        await st.close()
        return risk._last_entry_ms, len(opens)

    cooldowns, n_open = asyncio.run(run())
    assert "MRVL" in cooldowns and "BTC" in cooldowns   # within the 15-min window
    assert "ETH" not in cooldowns                        # older than cooldown -> excluded
    assert n_open == 3    # a,b,c valid; the unparseable 'bad' row skipped without crashing


def test_latest_news_ms_for_backfill_gap():
    async def run():
        st = Store(_tmp("nb.sqlite"))
        await st.init()
        assert await st.latest_news_ms() is None      # empty -> full backfill on first boot
        from hlbot.models import NewsItem
        await st.save_news(NewsItem(id="a", title="t", body="b", source="s", link=None,
                                    time_ms=1_700_000_000_000, received_ms=0))
        await st.save_news(NewsItem(id="b", title="t2", body="b2", source="s", link=None,
                                    time_ms=1_700_000_500_000, received_ms=0))
        latest = await st.latest_news_ms()
        await st.close()
        return latest
    assert asyncio.run(run()) == 1_700_000_500_000     # max ts -> warm restart fetches only the gap


def test_duplicate_signal_window_survives_restart():
    # The 7200s duplicate window is validated to close the rebroadcast-after-stop-out hole;
    # a restart inside the window must rebuild it from the analyses/news tables.
    from hlbot.models import Analysis, NewsItem

    async def run():
        st = Store(_tmp("dup.sqlite"))
        await st.init()
        now = int(time.time() * 1000)
        await st.save_news(NewsItem(id="n1", title="MRVL beats", body="huge guide up",
                                    source="s", link=None, time_ms=now - 60_000,
                                    received_ms=now - 60_000))
        await st.save_analysis(Analysis("n1", "MRVL", "equity", "long", 0.9, "days",
                                        False, "r", "m"))
        # a none-direction analysis must NOT enter the window
        await st.save_news(NewsItem(id="n2", title="BTC chatter", body="meh", source="s",
                                    link=None, time_ms=now - 60_000, received_ms=now - 60_000))
        await st.save_analysis(Analysis("n2", "BTC", "crypto", "none", 0.1, "none",
                                        False, "r", "m"))
        risk = RiskEngine(Config(), None, st, None)
        await risk.restore()
        await st.close()
        return dict(risk._recent_signal)

    sig = asyncio.run(run())
    assert ("MRVL", "long") in sig and len(sig) == 1
    ts, fp = sig[("MRVL", "long")]
    assert fp                                        # fingerprint rebuilt from the news text
