"""Telegram /start position browser: card rendering, snapshot math, update routing,
chat restriction, and the two-step manual close."""
import asyncio

from hlbot.bus import EventBus
from hlbot.config import Config
from hlbot.models import Position, now_ms
from hlbot.notify.formatting import format_position_card
from hlbot.notify.telegram import TelegramNotifier
from hlbot.trading.position_manager import PositionManager

CHAT = "777"


class FakeHL:
    def __init__(self, mids=None):
        self.mids = mids or {}

    async def mid(self, market):
        return self.mids.get(market.name)


class FakeExecutor:
    """Records close calls; fills at the live mid like the paper path."""

    def __init__(self, fill_px=105.0, fail=False):
        self.fill_px = fill_px
        self.fail = fail
        self.closed = []

    async def close(self, pos, reason, exit_px=None):
        if self.fail:
            return None   # live close did not fill -> position stays open
        pos.exit_px = exit_px or self.fill_px
        pos.pnl_usd = 12.34
        pos.exit_reason = reason
        pos.status = "closed"
        self.closed.append((pos.id, reason))
        return pos


def make_pos(pid="abc123def456", side="long", entry=100.0, market="xyz:MRVL",
             symbol="MRVL") -> Position:
    t = now_ms()
    return Position(
        id=pid, news_id="n1", market=market, symbol=symbol, dex="xyz", side=side,
        size=2.0, entry_px=entry, stop_loss=entry * (0.97 if side == "long" else 1.03),
        take_profit=0.0, leverage=5, notional_usd=entry * 2, opened_ms=t - 3_600_000,
        time_exit_ms=t + 71 * 3_600_000, dry_run=False, trail_pct=0.08, peak_px=entry,
        confidence=0.84, rationale="strong datacenter guide", news_title="MRVL beats",
        news_source="tradfi")


def make_pm(mids=None, executor=None) -> PositionManager:
    return PositionManager(Config(), FakeHL(mids), executor or FakeExecutor())


def make_notifier(pm) -> tuple[TelegramNotifier, list]:
    cfg = Config()
    n = TelegramNotifier(cfg, EventBus(), pm=pm)
    n.token, n.chat_id, n.enabled = "x:y", CHAT, True
    calls = []

    async def fake_api(method, payload):
        calls.append((method, payload))
        return {"ok": True, "result": {"message_id": 42}}

    n.api = fake_api
    return n, calls


# ---- snapshot math ----------------------------------------------------------

def test_snapshot_long_and_short():
    async def go():
        pm = make_pm(mids={"xyz:MRVL": 105.0, "BTC": 90.0})
        long = make_pos()
        pm.track(long)
        s = await pm.snapshot(long)
        assert s["mark"] == 105.0
        assert abs(s["upnl"] - 10.0) < 1e-9          # (105-100) * 2
        assert abs(s["roe"] - 25.0) < 1e-9           # 10 / (200/5) * 100
        assert s["eff_stop"] == long.stop_loss and s["stop_label"] == "stop loss"
        assert long.peak_px == 100.0                 # read-only: trail NOT advanced

        short = make_pos(pid="ffffffffffff", side="short", market="BTC", symbol="BTC")
        pm.track(short)
        s2 = await pm.snapshot(short)
        assert abs(s2["upnl"] - 20.0) < 1e-9         # (100-90) * 2

        # no live price -> unknowns, never raises
        pm2 = make_pm(mids={})
        pm2.track(long)
        s3 = await pm2.snapshot(long)
        assert s3["mark"] is None and s3["upnl"] is None
    asyncio.run(go())


def test_position_card_fields():
    pos = make_pos()
    snap = {"mark": 105.0, "upnl": 10.0, "roe": 25.0,
            "eff_stop": 99.0, "stop_label": "trailing stop"}
    card = format_position_card(pos, snap, 0, 2, now_ms())
    for needle in ("MRVL", "LONG", "1/2", "+$10.00", "trailing stop",
                   "Exits in", "datacenter", "×5", "updated"):
        assert needle in card, needle
    assert "\n\n" in card   # grouped blocks, not one flat list
    # no-price snapshot degrades, doesn't blow up
    card2 = format_position_card(pos, {"mark": None}, 0, 1, now_ms())
    assert "no live price" in card2


# ---- update routing ---------------------------------------------------------

def msg_update(text, chat=CHAT):
    return {"update_id": 1, "message": {"chat": {"id": int(chat)}, "text": text}}


def cb_update(data, chat=CHAT):
    return {"update_id": 2, "callback_query": {
        "id": "cb1", "data": data,
        "message": {"message_id": 42, "chat": {"id": int(chat)}}}}


def test_start_command_and_chat_restriction():
    async def go():
        pm = make_pm(mids={"xyz:MRVL": 105.0})
        pos = make_pos()
        pm.track(pos)
        n, calls = make_notifier(pm)

        await n._handle_update(msg_update("/start", chat="666"))   # stranger -> silence
        assert calls == []

        await n._handle_update(msg_update("/start"))
        method, payload = calls[-1]
        assert method == "sendMessage" and "MRVL" in payload["text"]
        kb = payload["reply_markup"]["inline_keyboard"]
        assert kb[-1][-1]["callback_data"] == f"pb:a:0:{pos.id}"   # close asks, by id
        assert len(kb) == 1                                        # single pos: no nav row

        pm.track(make_pos(pid="222222222222", market="BTC", symbol="BTC"))
        calls.clear()
        await n._handle_update(msg_update("/positions"))
        kb = calls[-1][1]["reply_markup"]["inline_keyboard"]
        assert len(kb) == 2 and kb[0][1]["text"] == "1/2"          # nav row appears
    asyncio.run(go())


def test_nav_refresh_and_callback_restriction():
    async def go():
        pm = make_pm(mids={"xyz:MRVL": 105.0, "BTC": 105.0})
        pm.track(make_pos())
        pm.track(make_pos(pid="222222222222", market="BTC", symbol="BTC"))
        n, calls = make_notifier(pm)

        await n._handle_update(cb_update("pb:n:1", chat="666"))    # stranger callback
        assert calls == []

        await n._handle_update(cb_update("pb:n:1"))
        methods = [m for m, _ in calls]
        assert "editMessageText" in methods and "answerCallbackQuery" in methods
        edit = next(p for m, p in calls if m == "editMessageText")
        assert "2/2" in edit["text"] and edit["message_id"] == 42
    asyncio.run(go())


def test_close_two_step_flow():
    async def go():
        ex = FakeExecutor()
        pm = make_pm(mids={"xyz:MRVL": 105.0}, executor=ex)
        pos = make_pos()
        pm.track(pos)
        n, calls = make_notifier(pm)

        # step 1: ask -> confirmation keyboard, nothing closed yet
        await n._handle_update(cb_update(f"pb:a:0:{pos.id}"))
        edit = next(p for m, p in calls if m == "editMessageText")
        assert "Market-close" in edit["text"]
        kb = edit["reply_markup"]["inline_keyboard"][0]
        assert kb[0]["callback_data"] == f"pb:k:{pos.id}"
        assert kb[1]["callback_data"] == "pb:n:0"                  # cancel returns to card
        assert ex.closed == [] and pm.position_by_id(pos.id) is not None

        # step 2: confirm -> force_close with the manual reason, untracked after
        calls.clear()
        await n._handle_update(cb_update(f"pb:k:{pos.id}"))
        assert ex.closed == [(pos.id, "manual close (telegram)")]
        assert pm.position_by_id(pos.id) is None
        edit = next(p for m, p in calls if m == "editMessageText")
        assert "Closed" in edit["text"] and "No open positions" in edit["text"]

        # confirming again (stale button) is harmless
        calls.clear()
        await n._handle_update(cb_update(f"pb:k:{pos.id}"))
        assert ex.closed == [(pos.id, "manual close (telegram)")]
        assert "Already closed" in next(p for m, p in calls if m == "editMessageText")["text"]
    asyncio.run(go())


def test_close_not_filled_keeps_position():
    async def go():
        ex = FakeExecutor(fail=True)
        pm = make_pm(mids={"xyz:MRVL": 105.0}, executor=ex)
        pos = make_pos()
        pm.track(pos)
        n, calls = make_notifier(pm)
        await n._handle_update(cb_update(f"pb:k:{pos.id}"))
        assert pm.position_by_id(pos.id) is not None               # still tracked/managed
        edit = next(p for m, p in calls if m == "editMessageText")
        assert "NOT fill" in edit["text"]
    asyncio.run(go())


def test_auto_refresh_lifecycle():
    async def go():
        from hlbot.notify.telegram import CARD_REFRESH_TTL
        pm = make_pm(mids={"xyz:MRVL": 105.0})
        pos = make_pos()
        pm.track(pos)
        n, calls = make_notifier(pm)

        await n._refresh_card_once()                       # nothing tracked yet -> no-op
        assert calls == []

        await n._handle_update(msg_update("/start"))       # /start starts tracking the card
        assert n._card and n._card["message_id"] == 42
        calls.clear()
        await n._refresh_card_once()
        edit = next(p for m, p in calls if m == "editMessageText")
        assert "MRVL" in edit["text"] and edit["message_id"] == 42

        await n._handle_update(cb_update(f"pb:a:0:{pos.id}"))   # pending confirmation...
        calls.clear()
        await n._refresh_card_once()
        assert calls == []                                 # ...pauses refresh (no clobber)

        await n._handle_update(cb_update("pb:n:0"))        # cancel re-arms it
        calls.clear()
        await n._refresh_card_once()
        assert any(m == "editMessageText" for m, _ in calls)

        n._card["touched"] -= CARD_REFRESH_TTL + 1         # TTL expiry: stamp pause once
        calls.clear()
        await n._refresh_card_once()
        assert "paused" in next(p for m, p in calls if m == "editMessageText")["text"]
        calls.clear()
        await n._refresh_card_once()                       # then stay quiet
        assert calls == []

        await n._handle_update(cb_update("pb:r:0"))        # any tap re-arms
        calls.clear()
        await n._refresh_card_once()
        assert any(m == "editMessageText" for m, _ in calls)
    asyncio.run(go())


def test_poll_commands_disabled_returns_cleanly():
    async def go():
        n = TelegramNotifier(Config(), EventBus(), pm=None)
        n.enabled = False
        await n.poll_commands()   # clean return = supervised() treats it as disabled
    asyncio.run(go())
