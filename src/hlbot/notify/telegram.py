"""Telegram notifier + bot commands. Subscribes to trade events on the bus and pushes
rich alerts to your bot chat/channel, and long-polls that same bot for commands:
/start (or /positions) opens an inline-keyboard position browser — prev/next arrows,
refresh, and a two-step manual emergency close. No-ops gracefully if no token/chat id
is configured."""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

from ..bus import EventBus
from ..config import Config
from ..models import Position, now_ms
from .formatting import (SEP, _e, _num, _usd, format_entry, format_error, format_exit,
                         format_position_card)

log = logging.getLogger("hlbot.telegram")

POLL_TIMEOUT = 45   # getUpdates long-poll seconds (client timeout sits above it)

# The latest /start browser card auto-refreshes (editMessageText) on this cadence —
# well under Telegram's flood limits — and stops this long after the last button
# press, stamping the card "paused" so staleness is visible. Any tap re-arms it.
CARD_REFRESH_SECONDS = 30
CARD_REFRESH_TTL = 30 * 60


class TelegramNotifier:
    def __init__(self, config: Config, bus: EventBus, pm=None):
        self.config = config
        self.token = config.secrets.telegram_bot_token
        self.chat_id = config.secrets.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id and not self.token.startswith("123456"))
        self.pm = pm   # PositionManager; commands are disabled without it
        self._q = bus.subscribe("trade.open", "trade.close", "trade.error", "status")
        self._stop = asyncio.Event()
        self._client: httpx.AsyncClient | None = None   # long-lived client of the poll task
        # Latest /start browser message; only this one auto-refreshes (older cards freeze
        # at their "updated …" stamp): {chat_id, message_id, idx, touched, confirming,
        # expired}. confirming=True pauses refresh so an edit can't clobber a pending
        # close-confirmation prompt.
        self._card: dict | None = None

    async def run(self) -> None:
        if not self.enabled:
            log.warning("Telegram alerts disabled (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")
        while not self._stop.is_set():
            event = await self._q.get()
            try:
                await self._handle(event)
            except Exception:  # noqa: BLE001
                log.exception("Telegram handler error")

    async def _handle(self, event) -> None:
        payload = event.payload
        if event.topic == "trade.open" and isinstance(payload, Position):
            await self.send(format_entry(payload))
        elif event.topic == "trade.close" and isinstance(payload, Position):
            await self.send(format_exit(payload))
        elif event.topic == "trade.error":
            await self.send(format_error(payload))
        elif event.topic == "status":
            await self.send(f"ℹ️ {payload}")
        if event.topic in ("trade.open", "trade.close"):
            # keep the browser card's count/PnL current without waiting a cadence
            try:
                await self._refresh_card_once()
            except Exception:  # noqa: BLE001 - display-only
                log.debug("card refresh after trade event failed")

    # ---- bot API plumbing ---------------------------------------------------
    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    async def api(self, method: str, payload: dict) -> dict:
        """One bot-API call. Returns the parsed response — {'ok': False, ...} on transport
        errors so callers branch on 'ok' instead of wrapping every call in try/except."""
        try:
            if self._client is not None:
                r = await self._client.post(self._url(method), json=payload)
            else:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(self._url(method), json=payload)
            return r.json()
        except Exception:  # noqa: BLE001
            log.exception("Telegram API %s failed", method)
            return {"ok": False, "description": "transport error"}

    async def send(self, text: str) -> None:
        if not self.enabled:
            return
        resp = await self.api("sendMessage", {
            "chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": False})
        if not resp.get("ok"):
            log.warning("Telegram send failed: %s", str(resp)[:200])

    # ---- command loop (getUpdates long poll) --------------------------------
    async def poll_commands(self) -> None:
        """Long-poll bot commands. A clean return (= supervised() won't restart it) when
        the notifier is unconfigured or no position manager is attached."""
        if not self.enabled or self.pm is None:
            log.info("Telegram commands disabled (no token/chat id, or no position manager)")
            return
        async with httpx.AsyncClient(timeout=httpx.Timeout(POLL_TIMEOUT + 15,
                                                           connect=10)) as client:
            self._client = client
            refresher = asyncio.create_task(self._card_refresh_loop())
            try:
                offset = await self._skip_backlog(client)
                log.info("Telegram command polling started (/start shows positions)")
                while not self._stop.is_set():
                    offset = await self._poll_once(client, offset)
            finally:
                refresher.cancel()
                self._client = None

    async def _skip_backlog(self, client: httpx.AsyncClient) -> int | None:
        """Discard updates queued while the bot was down. A stale /start is just noise,
        but a stale close-confirmation callback replaying on reboot would be a money
        action — start strictly after the newest pending update."""
        try:
            r = await client.post(self._url("getUpdates"), json={"offset": -1, "timeout": 0})
            res = r.json().get("result") or []
            return res[-1]["update_id"] + 1 if res else None
        except Exception:  # noqa: BLE001 - fall through; worst case we re-skip on first poll
            return None

    async def _poll_once(self, client: httpx.AsyncClient, offset: int | None) -> int | None:
        try:
            payload = {"timeout": POLL_TIMEOUT, "allowed_updates": ["message", "callback_query"]}
            if offset is not None:
                payload["offset"] = offset
            r = await client.post(self._url("getUpdates"), json=payload)
            if r.status_code == 409:   # webhook set or a second poller — never spin on it
                log.warning("getUpdates conflict (409): another consumer of this bot token")
                await asyncio.sleep(30)
                return offset
            data = r.json()
            if not data.get("ok"):
                log.warning("getUpdates failed: %s", str(data)[:200])
                await asyncio.sleep(5)
                return offset
            for upd in data.get("result", []):
                offset = max(offset or 0, upd.get("update_id", 0) + 1)
                try:
                    await self._handle_update(upd)
                except Exception:  # noqa: BLE001 - one bad update must not kill the loop
                    log.exception("Telegram command handler error")
            return offset
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("getUpdates poll error")
            await asyncio.sleep(5)
            return offset

    async def _handle_update(self, upd: dict) -> None:
        msg = upd.get("message")
        if msg:
            # Only the configured chat may command the bot — anyone can find a bot
            # username, and these buttons close real positions.
            if str((msg.get("chat") or {}).get("id")) != str(self.chat_id):
                return
            text = (msg.get("text") or "").strip().lower()
            if text.startswith("/start") or text.startswith("/positions"):
                body, kb = await self._render_browser(0)
                resp = await self.api("sendMessage", {
                    "chat_id": self.chat_id, "text": body, "parse_mode": "HTML",
                    "disable_web_page_preview": True, "reply_markup": kb})
                sent_id = ((resp.get("result") or {}).get("message_id")
                           if resp.get("ok") else None)
                if sent_id:
                    self._touch_card(self.chat_id, sent_id, 0)
            return
        cq = upd.get("callback_query")
        if cq:
            chat = ((cq.get("message") or {}).get("chat") or {})
            if str(chat.get("id")) != str(self.chat_id):
                return
            await self._handle_callback(cq)

    # ---- position browser ----------------------------------------------------
    # Callback data (Telegram caps it at 64 bytes):
    #   pb:n:<idx>          show position idx (nav / cancel-close)
    #   pb:r:<idx>          refresh position idx
    #   pb:a:<idx>:<pos_id> ask close confirmation
    #   pb:k:<pos_id>       confirmed close — ALWAYS keyed by position id, never index:
    #                       the list can shift between presses and an index-addressed
    #                       close could market-close the wrong position.

    def _sorted_positions(self) -> list[Position]:
        return sorted(self.pm.open_positions(), key=lambda p: -p.opened_ms)

    async def _render_browser(self, idx: int) -> tuple[str, dict]:
        poss = self._sorted_positions()
        if not poss:
            return ("📭 <b>No open positions.</b>",
                    {"inline_keyboard": [[{"text": "🔄 Refresh", "callback_data": "pb:r:0"}]]})
        idx %= len(poss)
        pos = poss[idx]
        snap = await self.pm.snapshot(pos)
        text = format_position_card(pos, snap, idx, len(poss), now_ms())
        rows = []
        if len(poss) > 1:
            rows.append([
                {"text": "◀", "callback_data": f"pb:n:{(idx - 1) % len(poss)}"},
                {"text": f"{idx + 1}/{len(poss)}", "callback_data": f"pb:r:{idx}"},
                {"text": "▶", "callback_data": f"pb:n:{(idx + 1) % len(poss)}"},
            ])
        rows.append([{"text": "🔄 Refresh", "callback_data": f"pb:r:{idx}"},
                     {"text": "🚨 Close", "callback_data": f"pb:a:{idx}:{pos.id}"}])
        return text, {"inline_keyboard": rows}

    # ---- card auto-refresh -----------------------------------------------------
    def _touch_card(self, chat_id, message_id, idx: int, confirming: bool = False) -> None:
        """(Re)track the browser card being interacted with and re-arm its TTL."""
        self._card = {"chat_id": chat_id, "message_id": message_id, "idx": idx,
                      "touched": time.monotonic(), "confirming": confirming,
                      "expired": False}

    async def _card_refresh_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(CARD_REFRESH_SECONDS)
            try:
                await self._refresh_card_once()
            except Exception:  # noqa: BLE001 - display-only, keep ticking
                log.exception("Card auto-refresh error")

    async def _refresh_card_once(self) -> None:
        card = self._card
        if not card or card["confirming"]:   # never clobber a pending close confirmation
            return
        if time.monotonic() - card["touched"] > CARD_REFRESH_TTL:
            if not card["expired"]:          # stamp the pause exactly once, then go quiet
                card["expired"] = True
                text, kb = await self._render_browser(card["idx"])
                await self._edit(card["chat_id"], card["message_id"],
                                 text + "\n⏸ <i>auto-refresh paused — tap 🔄 to resume</i>",
                                 kb)
            return
        text, kb = await self._render_browser(card["idx"])
        await self._edit(card["chat_id"], card["message_id"], text, kb)

    async def _handle_callback(self, cq: dict) -> None:
        data = (cq.get("data") or "")
        msg = cq.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        mid = msg.get("message_id")
        toast = None
        try:
            parts = data.split(":")
            if parts[0] != "pb" or len(parts) < 3 or mid is None:
                return
            action = parts[1]
            if action in ("n", "r"):
                idx = int(parts[2])
                self._touch_card(chat_id, mid, idx)
                text, kb = await self._render_browser(idx)
                changed = await self._edit(chat_id, mid, text, kb)
                if action == "r":
                    toast = "Updated" if changed else "No change"
            elif action == "a" and len(parts) >= 4:
                self._touch_card(chat_id, mid, int(parts[2]), confirming=True)
                toast = await self._ask_close(chat_id, mid, int(parts[2]), parts[3])
            elif action == "k":
                self._touch_card(chat_id, mid, 0)
                head = await self._close_position(parts[2])
                body, kb = await self._render_browser(0)
                await self._edit(chat_id, mid, head + "\n\n" + body, kb)
        finally:
            ack = {"callback_query_id": cq.get("id")}
            if toast:
                ack["text"] = toast
            await self.api("answerCallbackQuery", ack)

    async def _ask_close(self, chat_id, mid, idx: int, pos_id: str) -> str | None:
        pos = self.pm.position_by_id(pos_id)
        if pos is None:
            # Clear the confirming flag the caller just set, or auto-refresh stays silently
            # paused on this card until the next tap.
            self._touch_card(chat_id, mid, idx, confirming=False)
            text, kb = await self._render_browser(0)
            await self._edit(chat_id, mid, text, kb)
            return "Already closed"
        snap = await self.pm.snapshot(pos)
        total = max(len(self.pm.open_positions()), 1)
        text = format_position_card(pos, snap, idx % total, total, now_ms())
        mode = ("🧪 paper position" if pos.dry_run
                else "🔴 <b>LIVE position — real market order</b>")
        text += f"\n\n{SEP}\n⚠️ <b>Market-close {_e(pos.symbol)} now?</b>\n{mode}"
        kb = {"inline_keyboard": [[
            {"text": "✅ Confirm close", "callback_data": f"pb:k:{pos.id}"},
            {"text": "✖ Cancel", "callback_data": f"pb:n:{idx}"},
        ]]}
        await self._edit(chat_id, mid, text, kb)
        return None

    async def _close_position(self, pos_id: str) -> str:
        pos = self.pm.position_by_id(pos_id)
        if pos is None:
            return "ℹ️ Already closed."
        log.warning("Manual close requested via Telegram for %s (%s)", pos.id, pos.market)
        ok = await self.pm.force_close(pos, "manual close (telegram)")
        if ok:
            return (f"✅ <b>Closed {_e(pos.symbol)}</b> @ {_num(pos.exit_px)} · "
                    f"PnL <b>{_usd(pos.pnl_usd)}</b>")
        if self.pm.position_by_id(pos_id) is None:
            return "ℹ️ The bot closed this position concurrently."
        return ("⚠️ <b>Close did NOT fill</b> — position still open; the bot keeps "
                "managing it (stop/trail/time exits stay active and it will retry).")

    async def _edit(self, chat_id, message_id, text: str, kb: dict) -> bool:
        """editMessageText; 'message is not modified' is an expected no-op (a refresh
        with an unchanged card), anything else logs. True = the message changed."""
        resp = await self.api("editMessageText", {
            "chat_id": chat_id, "message_id": message_id, "text": text,
            "parse_mode": "HTML", "disable_web_page_preview": True, "reply_markup": kb})
        if resp.get("ok"):
            return True
        if "not modified" not in (resp.get("description") or ""):
            log.warning("editMessageText failed: %s", str(resp)[:200])
        return False

    def stop(self) -> None:
        self._stop.set()
