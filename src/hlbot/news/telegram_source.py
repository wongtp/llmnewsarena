"""Telegram channel ingestion via the MTProto user client (Telethon).

Reads new posts from the configured channels and feeds them into the SAME pipeline as Tree
news. Two ingestion paths run together for resilience:

  * a live ``NewMessage`` handler — lowest latency, used while the link is healthy; and
  * a **catch-up poll loop** that every ``telegram_poll_seconds`` re-fetches each channel's
    newest messages AND actively verifies/repairs the connection.

The live handler ALONE is not reliable: a network or VPN drop aborts the socket; Telethon
reconnects at the transport layer, but the handler silently stops firing — so the bot goes
deaf with no error and no trade. The poll loop is the backstop that keeps ingestion alive
(and forces a clean reconnect) within one poll interval even when the handler is dead. Both
paths share a per-channel "highest msg id seen" so neither re-feeds the other (dedup downstream
is a further safety net); the seed is the current tip, so only messages from NOW on are taken
(startup history backfill is handled separately).

Requires TELEGRAM_API_ID/HASH and a pre-created session (run scripts/telegram_login.py once).
If not configured/authorized, it no-ops instead of blocking on an interactive prompt.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from ..config import Config
from ..models import NewsItem

log = logging.getLogger("hlbot.telegram_src")

SESSION_PATH = "data/tg_session"


class TelegramSource:
    def __init__(self, config: Config, on_news: Callable[[NewsItem], Awaitable[None]]):
        self.config = config
        self.on_news = on_news
        self.api_id = int(config.secrets.telegram_api_id) if config.secrets.telegram_api_id else 0
        self.api_hash = config.secrets.telegram_api_hash
        self.channels = config.app.telegram_channels
        self.poll_seconds = config.app.telegram_poll_seconds
        self._client = None
        self._ents: dict[int, object] = {}     # peer_id -> entity
        self._seen_max: dict[int, int] = {}    # peer_id -> highest msg id processed
        self._unresolved: set[str] = set()     # channels that failed to resolve (retried by poll)
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.api_id and self.api_hash and self.channels)

    def status(self) -> dict:
        """Feed-link state for the dashboard header (cheap; no network round-trip)."""
        try:
            connected = bool(self._client is not None and self._client.is_connected())
        except Exception:  # noqa: BLE001 - display-only
            connected = False
        return {"enabled": self.enabled, "connected": connected}

    async def _ingest(self, chat, msg_id: int, text: str, date) -> bool:
        """Build a NewsItem from a channel message and feed the pipeline. Returns True if a real
        item was produced. Downstream dedup (by id) drops anything already seen."""
        if not (text and text.strip()):
            return False
        uname = getattr(chat, "username", None)
        title = getattr(chat, "title", None) or uname or "Telegram"
        item = NewsItem.from_telegram(
            channel=uname or str(getattr(chat, "id", "")),
            channel_title=title, msg_id=msg_id, text=text,
            date_ms=int(date.timestamp() * 1000), chat_id=getattr(chat, "id", ""))
        if item:
            await self.on_news(item)
            return True
        return False

    async def run(self) -> None:
        if not self.enabled:
            log.info("Telegram source disabled (set TELEGRAM_API_ID/HASH + telegram_channels)")
            return
        from telethon import TelegramClient, events
        from telethon.utils import get_peer_id

        # Robust client: many reconnect attempts + auto-reconnect. The poll loop is the real
        # safety net, but these reduce how often it has to force a reconnect.
        self._client = TelegramClient(
            SESSION_PATH, self.api_id, self.api_hash,
            connection_retries=10, retry_delay=3, auto_reconnect=True, request_retries=5)
        await self._client.connect()
        if not await self._client.is_user_authorized():
            log.error("Telegram session not authorized. Run: python scripts/telegram_login.py")
            await self._client.disconnect()
            return

        # Live fast-path handler.
        @self._client.on(events.NewMessage(chats=self.channels))
        async def handler(event):  # noqa: ANN001
            try:
                chat = await event.get_chat()
                if await self._ingest(chat, event.id, event.message.message or "",
                                      event.message.date):
                    self._seen_max[event.chat_id] = max(self._seen_max.get(event.chat_id, 0),
                                                         event.id)
            except Exception:  # noqa: BLE001
                log.exception("Telegram message handler error")

        # Resolve entities + seed seen-ids to the current tip so the poll only takes messages
        # arriving from NOW on (recent history is loaded by the startup backfill, not here).
        for ch in self.channels:
            if not await self._resolve_channel(ch, get_peer_id):
                self._unresolved.add(ch)   # poll loop keeps retrying — a startup blip must
                #                            not leave a channel dark until the next restart

        log.info("Telegram source listening on %d channels (live handler + %ds catch-up poll)",
                 len(self._ents), self.poll_seconds)
        await self._poll_loop()

    async def _resolve_channel(self, ch, get_peer_id) -> bool:
        """Resolve one configured channel to an entity and seed its seen-id to the current tip.
        False on failure (transient network / not yet joined)."""
        try:
            ent = await self._client.get_entity(ch)
        except Exception:  # noqa: BLE001
            log.warning("Cannot resolve Telegram channel %r (joined? correct username?)", ch)
            return False
        pid = get_peer_id(ent)
        self._ents[pid] = ent
        self._seen_max.setdefault(pid, 0)
        try:
            async for msg in self._client.iter_messages(ent, limit=1):
                self._seen_max[pid] = msg.id
        except Exception:  # noqa: BLE001
            pass
        return True

    async def _poll_loop(self) -> None:
        """Backstop + health: every poll_seconds, pull each channel's new messages and make sure
        the link is alive. Survives the live handler silently dying after a network/VPN drop."""
        if self.poll_seconds <= 0:
            await self._client.run_until_disconnected()   # poll disabled: old handler-only mode
            return
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
                break   # stop signaled
            except asyncio.TimeoutError:
                pass
            try:
                if not self._client.is_connected():
                    log.warning("Telegram link down; reconnecting")
                    await self._reconnect()
                if self._unresolved:
                    from telethon.utils import get_peer_id
                    for ch in list(self._unresolved):
                        if await self._resolve_channel(ch, get_peer_id):
                            self._unresolved.discard(ch)
                            log.info("Telegram channel %r resolved on retry", ch)
                total = 0
                for pid, ent in list(self._ents.items()):
                    total += await self._poll_channel(pid, ent,
                                                      timeout=max(20, self.poll_seconds))
                if total:
                    log.info("Telegram catch-up poll: ingested %d new message(s)", total)
            except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
                log.warning("Telegram poll stalled (%s); forcing reconnect", exc)
                await self._reconnect()
            except Exception:  # noqa: BLE001
                log.exception("Telegram poll loop error")

    async def _poll_channel(self, pid: int, ent, timeout: float = 60.0) -> int:
        """Fetch new messages, then ingest them. The timeout covers ONLY the network fetch
        (a hung fetch means a dead link; TimeoutError bubbles up to force a reconnect) —
        ingestion is a fast pipeline enqueue and must never be cancelled mid-handoff.
        `_seen_max` advances only AFTER a message's ingest attempt completes, so a crash or
        cancellation mid-batch re-fetches the unprocessed remainder next poll instead of
        silently losing it (downstream dedup-by-id drops any overlap)."""
        last = self._seen_max.get(pid, 0)
        msgs = []

        async def fetch() -> None:
            # reverse=True iterates OLDEST-first from min_id upward: if more than `limit`
            # messages landed in one poll interval, we take the oldest chunk and get the rest
            # next poll — newest-first would advance _seen_max past the overflow and skip it
            # forever (the staleness filter downstream drops what's too old to trade anyway).
            async for m in self._client.iter_messages(ent, min_id=last, limit=50, reverse=True):
                msgs.append(m)

        await asyncio.wait_for(fetch(), timeout=timeout)
        cnt = 0
        for msg in msgs:   # already oldest -> newest
            if msg.message and await self._ingest(ent, msg.id, msg.message, msg.date):
                cnt += 1
            self._seen_max[pid] = max(self._seen_max.get(pid, 0), msg.id)
        return cnt

    async def _reconnect(self) -> None:
        """Force a clean reconnect (disconnect + connect) with backoff — the registered handler
        persists across this, and the poll keeps ingesting regardless."""
        try:
            await self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        backoff = 2.0
        while not self._stop.is_set():
            try:
                await self._client.connect()
                if await self._client.is_user_authorized():
                    log.info("Telegram reconnected")
                    return
                log.error("Telegram reconnected but session not authorized")
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("Telegram reconnect failed (%s); retry in %.0fs", exc, backoff)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                return   # stop signaled during backoff
            except asyncio.TimeoutError:
                backoff = min(backoff * 2, 60)

    async def stop(self) -> None:
        self._stop.set()
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
