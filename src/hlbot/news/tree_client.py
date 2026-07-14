"""Tree of Alpha news websocket client.

Connects to wss://news.treeofalpha.com/ws, authenticates with `login {API_KEY}`
(your key removes the throttle delay), parses messages into NewsItem, and hands
each one to the supplied async callback. Auto-reconnects with backoff.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import websockets

from ..config import Config
from ..models import NewsItem

log = logging.getLogger("hlbot.tree")

WS_URL = "wss://news.treeofalpha.com/ws"


class TreeClient:
    def __init__(self, config: Config, on_news: Callable[[NewsItem], Awaitable[None]]):
        self.config = config
        self.on_news = on_news
        self.connected = False
        self._stop = asyncio.Event()

    def status(self) -> dict:
        """Feed-link state for the dashboard header."""
        return {"enabled": bool(self.config.app.enable_tree_feed), "connected": self.connected}

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    WS_URL, ping_interval=20, ping_timeout=20, max_size=2**22
                ) as ws:
                    await ws.send(f"login {self.config.secrets.tree_api_key}")
                    log.info("Connected to Tree of Alpha news websocket")
                    self.connected = True
                    backoff = 1.0
                    async for raw in ws:
                        await self._handle_raw(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any failure
                if self._stop.is_set():
                    break
                log.warning("Tree WS error (%s); reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                self.connected = False

    async def _handle_raw(self, raw) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return  # login ack / non-JSON control frame
        item = NewsItem.from_tree(msg)
        if item is None:
            return
        try:
            await self.on_news(item)
        except Exception:  # noqa: BLE001 - one bad item shouldn't kill the feed
            log.exception("Error handling news item %s", item.id)

    def stop(self) -> None:
        self._stop.set()
