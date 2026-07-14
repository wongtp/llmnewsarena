"""Tiny async pub/sub event bus.

Components publish events; the SQLite store, Telegram notifier, and the web UI
subscribe. The trade pipeline itself runs as a direct async chain (so ordering and
back-pressure are explicit); the bus is for fan-out to observers/persistence.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    topic: str
    payload: Any
    ts: float = field(default_factory=time.time)


class EventBus:
    def __init__(self, queue_maxsize: int = 1000):
        self._topic_subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._all_subs: list[asyncio.Queue] = []
        self._queue_maxsize = queue_maxsize

    def subscribe(self, *topics: str) -> asyncio.Queue:
        """Subscribe to specific topics, or to everything if no topic given."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        if not topics:
            self._all_subs.append(q)
        for t in topics:
            self._topic_subs[t].append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._all_subs:
            self._all_subs.remove(q)
        for subs in self._topic_subs.values():
            if q in subs:
                subs.remove(q)

    async def publish(self, topic: str, payload: Any) -> None:
        event = Event(topic=topic, payload=payload)
        for q in list(self._topic_subs.get(topic, [])) + list(self._all_subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event for a slow consumer rather than block the pipeline.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass
