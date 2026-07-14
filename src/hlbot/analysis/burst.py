"""Earnings/news burst aggregation (default OFF: analyzer.burst_window_seconds = 0).

Wire services often print one event as several messages seconds apart — EPS line, then
revenue, then guidance. Analyzed in isolation, each piece undersells the event (an
in-line EPS line scores "none" even when the guidance line two messages later is a
shock). The analyzer demonstrably reads BUNDLED multi-line headlines well; this module
synthesizes the bundle when the wire dribbles:

- every analyzed piece with a resolved ticker is buffered (time, text);
- when a piece arrives for a ticker that already has pieces inside the window, the
  caller re-analyzes the chronological CONCATENATION and uses that verdict instead;
- the FIRST piece always trades on its solo verdict immediately (empty buffer) — a
  blowout EPS line never waits for the rest of the burst.

Used by the live pipeline and the replay engine with identical semantics; the replay
caches combined verdicts under a deterministic "<id>+b<n>" key so A/B re-runs are free.
"""
from __future__ import annotations

from dataclasses import replace

from ..models import NewsItem

MAX_PIECES = 5   # bound the combined prompt during a long dribble


class BurstBuffer:
    """Per-ticker rolling buffer of recently analyzed news pieces."""

    def __init__(self, window_seconds: float):
        self.window_ms = int(window_seconds * 1000)
        self._by_ticker: dict[str, list[tuple[int, str]]] = {}

    @property
    def enabled(self) -> bool:
        return self.window_ms > 0

    def prior(self, ticker: str, time_ms: int) -> list[str]:
        """Texts already buffered for this ticker still inside the window (oldest first)."""
        if not self.enabled or not ticker:
            return []
        kept = [(t, s) for t, s in self._by_ticker.get(ticker.upper(), [])
                if time_ms - t <= self.window_ms and t <= time_ms]
        self._by_ticker[ticker.upper()] = kept
        return [s for _, s in kept]

    def add(self, ticker: str, time_ms: int, text: str) -> None:
        if not self.enabled or not ticker or not text.strip():
            return
        buf = self._by_ticker.setdefault(ticker.upper(), [])
        if any(s == text for _, s in buf):   # exact re-print adds nothing
            return
        buf.append((time_ms, text))
        del buf[:-MAX_PIECES]


def build_burst_item(item: NewsItem, prior_texts: list[str]) -> NewsItem:
    """The combined pseudo-item: earlier wire lines + the current one, chronological,
    as one multi-line body (the same shape as a natively-bundled wire message). The id
    is deterministic for replay caching."""
    body = "\n".join([*prior_texts, item.body.strip() or item.title.strip()])
    return replace(item, id=f"{item.id}+b{len(prior_texts)}", body=body)
