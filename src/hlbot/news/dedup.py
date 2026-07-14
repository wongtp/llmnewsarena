"""Pre-analysis filtering & dedup: drop duplicates, retweets/replies, stale and
off-whitelist news before we spend a Claude call or risk capital."""
from __future__ import annotations

from collections import deque

from ..config import FiltersConfig
from ..models import NewsItem


class Dedup:
    def __init__(self, filters: FiltersConfig, memory: int = 20000):
        self.filters = filters
        self._seen_ids: set[str] = set()
        self._order: deque[str] = deque()  # for bounded eviction
        self.memory = memory

    def restore(self, ids: list[str]) -> int:
        """Seed the seen-set from persisted news ids (newest-first) so a restart doesn't
        re-process news already handled. Inserted oldest-first to keep eviction order."""
        for key in reversed(ids):
            if key:
                self._remember(key)
        return len(self._seen_ids)

    def _remember(self, key: str) -> None:
        self._seen_ids.add(key)
        self._order.append(key)
        while len(self._order) > self.memory:
            self._seen_ids.discard(self._order.popleft())

    def check(self, item: NewsItem) -> tuple[bool, str]:
        """Return (keep, reason). reason is the rejection reason when keep is False.

        Dedup is by Tree's unique `_id` only. We deliberately do NOT dedup by author
        (Tree's twitterId is the account id, not the tweet id — deduping on it would
        drop every tweet after an account's first). Same-story repeats are handled
        downstream by the per-ticker cooldown.
        """
        if item.id in self._seen_ids:
            return False, "duplicate id"
        self._remember(item.id)

        f = self.filters
        if f.skip_retweets and item.is_retweet:
            return False, "retweet"
        if f.skip_replies and item.is_reply:
            return False, "reply"
        if f.skip_quotes and item.is_quote:
            return False, "quote tweet"
        if item.age_seconds > f.max_news_age_seconds:
            return False, f"stale ({item.age_seconds:.0f}s old)"

        haystack = f"{item.source or ''} {item.title or ''}".lower()
        if f.source_blacklist and any(b.lower() in haystack for b in f.source_blacklist):
            return False, "blacklisted source"
        if f.source_whitelist and not any(w.lower() in haystack for w in f.source_whitelist):
            return False, "source not in whitelist"

        return True, ""
