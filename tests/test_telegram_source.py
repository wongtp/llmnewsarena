"""Tests for the Telegram source's message->NewsItem ingest (the part that doesn't need a live
Telethon connection). The live handler + catch-up poll loop are integration-level (network)."""
import asyncio
import datetime as dt
import types

from hlbot.news.telegram_source import TelegramSource


def _src():
    # Bypass __init__ so we don't need real API creds; wire just what _ingest touches.
    s = object.__new__(TelegramSource)
    s._captured = []

    async def on_news(item):
        s._captured.append(item)

    s.on_news = on_news
    return s


def _chat():
    return types.SimpleNamespace(username="trad_fin", title="Tradfi", id=1643372154)


def _date():
    return dt.datetime(2026, 6, 8, 16, 0, 0, tzinfo=dt.timezone.utc)


def test_ingest_builds_news_item():
    s = _src()
    ok = asyncio.run(s._ingest(_chat(), 555, "AMD 1Q ADJ EPS $1.37, EST. $1.28 $AMD", _date()))
    assert ok and len(s._captured) == 1
    it = s._captured[0]
    assert it.id == "tg:trad_fin:555"
    assert it.source == "Telegram:trad_fin"
    assert "AMD" in it.body
    assert it.time_ms == int(_date().timestamp() * 1000)


def test_ingest_skips_empty_text():
    s = _src()
    assert asyncio.run(s._ingest(_chat(), 556, "   ", _date())) is False
    assert s._captured == []


# ---- catch-up poll: seen-id bookkeeping must never lose a message -----------
def _fake_client(msgs):
    class C:
        def iter_messages(self, ent, min_id=0, limit=50, reverse=False):
            async def gen():
                new = [m for m in msgs if m.id > min_id]
                # Telethon: newest-first by default; reverse=True walks oldest-first upward.
                new.sort(key=lambda m: m.id if reverse else -m.id)
                for m in new[:limit]:
                    yield m
            return gen()
    return C()


def _poll_src(msgs, on_news):
    s = object.__new__(TelegramSource)
    s._client = _fake_client(msgs)
    s._seen_max = {}
    s.on_news = on_news
    return s


def _msg(mid, text):
    return types.SimpleNamespace(id=mid, message=text, date=_date())


def test_poll_channel_ingests_oldest_first_and_advances_seen():
    captured = []

    async def on_news(item):
        captured.append(item.id)

    s = _poll_src([_msg(10, "first $AMD"), _msg(11, "second $AMD")], on_news)
    cnt = asyncio.run(s._poll_channel(7, _chat(), timeout=5))
    assert cnt == 2
    assert s._seen_max[7] == 11
    assert captured == ["tg:trad_fin:10", "tg:trad_fin:11"]


def test_poll_channel_overflow_takes_oldest_chunk_first():
    # More new messages than the fetch limit: the poll must take the OLDEST chunk and leave
    # _seen_max at its top, so the next poll picks up the remainder — advancing past the
    # overflow would skip those messages forever.
    captured = []

    async def on_news(item):
        captured.append(item.id)

    msgs = [_msg(i, f"m{i} $AMD") for i in range(1, 121)]   # 120 new, limit is 50
    s = _poll_src(msgs, on_news)
    cnt = asyncio.run(s._poll_channel(7, _chat(), timeout=5))
    assert cnt == 50
    assert s._seen_max[7] == 50          # oldest 1..50 processed; 51..120 next poll
    cnt2 = asyncio.run(s._poll_channel(7, _chat(), timeout=5))
    assert cnt2 == 50 and s._seen_max[7] == 100


def test_poll_channel_does_not_mark_seen_when_ingest_is_cancelled():
    # If ingestion dies mid-handoff (task cancelled), the message must NOT be recorded as
    # seen — the next poll re-fetches it (downstream dedup-by-id drops any overlap).
    async def on_news(item):
        raise asyncio.CancelledError()

    s = _poll_src([_msg(10, "first $AMD")], on_news)
    cancelled = False
    try:
        asyncio.run(s._poll_channel(7, _chat(), timeout=5))
    except asyncio.CancelledError:
        cancelled = True
    assert cancelled
    assert s._seen_max.get(7, 0) == 0
