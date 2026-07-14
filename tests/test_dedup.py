from hlbot.config import FiltersConfig
from hlbot.models import NewsItem, now_ms
from hlbot.news.dedup import Dedup


def _item(**kw):
    base = dict(id="1", title="t", body="b", source="X", link=None,
                time_ms=now_ms(), received_ms=now_ms())
    base.update(kw)
    return NewsItem(**base)


def test_duplicate_id_skipped():
    d = Dedup(FiltersConfig())
    keep, _ = d.check(_item(id="a"))
    assert keep
    keep2, reason = d.check(_item(id="a"))
    assert not keep2 and "duplicate" in reason


def test_same_author_different_tweets_both_kept():
    # Regression: Tree's twitterId is the author id; two distinct tweets from the
    # same account must BOTH be kept (we dedup by _id only).
    d = Dedup(FiltersConfig())
    k1, _ = d.check(_item(id="t1", author_id="999"))
    k2, _ = d.check(_item(id="t2", author_id="999"))
    assert k1 and k2


def test_retweet_skipped():
    d = Dedup(FiltersConfig(skip_retweets=True))
    keep, reason = d.check(_item(id="rt", is_retweet=True))
    assert not keep and reason == "retweet"


def test_stale_skipped():
    d = Dedup(FiltersConfig(max_news_age_seconds=60))
    old = now_ms() - 10 * 60 * 1000
    keep, reason = d.check(_item(id="old", time_ms=old))
    assert not keep and "stale" in reason


def test_whitelist_filters_sources():
    d = Dedup(FiltersConfig(source_whitelist=["tradfi"]))
    keep_yes, _ = d.check(_item(id="1", title="tradfi: big news", source="Twitter"))
    keep_no, reason = d.check(_item(id="2", title="random", source="Blogs"))
    assert keep_yes
    assert not keep_no and "whitelist" in reason
