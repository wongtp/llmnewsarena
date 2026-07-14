from hlbot.models import NewsItem, now_ms

GALA = {
    "_id": "1749403469352710423",
    "title": "Gala Games (@GoGalaGames)",
    "body": "Are you a Founder's Node operator?",
    "coin": "GALA",
    "link": "https://twitter.com/GoGalaGames/status/1",
    "time": 1705925264683,
    "rt": 1705925264813,
    "type": "direct",
    "suggestions": [{"coin": "GALA", "symbols": [{"exchange": "binance", "symbol": "GALAUSDT"}]}],
    "info": {"isRetweet": False, "isReply": False, "twitterId": "1288572182444961793"},
}


def test_from_tree_twitter():
    item = NewsItem.from_tree(GALA)
    assert item is not None
    assert item.id == "1749403469352710423"
    assert item.coin_hint == "GALA"
    assert item.symbol_hints == ["GALA"]
    assert item.author_id == "1288572182444961793"  # author account id, not tweet id
    assert not item.is_retweet


def test_from_tree_ignores_control_frames():
    assert NewsItem.from_tree({"type": "heartbeat"}) is None
    assert NewsItem.from_tree("login ok") is None
    assert NewsItem.from_tree({"_id": "x"}) is None  # no title/body


def test_age_seconds_fresh():
    item = NewsItem.from_tree({"_id": "1", "title": "t", "body": "b", "time": now_ms()})
    assert item.age_seconds < 2


def test_from_telegram():
    item = NewsItem.from_telegram(channel="tradfi", channel_title="TradFi News", msg_id=42,
                                  text="$MRVL Marvell signs huge AI deal", date_ms=now_ms(),
                                  chat_id=12345)
    assert item.id == "tg:tradfi:42"
    assert item.source == "Telegram:tradfi"
    assert item.link == "https://t.me/tradfi/42"
    assert "MRVL" in item.body
    # empty messages are ignored
    assert NewsItem.from_telegram(channel="x", channel_title="X", msg_id=1, text="  ",
                                  date_ms=now_ms(), chat_id=1) is None
