"""News-text normalization: source/channel name and appended translations must not
change what the analyzer sees, so the same wire from different aggregators is identical."""
from hlbot.analysis.prompts import build_system_blocks, build_user_text, clean_news_text
from hlbot.models import NewsItem

ENG = "COINBASE BECOMES OFFICIAL USDC TREASURY DEPLOYER ON HYPERLIQUID, ACQUIRES USDH BRAND ASSETS"
ZH = "Coinbase 成为 Hyperliquid 官方 USDC 财库部署者，收购 USDH 品牌资产"
# The two real feed formats observed in production.
AGGR_BODY = f"{ENG}\n\nLink"
BWE_BODY = f"AggrNews: {ENG}\n\nAggrNews: {ZH}\n\n————————————\n2026-05-14 20:00:47"


def test_clean_strips_translation_link_and_url():
    assert clean_news_text(ENG + "\n\nLink") == ENG          # boilerplate "Link" dropped
    assert clean_news_text(ENG + "\n\n" + ZH) == ENG          # Chinese translation dropped
    assert clean_news_text(ENG + "\nhttps://t.me/x/1") == ENG  # URL dropped


def test_clean_strips_attribution_separator_timestamp():
    # The real BWEnews wrapper (attribution prefix + translation + separator + timestamp)
    # must normalize to the exact same core wire as the plain AggrNewswire version.
    assert clean_news_text(AGGR_BODY) == ENG
    assert clean_news_text(BWE_BODY) == ENG
    assert clean_news_text(AGGR_BODY) == clean_news_text(BWE_BODY)


def test_build_user_text_identical_across_aggregators():
    i1 = NewsItem.from_telegram(channel="AggrNewswire", channel_title="aggrnews", msg_id=1,
                                text=AGGR_BODY, date_ms=1000, chat_id="x")
    i2 = NewsItem.from_telegram(channel="BWEnews", channel_title="BWEnews", msg_id=2,
                                text=BWE_BODY, date_ms=1000, chat_id="y")
    t1 = build_user_text(i1, age_seconds=10)
    t2 = build_user_text(i2, age_seconds=10)
    assert t1 == t2                       # same wire -> identical prompt -> identical confidence
    assert "SOURCE:" not in t1            # source/channel no longer exposed to the model
    assert "AggrNewswire" not in t1 and "BWEnews" not in t1
    assert ENG in t1


def test_exemplars_block_inserted_between_instructions_and_regime_cached():
    blocks = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on",
                                 recent_catalysts="- BTC short",
                                 exemplars='EXAMPLE:\nNEWS: "x"')
    texts = [b["text"] for b in blocks]
    i_ex = next(i for i, t in enumerate(texts) if "WORKED EXAMPLES" in t)
    i_reg = next(i for i, t in enumerate(texts) if "CURRENT MARKET / GEOPOLITICAL REGIME" in t)
    assert 0 < i_ex < i_reg                        # stability ordering: instructions, exemplars, regime
    assert "cache_control" in blocks[i_ex]         # part of the cached prefix
    assert "cache_control" not in blocks[-1]       # catalysts memory stays uncached last


def test_no_exemplars_means_prompt_unchanged():
    base = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on")
    with_empty = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on",
                                     exemplars="   ")
    assert base == with_empty


def test_context_bridge_block_between_exemplars_and_regime_cached():
    blocks = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on",
                                 recent_catalysts="- BTC short",
                                 exemplars='EXAMPLE:\nNEWS: "x"',
                                 context_bridge="- 2025-09: thing happened")
    texts = [b["text"] for b in blocks]
    i_ex = next(i for i, t in enumerate(texts) if "WORKED EXAMPLES" in t)
    i_br = next(i for i, t in enumerate(texts) if "BACKGROUND SINCE YOUR TRAINING" in t)
    i_reg = next(i for i, t in enumerate(texts) if "CURRENT MARKET / GEOPOLITICAL REGIME" in t)
    assert 0 < i_ex < i_br < i_reg                 # most- to least-stable ordering preserved
    assert "cache_control" in blocks[i_br]         # part of the cached prefix
    assert "- 2025-09: thing happened" in texts[i_br]


def test_no_context_bridge_means_prompt_unchanged():
    base = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on")
    with_empty = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on",
                                     context_bridge="  \n ")
    assert base == with_empty


def test_system_blocks_use_long_cache_ttl_by_default():
    # Sparse news cadence: 5m caches mostly expire unread, so stable blocks default to 1h.
    blocks = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on")
    cached = [b for b in blocks if "cache_control" in b]
    assert cached
    assert all(b["cache_control"] == {"type": "ephemeral", "ttl": "1h"} for b in cached)
    # "5m" maps to the API default (no ttl field).
    blocks = build_system_blocks(["MRVL"], ["BTC"], cache_ttl="5m")
    assert all(b["cache_control"] == {"type": "ephemeral"}
               for b in blocks if "cache_control" in b)
