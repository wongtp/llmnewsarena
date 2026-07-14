import asyncio

from hlbot.analysis.universe import Universe

# First-seen persistence is isolated per-test by the autouse conftest fixture.


class FakeHL:
    async def meta(self, dex=""):
        if dex == "xyz":
            return {"universe": [
                {"name": "xyz:MRVL", "szDecimals": 2, "maxLeverage": 5},
                {"name": "xyz:NVDA", "szDecimals": 2, "maxLeverage": 5},
            ]}
        return {"universe": [
            {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
            {"name": "ETH", "szDecimals": 4, "maxLeverage": 50},
        ]}


def _universe():
    u = Universe(FakeHL(), ["xyz", ""])
    asyncio.run(u.refresh())
    return u


def test_resolve_equity_and_crypto():
    u = _universe()
    mrvl = u.resolve("MRVL", "equity")
    assert mrvl and mrvl.name == "xyz:MRVL" and mrvl.dex == "xyz" and mrvl.sz_decimals == 2
    btc = u.resolve("BTC", "crypto")
    assert btc and btc.name == "BTC" and btc.dex == ""
    assert u.resolve("$mrvl") is not None       # strips $ and case-insensitive
    assert u.resolve("NOPE") is None
    assert "MRVL" in u.equity_symbols()
    assert "BTC" in u.crypto_symbols()
    # ticker collision: BTC exists only as crypto -> asking for it as an equity = no trade
    assert u.resolve("BTC", "equity") is None
    assert u.resolve("MRVL", "crypto") is None


def test_alias_resolution():
    u = _universe()
    assert u.resolve("nvidia", "equity").symbol == "NVDA"
    assert u.resolve("bitcoin", "crypto").symbol == "BTC"


def test_mentions_direct_symbol_cashtag_and_name():
    m = Universe.mentions
    assert m("NVDA", "NVIDIA CEO sees huge demand $NVDA")     # symbol + cashtag + name
    assert m("NVDA", "Nvidia raises guidance")               # company name alias
    assert m("MRVL", "Jensen calls Marvell the next big thing $MRVL")
    assert m("BTC", "Bitcoin breaks 100k")                   # name == alias
    assert m("AAVE", "Aave launches v4")                     # ticker doubles as the name
    # pre-IPO name != ticker: "SpaceX" must count as a DIRECT mention of SPCX, not indirect
    assert m("SPCX", "SPACEX IPO HAS DRAWN MORE THAN $250B OF INVESTOR DEMAND")
    assert m("SPCX", "Space X targets Q1 listing")


def test_mentions_indirect_is_false():
    m = Universe.mentions
    # 2nd-order: Alphabet capex headline, resolved to NVDA, which is NOT named -> False
    assert not m("NVDA", "ALPHABET SEES 2026 CAPEX $175B TO $185B $GOOGL")
    assert not m("NVDA", "stocksnvda inside a word")         # word-boundary, not substring


def test_mentions_uses_feed_hints_and_unknown_symbol():
    m = Universe.mentions
    assert m("SOL", "huge upgrade announced", hints=["SOL"])  # auto-detected by the feed
    assert m("", "anything")                                  # unknown symbol -> never penalize


def test_mentions_ticker_variants_equivalent():
    m = Universe.mentions
    # GOOG == GOOGL == "Google"/"Alphabet" — folded to one group, both directions:
    assert m("GOOGL", "shares pop on news $GOOG")             # variant cashtag counts
    assert m("GOOG", "Alphabet raises 2026 capex")           # name counts for the variant
    assert m("GOOGL", "Google unveils new TPU")
    assert m("GOOG", "$GOOGL up 3%")
