import time

from hlbot.analysis.prompts import build_system_blocks
from hlbot.analysis.regime import format_recent_catalysts


def test_format_recent_catalysts_renders_age_and_reason():
    now = time.time() * 1000
    rows = [
        {"ts": now - 7 * 86400 * 1000, "symbol": "MSTR", "side": "short",
         "reason": "Saylor sold 32 BTC, first sale in years — regime change"},
        {"ts": now - 3 * 3600 * 1000, "symbol": "MRVL", "side": "long", "reason": "raised guidance"},
    ]
    out = format_recent_catalysts(rows)
    assert "MSTR short" in out and "7d ago" in out
    assert "MRVL long" in out and "3h ago" in out
    assert format_recent_catalysts([]) == ""


def test_system_blocks_include_catalyst_memory_uncached():
    mem = "- MSTR short · \"Saylor sold 32 BTC\" · 7d ago"
    blocks = build_system_blocks(["MRVL"], ["BTC"], regime_context="risk-on",
                                 recent_catalysts=mem)
    cat = [b for b in blocks if "RECENT CATALYSTS ALREADY TRADED" in b["text"]]
    assert len(cat) == 1
    assert mem in cat[0]["text"]
    # The volatile memory block must NOT carry cache_control (protects the cached prefix).
    assert "cache_control" not in cat[0]
    # No catalyst block when memory is empty.
    assert all("RECENT CATALYSTS" not in b["text"]
               for b in build_system_blocks(["MRVL"], ["BTC"], recent_catalysts=""))
