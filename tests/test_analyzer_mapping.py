from hlbot.analysis.analyzer import Analyzer
from hlbot.models import NewsItem, now_ms


def _news():
    return NewsItem("n", "t", "b", "X", None, now_ms(), now_ms())


def test_to_analysis_clamps_and_normalizes():
    a = object.__new__(Analyzer)  # bypass __init__ (no network/client needed)
    raw = {"ticker": "$mrvl", "asset_class": "equity", "direction": "long",
           "confidence": 1.7, "time_sensitivity": "immediate", "is_stale": False,
           "rationale": "x" * 1200, "subject_relation": "direct"}
    res = Analyzer._to_analysis(a, _news(), raw, "model-x")
    assert res.ticker == "MRVL"
    assert res.confidence == 1.0
    assert res.direction == "long"
    assert len(res.rationale) <= 1000
    assert res.subject_relation == "direct"


def test_to_analysis_bad_values_default_to_none():
    a = object.__new__(Analyzer)
    raw = {"ticker": "", "direction": "sideways", "asset_class": "weird", "confidence": "abc"}
    res = Analyzer._to_analysis(a, _news(), raw, "m")
    assert res.ticker is None
    assert res.direction == "none"
    assert res.asset_class == "none"
    assert res.confidence == 0.0
    # missing/invalid subject_relation falls back to "derived" (regex backstop, prior behavior)
    assert res.subject_relation == "derived"


def test_to_analysis_discards_markup_garbage_ticker():
    # Seen live 2026-06-11: the model leaked tool-call markup INSIDE the ticker string.
    # Strict mode can't catch it (any string is schema-valid) — the mapper must.
    a = object.__new__(Analyzer)
    raw = {"ticker": '</ANTML_PARAMETER>\n<PARAMETER NAME="ASSET_CLASS">NONE',
           "direction": "long", "asset_class": "equity", "confidence": 0.9,
           "time_sensitivity": "days"}
    res = Analyzer._to_analysis(a, _news(), raw, "m")
    assert res.ticker is None                 # never reaches UI/DB as a "verdict"
    assert res.direction == "long"            # the rest of the verdict is preserved
    assert res.related_tickers == [] or all(
        r["ticker"].isalnum() or set(r["ticker"]) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        for r in res.related_tickers)


def test_to_analysis_ticker_shape_rules():
    a = object.__new__(Analyzer)
    for bad in ("TOOLONGTICKER", "AB CD", "NV\nDA", ".X", "nvda<"):
        res = Analyzer._to_analysis(a, _news(), {"ticker": bad, "direction": "none"}, "m")
        assert res.ticker is None, bad
    for ok, want in (("$nvda", "NVDA"), ("brk.b", "BRK.B"), ("xyz-1", "XYZ-1")):
        res = Analyzer._to_analysis(a, _news(), {"ticker": ok, "direction": "none"}, "m")
        assert res.ticker == want


def test_to_analysis_filters_garbage_related_and_sensitivity():
    a = object.__new__(Analyzer)
    raw = {"ticker": "NVDA", "direction": "long", "asset_class": "equity",
           "confidence": 0.85, "time_sensitivity": "<INVOKE>",
           "related_tickers": [{"ticker": "</X>", "asset_class": "equity"},
                               {"ticker": "AMD", "asset_class": "equity"}]}
    res = Analyzer._to_analysis(a, _news(), raw, "m")
    assert res.time_sensitivity == "none"
    assert [r["ticker"] for r in res.related_tickers] == ["NVDA", "AMD"]


def test_to_analysis_traded_ticker_always_leads_related():
    # The model may list the traded ticker AFTER a secondary; the traded one must still
    # lead the display list (and never be dropped by the [:2] cut).
    a = object.__new__(Analyzer)
    raw = {"ticker": "AMD", "direction": "long", "asset_class": "equity", "confidence": 0.85,
           "time_sensitivity": "days",
           "related_tickers": [{"ticker": "NVDA", "asset_class": "equity"},
                               {"ticker": "INTC", "asset_class": "equity"},
                               {"ticker": "AMD", "asset_class": "equity"}]}
    res = Analyzer._to_analysis(a, _news(), raw, "m")
    assert [r["ticker"] for r in res.related_tickers] == ["AMD", "NVDA"]
