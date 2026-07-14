"""Skeptic entry confirmation: Analyzer.confirm parsing/fail-open, the market-context
block (incl. lookahead safety), and the risk-engine policy (veto / min / fail-open).
All offline — fake clients, same style as test_speed_cost."""
import asyncio
import types

import httpx
from anthropic import BadRequestError

from hlbot.analysis.analyzer import Analyzer
from hlbot.analysis.market_context import format_context
from hlbot.analysis.prompts import CONFIRM_TOOL, build_confirm_user_text
from hlbot.models import NewsItem


def _cfg(**over):
    base = dict(model_fast="claude-sonnet-4-6", model_smart="claude-opus-4-8",
                temperature=0.0, max_tokens=1024, max_retries=1, retry_base_delay=0.0,
                strict_tool=True, effort="", cache_ttl="1h", include_crypto_universe=True,
                triage_model="claude-haiku-4-5-20251001",
                confirm_entries=True, confirm_model="claude-opus-4-8",
                confirm_rule="min", confirm_gate=0.78, confirm_max_tokens=512)
    base.update(over)
    return types.SimpleNamespace(**base)


def _analyzer(create, **cfg_over):
    a = object.__new__(Analyzer)
    a.cfg = _cfg(**cfg_over)
    a.client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    a.regime_context = "risk-on"
    a.exemplars = ""
    a.context_bridge = ""
    a.recent_catalysts = ""
    a._strict_ok = True
    a._effort_ok = True
    a._warm_key = None
    a._prefix_used_at = 0.0
    a._record = lambda model, resp: None
    return a


def _news():
    return NewsItem(id="n1", title="t", body="Strategy sells BTC", source="Telegram:x",
                    link=None, time_ms=0, received_ms=0)


def _analysis():
    from hlbot.models import Analysis
    return Analysis(news_id="n1", ticker="BTC", asset_class="crypto", direction="short",
                    confidence=0.86, time_sensitivity="days", is_stale=False,
                    rationale="first sale ever", model="claude-sonnet-4-6")


def _verdict_resp(agree=True, conf=0.82, risk="could be priced in"):
    block = types.SimpleNamespace(type="tool_use", name="confirm_trade",
                                  input={"agree_direction": agree, "confidence": conf,
                                         "risk": risk})
    return types.SimpleNamespace(content=[block], usage=None)


# ---- Analyzer.confirm -------------------------------------------------------------


def test_confirm_parses_verdict_and_no_temperature_on_opus():
    calls = []

    async def create(**kw):
        calls.append(kw)
        return _verdict_resp(agree=False, conf=0.3, risk="already repriced")

    a = _analyzer(create)
    v = asyncio.run(a.confirm(_news(), _analysis(), market_context="MARKET TAPE ...",
                              pre_move_pct=0.02))
    assert v == {"agree_direction": False, "confidence": 0.3, "risk": "already repriced"}
    kw = calls[0]
    assert kw["model"] == "claude-opus-4-8"
    assert "temperature" not in kw                       # Opus 4.8 rejects sampling params
    assert kw["tool_choice"]["name"] == "confirm_trade"
    assert kw["tools"][0].get("strict") is True
    user = kw["messages"][0]["content"]
    assert "MARKET TAPE" in user and "PRE-NEWS MOVE" in user and "FIRST-PASS" in user


def test_confirm_fails_open_on_api_error():
    async def create(**kw):
        raise RuntimeError("api down")

    a = _analyzer(create)
    assert asyncio.run(a.confirm(_news(), _analysis())) is None


def test_confirm_strict_falls_back_on_400():
    calls = []

    async def create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise BadRequestError("strict rejected",
                                  response=httpx.Response(400, request=req), body=None)
        return _verdict_resp()

    a = _analyzer(create)
    v = asyncio.run(a.confirm(_news(), _analysis()))
    assert v is not None and v["agree_direction"] is True
    assert len(calls) == 2 and not calls[1]["tools"][0].get("strict")


def test_confirm_clamps_garbage_confidence():
    async def create(**kw):
        return _verdict_resp(conf=7.5)

    v = asyncio.run(_analyzer(create).confirm(_news(), _analysis()))
    assert v["confidence"] == 1.0


def test_confirm_tool_schema_strict_compatible():
    schema = CONFIRM_TOOL["input_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_confirm_user_text_without_optional_evidence():
    txt = build_confirm_user_text(_news(), _analysis())
    assert "MARKET TAPE" not in txt and "PRE-NEWS MOVE" not in txt
    assert "FIRST-PASS" in txt


# ---- market context ----------------------------------------------------------------


def _bar(t, o, h, l, c):
    return {"t": t, "T": t + 900_000, "o": o, "h": h, "l": l, "c": c}


class _Mkt:
    name = "BTC"
    day_volume_usd = 48_000_000.0


def test_format_context_basic_numbers():
    bars = [_bar(i * 900_000, 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(8)]
    at = 8 * 900_000
    txt = format_context(_Mkt(), bars, at)
    assert "MARKET TAPE for BTC" in txt
    assert "24h change" in txt and "24h range" in txt and "$48M" in txt


def test_format_context_excludes_in_progress_bar():
    # The bar straddling at_ms must NOT contribute (it would leak post-news price).
    done = [_bar(i * 900_000, 100, 101, 99, 100) for i in range(6)]
    leaking = _bar(6 * 900_000, 100, 150, 100, 150)   # huge post-news spike inside the bar
    at = 6 * 900_000 + 60_000                          # one minute into the last bar
    txt = format_context(_Mkt(), done + [leaking], at)
    assert "150" not in txt


def test_format_context_too_little_history_is_empty():
    assert format_context(_Mkt(), [_bar(0, 100, 101, 99, 100)], 10**12) == ""
    assert format_context(_Mkt(), [], 10**12) == ""


# ---- risk-engine policy (full evaluate path; fakes from test_risk.py) ---------------


from hlbot.config import Config  # noqa: E402
from hlbot.models import Analysis as _An  # noqa: E402
from hlbot.models import Market, now_ms  # noqa: E402
from hlbot.trading.risk import RiskEngine  # noqa: E402


class _FakeHL:
    async def mid(self, market):
        return 100.0
    # no .candles / .l2_book: pre-move ref + tape + book guard all fail OPEN


class _FakeStore:
    async def realized_pnl_today(self, dry_run, model_id=None):
        return 0.0


class _FakePM:
    def has_open(self, name):
        return False

    def open_count(self):
        return 0

    def total_exposure(self):
        return 0.0

    def unrealized_pnl(self, dry_run):
        return 0.0


def _eng(verdict, rule="min"):
    c = Config()
    c.runtime.dry_run = True
    c.app.analyzer.confirm_entries = True
    c.app.analyzer.confirm_rule = rule
    calls = []

    async def confirmer(item, analysis, market_context="", pre_move_pct=None):
        calls.append((market_context, pre_move_pct))
        return verdict

    eng = RiskEngine(c, _FakeHL(), _FakeStore(), _FakePM(), confirmer=confirmer)
    return eng, calls


def _mrvl():
    return Market("xyz:MRVL", "MRVL", "xyz", "equity", 2, 5)


def _item():
    return NewsItem("n1", "t", "b", "X", None, now_ms(), now_ms())


def _first_pass(conf=0.95):
    return _An("n1", "MRVL", "equity", "long", conf, "immediate", False, "x", "m",
               subject_relation="direct")


def test_evaluate_confirmation_veto_rejects():
    eng, _ = _eng({"agree_direction": False, "confidence": 0.9, "risk": "wrong ticker"})
    d = asyncio.run(eng.evaluate(_item(), _first_pass(), _mrvl()))
    assert d.action == "reject" and "confirmation veto" in d.reason


def test_evaluate_min_rule_resizes_and_gates():
    # Skeptic at 0.83: min(0.95, 0.83) clears the 0.78 confirm gate -> enter, sized off the
    # 0.83 tier from config, NOT off the first pass's 0.95 (max tier). Tier values are read
    # from config.yaml so live sizing changes don't silently break this test's intent.
    eng, calls = _eng({"agree_direction": True, "confidence": 0.83, "risk": "ok"})
    tier_083 = max(n for c, n in eng.r.size_tiers if 0.83 >= c)
    tier_095 = max(n for c, n in eng.r.size_tiers if 0.95 >= c)
    assert tier_083 < tier_095          # meaningless if the tiers coincide
    d = asyncio.run(eng.evaluate(_item(), _first_pass(0.95), _mrvl()))
    assert d.action == "enter" and abs(d.notional_usd - tier_083) < 1e-6
    assert calls   # confirmer actually consulted
    # Skeptic at 0.5: min falls below the confirm gate -> reject.
    eng, _ = _eng({"agree_direction": True, "confidence": 0.50, "risk": "weak"})
    d = asyncio.run(eng.evaluate(_item(), _first_pass(0.95), _mrvl()))
    assert d.action == "reject" and "confirm gate" in d.reason


def test_evaluate_veto_only_rule_keeps_fast_confidence():
    eng, _ = _eng({"agree_direction": True, "confidence": 0.10, "risk": "meh"},
                  rule="veto_only")
    d = asyncio.run(eng.evaluate(_item(), _first_pass(0.95), _mrvl()))
    assert d.action == "enter"
    assert d.notional_usd >= 10000.0 - 1e-6   # sized off the untouched 0.95


def test_evaluate_fails_open_when_confirmer_unavailable():
    eng, _ = _eng(None)   # confirmer returns None (API error path)
    d = asyncio.run(eng.evaluate(_item(), _first_pass(), _mrvl()))
    assert d.action == "enter"


def test_evaluate_confirm_off_never_calls_confirmer():
    eng, calls = _eng({"agree_direction": False, "confidence": 0.0, "risk": "x"})
    eng.config.app.analyzer.confirm_entries = False
    d = asyncio.run(eng.evaluate(_item(), _first_pass(), _mrvl()))
    assert d.action == "enter" and not calls
