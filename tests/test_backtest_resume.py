"""Resume safety for the backtest analysis cache: a failed call (out-of-credits /
rate-limit-exhausted / dropped connection / truncated thinking model) must NOT be cached as a
false 'none' — otherwise a resume skips it forever, poisoning the window. Failures stay
uncached (re-run on resume); a pile-up trips the consecutive-failure breaker. All offline."""
import asyncio
import types

import pytest

from hlbot.backtest.engine import AnalysisCache, analyze_all
from hlbot.models import Analysis, NewsItem, now_ms


def _items(n):
    return [NewsItem(f"id{i}", "t", "b", "X", None, now_ms(), now_ms()) for i in range(n)]


class _FakeAnalyzer:
    """Returns a good verdict, or an error analysis (mirrors Analyzer.analyze's except path),
    per a plan keyed by news id. Records which ids it was actually asked to analyze."""

    def __init__(self, plan: dict):
        self.plan = plan          # id -> "ok" | "fail"
        self.calls: list[str] = []

    async def analyze(self, it, universe, age_seconds=None, model=None):
        self.calls.append(it.id)
        if self.plan.get(it.id, "ok") == "fail":
            return Analysis(news_id=it.id, ticker=None, asset_class="none", direction="none",
                            confidence=0.0, time_sensitivity="none", is_stale=True,
                            rationale="analyzer error", model=model or "m", error="boom 429")
        return Analysis(news_id=it.id, ticker="BTC", asset_class="crypto", direction="long",
                        confidence=0.85, time_sensitivity="immediate", is_stale=False,
                        rationale="ok", model=model or "m")


def test_failed_analyses_are_not_cached(tmp_path):
    cache = AnalysisCache(str(tmp_path / "c.json"))
    plan = {"id0": "ok", "id1": "fail", "id2": "ok", "id3": "fail", "id4": "ok"}
    az = _FakeAnalyzer(plan)
    out = asyncio.run(analyze_all(az, None, _items(5), cache, concurrency=1))
    assert set(out) == {f"id{i}" for i in range(5)}          # every item is in the result map
    assert set(cache.data) == {"id0", "id2", "id4"}          # but only the GOOD ones are cached
    # persisted to disk too (so a crash keeps the goods)
    reloaded = AnalysisCache(str(tmp_path / "c.json"))
    assert set(reloaded.data) == {"id0", "id2", "id4"}


def test_resume_reruns_only_failures(tmp_path):
    path = str(tmp_path / "c.json")
    plan = {"id0": "ok", "id1": "fail", "id2": "ok", "id3": "fail", "id4": "ok"}
    asyncio.run(analyze_all(_FakeAnalyzer(plan), None, _items(5), AnalysisCache(path), concurrency=1))
    # Resume: failures now succeed; cached goods must be SKIPPED (not re-analyzed).
    az2 = _FakeAnalyzer({"id1": "ok", "id3": "ok"})
    out = asyncio.run(analyze_all(az2, None, _items(5), AnalysisCache(path), concurrency=1))
    assert sorted(az2.calls) == ["id1", "id3"]               # only the previously-failed re-run
    assert all(out[i].direction == "long" for i in ("id0", "id1", "id2", "id3", "id4"))
    assert set(AnalysisCache(path).data) == {f"id{i}" for i in range(5)}   # now fully cached


def test_consecutive_failure_breaker_aborts_and_preserves_progress(tmp_path):
    path = str(tmp_path / "c.json")
    # 2 goods, then a long run of failures (out-of-credits etc.): the breaker must abort the run
    # and leave only the goods cached for a clean resume — never a failure as a false 'none'.
    # (Precise early-stop — no churn — is guaranteed on the SEQUENTIAL run_live_replay path the
    # arena uses; under analyze_all's concurrent gather the abort still fires and progress is safe.)
    plan = {"id0": "ok", "id1": "ok"}
    plan.update({f"id{i}": "fail" for i in range(2, 60)})
    az = _FakeAnalyzer(plan)
    with pytest.raises(RuntimeError, match="consecutive analysis failures"):
        asyncio.run(analyze_all(az, None, _items(60), AnalysisCache(path), concurrency=1))
    assert set(AnalysisCache(path).data) == {"id0", "id1"}   # goods saved; no failure cached
