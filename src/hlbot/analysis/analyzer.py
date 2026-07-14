"""Claude-based news analyzer: news item -> structured (ticker, direction, confidence).

Uses tool-use for a guaranteed-shaped result and prompt caching on the static
instructions + tradable universe so repeated calls are cheap. A fast model handles
the common case; borderline calls can escalate to a stronger model.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Optional
from collections import defaultdict

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from ..config import Config
from ..models import Analysis, NewsItem
from . import usage_ledger
from .pricing import response_cost_usd
from .providers import ProviderResponseError, ProviderRouter, is_anthropic
from .prompts import (
    ANALYSIS_TOOL,
    CONFIRM_INSTRUCTIONS,
    CONFIRM_TOOL,
    build_confirm_user_text,
    build_system_blocks,
    build_user_text,
    supports_temperature,
)
from .universe import Universe

log = logging.getLogger("hlbot.analyzer")

_VALID_DIR = {"long", "short", "none"}
_VALID_CLASS = {"equity", "index", "commodity", "crypto", "none"}
_VALID_RELATION = {"direct", "derived"}
_VALID_SENSITIVITY = {"immediate", "hours", "days", "none"}
# Plausible ticker shape. Strict tool mode guarantees the FIELD is a string, not that the
# string is sane — the model has leaked tool-call markup inside the ticker value (seen
# live 2026-06-11). A garbage ticker can only cause a MISSED trade (universe.resolve
# fails closed), but it must not reach the UI/DB as a "verdict"; null it loudly instead.
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$")

# Transient API failures worth retrying (vs. e.g. a 400 bad-request, which won't fix itself).
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", 0) or 0
        return code in (408, 409, 429) or code >= 500
    return False


class Analyzer:
    def __init__(self, config: Config, ledger_path: str | None = None):
        self.config = config
        self.cfg = config.app.analyzer
        # max_retries=0: we own the retry loop (see _create) so attempts are logged + tunable.
        # Tight timeout: the SDK default (10 min) would stall an item far past staleness.
        self.client = AsyncAnthropic(api_key=config.secrets.anthropic_api_key, max_retries=0,
                                     timeout=self.cfg.request_timeout_seconds)
        # Arena: non-Anthropic models (GPT/Gemini/DeepSeek/GLM/Grok, prefixed e.g.
        # "openai:gpt-5.4") route through provider adapters; the Claude path above stays native.
        # Bare strings and "claude-*" never touch the router (is_anthropic -> True).
        self.router = ProviderRouter(config.secrets, self.cfg)
        self.regime_context = self._load_regime(config.app.regime_context_file)
        # Hand-curated few-shot calibration anchors (data/exemplars.md; empty/absent = no
        # block, prior prompt unchanged). Curate with scripts/mine_exemplars.py.
        self.exemplars = self._load_regime(config.app.exemplars_file)
        # Knowledge bridge: dated post-cutoff facts (data/context_bridge.md; empty/absent =
        # no block). Grounds staleness/novelty calls the 14d regime brief can't cover.
        self.context_bridge = self._load_regime(config.app.context_bridge_file)
        self.recent_catalysts = ""   # anti-re-trade memory; refreshed from the store while running
        self.ledger_path = ledger_path   # set (live) -> persist all-time token usage; None (backtest) -> off
        # Per-model token accounting (input/output/cache) for cost analysis.
        self.usage: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_creation": 0})
        # Optional request features; flipped off for the session if the API rejects them
        # (one logged retry) so a 400 can never blind the bot. See _without_optional_features.
        self._strict_ok = True
        self._effort_ok = True
        # Prompt-cache keep-warm state: fingerprint of the last-warmed prefix, and the
        # monotonic time the model_fast prefix last hit the cache (real call or warm).
        self._warm_key: int | None = None
        self._prefix_used_at = 0.0

    def _record(self, model: str, resp, *, infra: bool = False) -> float:
        """Account one response's tokens; returns its estimated $ cost so callers can
        attribute spend to the analysis that incurred it. infra=True marks plumbing calls
        (cache keep-warm; regime briefs tag theirs in regime.py) — they persist under a
        "<model>#infra" ledger key so per-model spend comparisons stay analysis-only
        (pricing strips the suffix)."""
        u, usage = self.usage[model], getattr(resp, "usage", None)
        if not usage:
            return 0.0
        u["calls"] += 1
        u["input"] += getattr(usage, "input_tokens", 0) or 0
        u["output"] += getattr(usage, "output_tokens", 0) or 0
        u["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        u["cache_creation"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        usage_ledger.record(self.ledger_path, f"{model}#infra" if infra else model,
                            resp)   # persist all-time (live only)
        return response_cost_usd(model, usage)

    def usage_summary(self) -> dict:
        return {m: dict(v) for m, v in self.usage.items()}

    @staticmethod
    def _load_regime(path: str) -> str:
        import pathlib
        p = pathlib.Path(path)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                return ""
        return ""

    def _none(self, item: NewsItem, reason: str) -> Analysis:
        return Analysis(item.id, None, "none", "none", 0.0, "none", False, reason, "triage")

    async def _create(self, **kwargs):
        """messages.create with bounded exponential backoff on transient errors. A breaking
        catalyst can't be re-fetched, so don't lose the trade to a 429/timeout/5xx blip; a
        non-retryable error (e.g. 400) raises immediately."""
        attempts = max(1, self.cfg.max_retries)
        for i in range(1, attempts + 1):
            try:
                return await self.client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised below unless retryable
                if i >= attempts or not _is_retryable(exc):
                    raise
                delay = self.cfg.retry_base_delay * (2 ** (i - 1)) * (0.5 + random.random())
                log.warning("Claude call failed (%s); retry %d/%d in %.1fs",
                            type(exc).__name__, i, attempts - 1, delay)
                await asyncio.sleep(delay)

    async def _triage(self, item: NewsItem, cost: list) -> bool:
        """Cheap Haiku relevance gate: is this a fresh, actionable catalyst at all?"""
        try:
            kwargs: dict = dict(
                model=self.cfg.triage_model, max_tokens=5,
                system=(
                    "You are a fast filter for a trading bot. Decide if the news is a FRESH, "
                    "specific, market-moving CATALYST for a tradable company, crypto, or "
                    "commodity. NOT a catalyst: opinions/analysis, recaps, already-happened "
                    "price moves (results), routine or pre-announced/expected updates, generic "
                    "macro chatter, back-and-forth war headlines, or exchange listing chatter. "
                    "Answer with ONLY 'yes' or 'no'."),
                messages=[{"role": "user", "content": item.text[:800]}],
            )
            if supports_temperature(self.cfg.triage_model):
                kwargs["temperature"] = 0.0
            resp = await self._create(**kwargs)
            cost[0] += self._record(self.cfg.triage_model, resp)
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text").strip().lower()
            # Empty/garbled reply fails OPEN like the exception path — never silently drop
            # a catalyst because the filter produced nothing.
            return text.startswith("y") if text else True
        except Exception:  # noqa: BLE001 - on triage error, don't drop; fall through to full analysis
            return True

    async def analyze(self, item: NewsItem, universe: Universe,
                      age_seconds: float | None = None, model: str | None = None) -> Analysis:
        # Per-analysis observability (surfaced in the UI next to the verdict): wall-clock
        # latency of THIS item's analysis chain and the $ cost of every call it made
        # (triage + main + escalation). A list so the nested calls can accumulate into it.
        t0 = time.perf_counter()
        cost = [0.0]

        def done(a: Analysis) -> Analysis:
            a.latency_ms = int((time.perf_counter() - t0) * 1000)
            a.cost_usd = round(cost[0], 6)
            return a

        # Tier 0: free regex prefilter — skip anything not naming a tradable symbol/alias.
        if self.cfg.use_prefilter and not universe.matches(item.text):
            return done(self._none(item, "no tradable entity mentioned"))
        # Tier 1: cheap Haiku relevance gate.
        if self.cfg.use_triage and not await self._triage(item, cost):
            return done(self._none(item, "triage: not an actionable catalyst"))

        # Tier 2: full structured analysis with the primary model.
        crypto = universe.crypto_symbols() if self.cfg.include_crypto_universe else []
        system = build_system_blocks(universe.equity_symbols(), crypto, self.regime_context,
                                     self.recent_catalysts, cache_ttl=self.cfg.cache_ttl,
                                     exemplars=self.exemplars, context_bridge=self.context_bridge)
        user = build_user_text(item, age_seconds)
        use_model = model or self.cfg.model_fast
        try:
            raw = await self._call(use_model, system, user, cost)
            analysis = self._to_analysis(item, raw, use_model)
        except Exception as exc:  # noqa: BLE001
            log.exception("Analyzer call failed for %s", item.id)
            return done(Analysis(item.id, None, "none", "none", 0.0, "none", True,
                                 "analyzer error", use_model, error=str(exc)))

        # Escalate borderline, actionable calls to the stronger model (default mode only).
        thr = self.config.app.risk.confidence_threshold
        margin = self.cfg.escalate_margin
        if (model is None and margin > 0 and analysis.direction != "none"
                and not analysis.is_stale and thr - margin <= analysis.confidence < thr):
            try:
                raw2 = await self._call(self.cfg.model_smart, system, user, cost)
                analysis = self._to_analysis(item, raw2, self.cfg.model_smart)
            except Exception:  # noqa: BLE001
                log.warning("Escalation to %s failed; keeping fast result", self.cfg.model_smart)
        return done(analysis)

    async def confirm(self, item: NewsItem, analysis: Analysis, market_context: str = "",
                      pre_move_pct: float | None = None,
                      age_seconds: float | None = None) -> dict | None:
        """Skeptic second opinion on a would-be ENTRY (post-gate, pre-sizing): a stronger
        model argues against the trade with evidence the first pass never saw (the tape +
        the measured pre-news move), then returns an independent verdict
        {agree_direction, confidence, risk}. Returns None on ANY error — the caller must
        fail OPEN: a 429 must not kill a validated catalyst trade. Volume is a few calls
        per day, so the latency (3-8s) and cost (cents) are noise against the notional."""
        model = self.cfg.confirm_model
        system: list[dict] = [{"type": "text", "text": CONFIRM_INSTRUCTIONS,
                               "cache_control": {"type": "ephemeral", "ttl": "1h"}
                               if self.cfg.cache_ttl != "5m" else {"type": "ephemeral"}}]
        if self.regime_context.strip():
            system.append({"type": "text",
                           "text": "CURRENT MARKET / GEOPOLITICAL REGIME:\n"
                                   + self.regime_context.strip()})
        tools = ([{**CONFIRM_TOOL, "strict": True}]
                 if (self.cfg.strict_tool and self._strict_ok) else [CONFIRM_TOOL])
        kwargs: dict = dict(
            model=model, max_tokens=self.cfg.confirm_max_tokens, system=system, tools=tools,
            tool_choice={"type": "tool", "name": "confirm_trade"},
            messages=[{"role": "user", "content": build_confirm_user_text(
                item, analysis, market_context, pre_move_pct, age_seconds=age_seconds)}],
        )
        if supports_temperature(model):
            kwargs["temperature"] = self.cfg.temperature
        try:
            try:
                resp = await self._create(**kwargs)
            except BadRequestError as exc:
                retry = self._without_optional_features(kwargs, exc)
                if retry is None:
                    raise
                resp = await self._create(**retry)
        except Exception:  # noqa: BLE001 - caller fails OPEN; log loudly
            log.exception("Entry confirmation call failed for %s (failing open)", item.id)
            return None
        self._record(model, resp)
        for block in resp.content:
            if block.type == "tool_use" and block.name == "confirm_trade":
                raw = dict(block.input)
                try:
                    conf = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
                except (TypeError, ValueError):
                    conf = 0.0
                return {"agree_direction": bool(raw.get("agree_direction", True)),
                        "confidence": conf, "risk": str(raw.get("risk") or "")[:300]}
        log.warning("Entry confirmation returned no verdict for %s (failing open)", item.id)
        return None

    def _tools(self) -> list[dict]:
        """The analysis tool, with API-side strict schema validation when enabled. Shared by
        real calls AND the cache warmer — the tool bytes are part of the cached prefix, so
        both must send the identical definition."""
        if self.cfg.strict_tool and self._strict_ok:
            return [{**ANALYSIS_TOOL, "strict": True}]
        return [ANALYSIS_TOOL]

    def _extra_body(self, model: str) -> dict | None:
        """output_config.effort when configured — sent via extra_body so older SDK pins still
        work. Haiku models don't support effort, so it is never attached there."""
        if self.cfg.effort and self._effort_ok and "haiku" not in (model or "").lower():
            return {"output_config": {"effort": self.cfg.effort}}
        return None

    def _without_optional_features(self, kwargs: dict, exc: Exception) -> dict | None:
        """A 400 may be caused by the OPTIONAL request features (strict tool validation,
        output_config.effort) on an API/account that rejects them. Strip whichever were sent,
        disable them for the session (logged), and let the caller retry once. Returns None if
        none were sent — the 400 is then genuine and must propagate."""
        stripped = []
        out = dict(kwargs)
        if any(t.get("strict") for t in out.get("tools", [])):
            out["tools"] = [{k: v for k, v in t.items() if k != "strict"} for t in out["tools"]]
            self._strict_ok = False
            stripped.append("strict tool validation")
        if out.pop("extra_body", None) is not None:
            self._effort_ok = False
            stripped.append(f"effort={self.cfg.effort!r}")
        if not stripped:
            return None
        log.warning("Claude 400 (%s) — retrying without %s (disabled for this session)",
                    exc, " + ".join(stripped))
        return out

    async def _call_provider(self, model: str, system: list[dict], user_text: str,
                             cost: list | None = None) -> dict:
        """Route a non-Anthropic arena model through its provider adapter. Same prompt (the
        adapter flattens the system blocks) and the same _to_analysis mapping as the native
        path — only the API call differs. Cost/usage are accounted under the full prefixed model
        string (e.g. "openai:gpt-5.4"), which is also what pricing.PRICING is keyed by."""
        provider, bare = self.router.provider_for(model)
        temperature = self.cfg.temperature if provider.supports_temperature(bare) else None
        result = await provider.call_tool(
            model=bare, system_blocks=system, user_text=user_text,
            tool=ANALYSIS_TOOL, tool_name="submit_analysis",
            max_tokens=self.cfg.max_tokens, temperature=temperature)
        c = self._record(model, result)
        if cost is not None:
            cost[0] += c
        return result.tool_input

    async def _call(self, model: str, system: list[dict], user_text: str,
                    cost: list | None = None) -> dict:
        if not is_anthropic(model):
            return await self._call_provider(model, system, user_text, cost)
        kwargs: dict = dict(
            model=model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            tools=self._tools(),
            tool_choice={"type": "tool", "name": "submit_analysis"},
            messages=[{"role": "user", "content": user_text}],
        )
        if supports_temperature(model):   # sampling params 400 on Opus 4.7+/Fable
            kwargs["temperature"] = self.cfg.temperature
        extra = self._extra_body(model)
        if extra:
            kwargs["extra_body"] = extra
        try:
            resp = await self._create(**kwargs)
        except BadRequestError as exc:
            retry = self._without_optional_features(kwargs, exc)
            if retry is None:
                raise
            resp = await self._create(**retry)
        if model == self.cfg.model_fast:
            self._prefix_used_at = time.monotonic()   # a real call refreshed the cache TTL
        c = self._record(model, resp)
        if cost is not None:
            cost[0] += c
        for block in resp.content:
            if block.type == "tool_use" and block.name == "submit_analysis":
                return dict(block.input)
        # No tool block (e.g. max_tokens truncation). Raise — same contract as the provider
        # adapters — so this becomes an error-stamped verdict the backtest will NOT cache,
        # instead of a legit-looking none/0.0 cached forever.
        raise ProviderResponseError(f"{model}: response contained no submit_analysis tool call")

    # ----- prompt-cache keep-warm ------------------------------------------------ #
    def _prefix_key(self, universe: Universe) -> int:
        """Fingerprint of everything that renders into the CACHED prefix (tools + system
        blocks up to the last cache breakpoint). recent_catalysts is excluded on purpose —
        it is appended uncached after the last breakpoint and never invalidates the prefix."""
        eq = universe.equity_symbols()
        cr = universe.crypto_symbols() if self.cfg.include_crypto_universe else []
        strict = self.cfg.strict_tool and self._strict_ok
        return hash((self.cfg.model_fast, self.cfg.cache_ttl, strict, self.exemplars,
                     self.context_bridge, self.regime_context, tuple(eq), tuple(cr)))

    async def maybe_warm_cache(self, universe: Universe, max_idle_seconds: float) -> bool:
        """Keep the cached prefix HOT so a headline after a quiet stretch never pays the
        cold-cache penalty (full-price prefill + 2x cache rewrite + extra latency) on the
        news->order critical path. A cache read refreshes the TTL, so re-warming just under
        it keeps the prefix alive at ~0.1x input price; when the prefix CHANGED (regime /
        universe refresh) the warm pre-pays the 2x write off the critical path. max_tokens=0
        is the API's prewarm idiom (prefill only, no output billed); falls back to 1 token
        if rejected. Returns True if a warm call was made."""
        key = self._prefix_key(universe)
        if key == self._warm_key and (time.monotonic() - self._prefix_used_at) < max_idle_seconds:
            return False
        crypto = universe.crypto_symbols() if self.cfg.include_crypto_universe else []
        system = build_system_blocks(universe.equity_symbols(), crypto, self.regime_context,
                                     "", cache_ttl=self.cfg.cache_ttl,
                                     exemplars=self.exemplars, context_bridge=self.context_bridge)
        # Same model + tools + system as the real calls (byte-identical prefix) but NO forced
        # tool_choice (rejected with max_tokens=0; tool_choice differences don't invalidate
        # the tools/system cache) and no temperature/effort (output is discarded anyway).
        kwargs: dict = dict(model=self.cfg.model_fast, system=system, tools=self._tools(),
                            messages=[{"role": "user", "content": "warmup"}])
        try:
            resp = await self._create(max_tokens=0, **kwargs)
        except BadRequestError:
            try:
                resp = await self._create(max_tokens=1, **kwargs)   # maybe only max_tokens=0 rejected
            except BadRequestError as exc:
                # Still 400 at max_tokens=1: the strict tool is the likely culprit — strip the
                # optional features exactly like the real-call path does, or the keep-warm loop
                # would fail every cycle and the cache stays cold until the first real call.
                retry = self._without_optional_features(kwargs, exc)
                if retry is None:
                    raise
                resp = await self._create(max_tokens=1, **retry)
        self._record(self.cfg.model_fast, resp, infra=True)
        self._warm_key = key
        self._prefix_used_at = time.monotonic()
        u = getattr(resp, "usage", None)
        log.info("Prompt cache warmed: read %s / wrote %s prefix tokens",
                 getattr(u, "cache_read_input_tokens", "?"),
                 getattr(u, "cache_creation_input_tokens", "?"))
        return True

    @staticmethod
    def _clean_ticker(value, news_id: str, field: str = "ticker") -> Optional[str]:
        """Uppercased bare ticker, or None if the value doesn't look like one (e.g. the
        model leaked markup into the string — log loudly, fail to 'no ticker')."""
        t = str(value or "").strip().lstrip("$").upper()
        if not t:
            return None
        if not _TICKER_RE.match(t):
            log.warning("Discarding malformed %s for %s: %r", field, news_id, t[:80])
            return None
        return t

    def _to_analysis(self, item: NewsItem, raw: dict, model: str) -> Analysis:
        ticker = self._clean_ticker(raw.get("ticker"), item.id)
        direction = raw.get("direction") if raw.get("direction") in _VALID_DIR else "none"
        asset_class = raw.get("asset_class") if raw.get("asset_class") in _VALID_CLASS else "none"
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        relation = raw.get("subject_relation")
        relation = relation if relation in _VALID_RELATION else "derived"
        related: list[dict] = []
        seen: set[str] = set()
        for r in raw.get("related_tickers") or []:
            if not isinstance(r, dict):
                continue
            t = self._clean_ticker(r.get("ticker"), item.id, field="related_ticker")
            if not t or t in seen:
                continue
            seen.add(t)
            kls = r.get("asset_class")
            related.append({"ticker": t,
                            "asset_class": kls if kls in _VALID_CLASS else "none"})
        # The traded ticker always leads the display list, whatever the model returned —
        # including when the model listed it after a secondary (dedup would otherwise leave
        # the secondary in front, and the [:2] cut could drop the traded ticker entirely).
        if ticker:
            related = [r for r in related if r["ticker"] != ticker]
            related.insert(0, {"ticker": ticker, "asset_class": asset_class})
        return Analysis(
            news_id=item.id,
            ticker=ticker,
            asset_class=asset_class,
            direction=direction,
            confidence=confidence,
            time_sensitivity=(raw.get("time_sensitivity")
                              if raw.get("time_sensitivity") in _VALID_SENSITIVITY else "none"),
            is_stale=bool(raw.get("is_stale", False)),
            rationale=(raw.get("rationale") or "")[:1000],
            model=model,
            subject_relation=relation,
            related_tickers=related[:2],   # primary + at most 1 secondary (latency: output tokens are serial)
        )
