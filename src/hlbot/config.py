"""Configuration: secrets from .env, tunables from config.yaml, mutable runtime state."""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Loaded from environment / .env. Never logged."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    hl_account_address: str = ""
    hl_secret_key: str = ""
    hl_network: str = "mainnet"  # mainnet | testnet
    tree_api_key: str = ""
    anthropic_api_key: str = ""
    # Arena multi-LLM providers (native SDKs). Optional — only required when a model routed to
    # that provider is actually invoked (the router raises ProviderConfigError otherwise, which
    # the analyzer turns into a 'none' verdict). zhipu = GLM, xai = Grok.
    openai_api_key: str = ""
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    xai_api_key: str = ""
    zhipu_api_key: str = ""
    # Arena per-model wallets (LIVE only; paper/dry-run uses virtual capital). Funded after the
    # paper week, then flipped one at a time. ARENA_<KEY>_ADDRESS / ARENA_<KEY>_SECRET per entrant.
    arena_sonnet_address: str = ""
    arena_sonnet_secret: str = ""
    arena_gpt_address: str = ""
    arena_gpt_secret: str = ""
    arena_gemini_address: str = ""
    arena_gemini_secret: str = ""
    arena_deepseek_address: str = ""
    arena_deepseek_secret: str = ""
    arena_grok_address: str = ""
    arena_grok_secret: str = ""
    twelvedata_api_key: str = ""    # scripts/earnings_bench.py minute-candle source (free tier)
    alpaca_api_key: str = ""        # scripts/earnings_bench.py minute candles incl. after-hours
    alpaca_api_secret: str = ""     # (IEX feed, free paper account)
    telegram_bot_token: str = ""    # bot for sending alerts to you
    telegram_chat_id: str = ""
    telegram_api_id: str = ""       # MTProto user client (reading source channels)
    telegram_api_hash: str = ""

    def arena_wallet(self, key: str) -> tuple[str, str]:
        """(address, secret) for an arena entrant's funded wallet, or ('','') if unfunded
        (paper). Keyed by the entrant's `wallet` prefix (e.g. 'sonnet' -> ARENA_SONNET_*)."""
        return (getattr(self, f"arena_{key}_address", "") or "",
                getattr(self, f"arena_{key}_secret", "") or "")

    def missing(self, require_tree: bool = True) -> list[str]:
        """Required secrets that are absent or placeholders. TREE_API_KEY is only required
        when the Tree feed is actually enabled (pass require_tree=False for a
        Telegram-channels-only setup, i.e. enable_tree_feed: false)."""
        required = {
            "HL_ACCOUNT_ADDRESS": self.hl_account_address,
            "HL_SECRET_KEY": self.hl_secret_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
        }
        if require_tree:
            required["TREE_API_KEY"] = self.tree_api_key
        return [k for k, v in required.items() if not v or v.startswith("0xYOUR") or v.startswith("your")]


class RiskConfig(BaseModel):
    confidence_threshold: float = 0.80
    account_size_usd: float = 10000.0
    # Confidence-tiered NOTIONAL (USD). At 5x leverage the margin used is notional/5,
    # i.e. $2.5k notional = $500 margin. Highest tier whose confidence <= the signal wins.
    size_tiers: list = [[0.78, 2500.0], [0.82, 5000.0], [0.88, 7500.0], [0.90, 10000.0]]
    max_notional_usd: float = 10000.0     # hard cap ($2k margin at 5x)
    base_notional_usd: float = 2500.0     # fallback if no tier matches
    max_leverage: int = 5                 # constant 5x (clamped to each market's max)
    max_concurrent_positions: int = 3
    max_total_exposure_usd: float = 30000.0
    daily_loss_limit_usd: float = 1500.0
    per_ticker_cooldown_seconds: int = 900
    # Suppress a repeat of the SAME (ticker, direction) signal seen within this window —
    # even if the first was faded below the gate — so near-identical headlines don't get
    # inconsistent outcomes (one fires, one doesn't). 0 disables.
    duplicate_window_seconds: int = 900
    # ...but ONLY suppress when the prior same-(ticker,direction) signal is the SAME STORY:
    # token-overlap (Jaccard) >= this. Below it, two distinct catalysts that map to the same
    # ticker+direction (e.g. a supplier's earnings, then the company's own minutes later) both
    # pass. 0 => suppress on (ticker,direction) alone (old behavior).
    duplicate_similarity_min: float = 0.5
    # Risk safeguard: close an OPEN position immediately on contrary news (bearish on a
    # long / bullish on a short) at/above this confidence — below the entry gate, since
    # exiting a bad position is lower-risk than entering. 0 disables.
    contrary_exit_min_confidence: float = 0.50
    slippage_pct: float = 0.01            # max slippage tolerated on live market (IOC) orders
    dry_run_slippage_pct: float = 0.0005  # modeled adverse fill slippage in dry-run (5 bps) for realism
    # --- backtest-only realism knobs (never touch live behavior) ---
    # Adverse per-side fill slippage in the backtest. None -> use dry_run_slippage_pct, so
    # paper and backtest share one assumption; calibrate from scripts/slippage_report.py.
    backtest_slippage_pct: Optional[float] = None
    # EXTRA adverse slippage on stop/trail exits only (a stop-market fires into a moving
    # market). 0 until live fills say otherwise (scripts/slippage_report.py).
    backtest_stop_slippage_pct: float = 0.0
    # Fallbacks (used if a time_sensitivity is missing from the maps below).
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.03
    time_exit_seconds: int = 3600
    # Adaptive exits keyed by the analyzer's time_sensitivity. Wider stops everywhere;
    # quick "immediate" pops take a fixed TP (no whipsaw), while longer "hours"/"days"
    # plays use a trailing stop (trail_pct > 0 => no fixed TP, let it run).
    exit_horizons: dict[str, int] = {"immediate": 3600, "hours": 21600, "days": 259200}
    stop_loss_by_sensitivity: dict[str, float] = {"immediate": 0.03, "hours": 0.03, "days": 0.03}
    trail_pct_by_sensitivity: dict[str, float] = {"immediate": 0.0, "hours": 0.08, "days": 0.08}
    # Breakeven stop: once the trade has moved breakeven_arm_pct in our favor, the effective
    # stop floors at entry*(1 +/- breakeven_offset_pct) — a faded pop is evidence against a
    # news thesis, so stop paying full stop-distance for it. Composes with the trailing stop
    # (binding floor wins). One flat setting across sensitivities (fewer knobs to overfit).
    # 0 = off (default; validate via replay A/B + scripts/optimize.py --breakeven first).
    breakeven_arm_pct: float = 0.0
    breakeven_offset_pct: float = 0.0      # >0 locks in a sliver (covers fees+slippage)
    # Already-moved entry guard: we enter seconds-to-minutes behind HFT, so measure how much
    # of the repricing ALREADY happened (pre-news price -> entry price, in the signal's
    # direction) and haircut/reject accordingly. 'days' regime-change catalysts are never
    # rejected (haircut only) — a one-stop-sized pop doesn't exhaust a multi-session thesis.
    # Both thresholds 0 = guard disabled; the backtest still RECORDS pre_move_pct on every
    # gate-passing signal so edge_report.py can show whether chased entries lose before you
    # enable it. Suggested post-validation: haircut 0.0125, penalty 0.05, reject 0.03.
    pre_move_lookback_seconds: int = 180   # ref = last 1m close at/before the news
    pre_move_haircut_pct: float = 0.0      # in-direction move >= this -> confidence haircut
    pre_move_penalty: float = 0.05
    pre_move_reject_pct: float = 0.0       # >= this -> reject (haircut-only for 'days')
    # Confidence haircuts applied AFTER analysis (deterministic, before the gate/sizing):
    # illiquid crypto alts (by 24h notional volume) and front-run-prone listing news.
    liquidity_high_usd: float = 25_000_000   # >= this 24h volume: no penalty
    liquidity_med_usd: float = 10_000_000    # >= this: medium penalty; below: low-liquidity penalty
    liquidity_penalty_med: float = 0.05
    liquidity_penalty_low: float = 0.05
    listing_penalty: float = 0.20            # exchange listing/delisting news (competitive)
    # De-prioritize 2nd-order calls: if the resolved ticker (or its name/cashtag) is NOT
    # directly named in the news, the trade rests on an inferred chain (e.g. "Alphabet raises
    # capex" -> long NVDA), which is weaker and slower. Haircut confidence. 0 disables.
    indirect_mention_penalty: float = 0.05
    # Per-symbol confidence haircut for chronic backtest underperformers, applied like the
    # other deterministic haircuts. e.g. {"NVDA": 0.05} de-prioritizes NVDA across the board.
    symbol_penalties: dict[str, float] = {}
    # Brand-new Hyperliquid markets are thin right after listing -> haircut confidence
    # for the first few hours (universe tracks each market's first-seen time).
    new_listing_age_hours: float = 3.0
    new_listing_penalty: float = 0.15
    # Pre-IPO / "premarket" synthetic perps are riskier & thinner -> trade them smaller.
    # (No HL metadata flag exists for these, so this list is hand-maintained.)
    premarket_symbols: list[str] = []
    premarket_size_factor: float = 0.5
    # Live order-book liquidity guard at entry (LIVE + dry-run; NOT backtested — needs the live
    # book). trade.xyz equity perps are 24/7 but thin off-hours: a wide spread or shallow book
    # means a bad fill, so reject rather than trade into it. Applies only to the listed dexes
    # ("xyz" = equities; "" = crypto). max_spread_pct <= 0 disables the spread check; the depth
    # check is off unless min_top_depth_usd > 0. Fails OPEN (proceeds) on any book-fetch error.
    spread_guard_dexes: list[str] = ["xyz"]
    max_spread_pct: float = 0.015        # reject if (ask-bid)/mid exceeds this (1.5%)
    min_top_depth_usd: float = 0.0       # reject if top-of-book notional (thinner side) is below this


class FiltersConfig(BaseModel):
    skip_retweets: bool = True
    skip_replies: bool = True
    skip_quotes: bool = False
    max_news_age_seconds: int = 120
    allowed_dexes: list[str] = ["xyz", ""]
    source_whitelist: list[str] = []
    source_blacklist: list[str] = []
    market_blacklist: list[str] = []   # bare symbols never to trade, e.g. ["BRENTOIL"]


class AnalyzerConfig(BaseModel):
    model_fast: str = "claude-sonnet-4-6"
    model_smart: str = "claude-opus-4-8"
    escalate_margin: float = 0.0   # 0 = no Opus escalation (cost); >0 escalates borderline calls
    max_tokens: int = 1024
    temperature: float = 0.0   # 0 = deterministic, consistent classifications
    # Tiered triage to cut cost/latency: free regex prefilter -> cheap Haiku gate -> full model.
    # OFF by default: on low-volume curated channels these gates false-negative real catalysts
    # (Haiku is unreliable as a gate; name-only news misses the ticker regex). Enable only for
    # high-volume/noisy sources where the cost cut outweighs some missed signals.
    use_prefilter: bool = False
    use_triage: bool = False
    triage_model: str = "claude-haiku-4-5-20251001"
    # Burst aggregation (0 = OFF): wires print one event as several messages (EPS line,
    # then revenue, then guidance). When a piece arrives for a ticker that already has
    # analyzed pieces within this window, re-analyze the chronological concatenation and
    # act on THAT verdict — the first piece still acts immediately on its solo verdict.
    # Validate via replay A/B before enabling (the combined calls add ~1 analysis per
    # multi-piece burst). See analysis/burst.py.
    burst_window_seconds: float = 0.0
    include_crypto_universe: bool = True   # set false to trim crypto symbols from the prompt
    # Prompt-cache TTL for the static system blocks ("5m" or "1h"). News often arrives more
    # than 5 minutes apart, so the default 5m cache mostly MISSES — every call re-pays the
    # 1.25x cache-write premium on the full prefix. "1h" costs 2x on a cold write but 0.1x
    # for every call within the following hour: cheaper for this bot's bursty/sparse cadence.
    cache_ttl: str = "1h"
    # Transient-error resilience: a one-shot breaking-news catalyst can't be re-fetched, so a
    # 429 / timeout / connection drop / 5xx on the Claude call would PERMANENTLY miss the trade.
    # We own the retry loop (SDK retries disabled) so attempts are logged and tunable. Backoff is
    # exponential from retry_base_delay (with jitter); non-retryable errors (e.g. 400) fail fast.
    max_retries: int = 4
    retry_base_delay: float = 0.5
    # Per-attempt HTTP timeout (seconds). The SDK default is 10 MINUTES — a wedged connection
    # would stall an item far past the staleness gate. A tight cap hands control to our retry
    # loop instead (typical analyzer calls finish in 2-8s; escalations well under 30s).
    request_timeout_seconds: float = 30.0
    # API-enforced schema-valid tool output (structured-outputs "strict" mode): the analyzer can
    # never return a malformed direction / missing field. If the API/account rejects strict, the
    # analyzer logs a warning, retries that call without it, and disables it for the session.
    strict_tool: bool = True
    # output_config.effort for the full-analysis calls ("" = model default, i.e. unchanged).
    # "low" trims latency + output tokens on classification per Anthropic's Sonnet 4.6 guidance,
    # but can shift confidence calibration — A/B with scripts/backtest.py before enabling.
    # Never sent to Haiku models (unsupported); auto-disables on API rejection like strict_tool.
    effort: str = ""
    # Keep the cached system prefix WARM: if no analysis call touched it for this many seconds,
    # issue a ~free max_tokens=0 warm call. A cache read refreshes the TTL, so warming just under
    # it keeps the prefix alive at the 0.1x cached-read rate — the first headline after a quiet
    # stretch is then never on a cold cache (no full-price prefill / 2x rewrite / extra latency
    # on the news->order critical path). Keep comfortably under cache_ttl (2700s < 1h); for a
    # "5m" cache_ttl use ~240, or 0 to disable.
    cache_keepwarm_seconds: int = 2700
    # Skeptic ENTRY confirmation: every would-be entry (post-gate, pre-sizing) gets a second
    # call to a stronger model framed as a skeptic, WITH the market tape + measured pre-news
    # move (evidence the first pass never sees). Direction disagreement vetoes the entry;
    # under rule "min" the effective confidence becomes min(first-pass, skeptic) and must
    # clear confirm_gate (separate from the 0.80 fast gate — Opus calibrates conservative).
    # Fails OPEN on API errors. Adds one Opus call (~3-8s, cents) per would-be entry, inside
    # the serialized trade section. Validate via the replay A/B before enabling live.
    confirm_entries: bool = False
    confirm_model: str = "claude-opus-4-8"
    confirm_rule: str = "min"        # "min" | "veto_only" (direction check only)
    confirm_gate: float = 0.78       # min(fast, skeptic) must clear this; tune from backtest
    confirm_max_tokens: int = 512


class UIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    # Token required to hit the mutating /api/control endpoint (toggle LIVE / halt) from a
    # NON-localhost client. Empty => control is allowed from localhost only (safe even if you
    # bind host to 0.0.0.0). Set this if you expose the dashboard off-box.
    auth_token: str = ""


class ArenaEntrant(BaseModel):
    """One competitor in the live arena. Only `gate` varies between entrants — every other
    risk/sizing/exit parameter comes from the shared RiskConfig. `wallet` is the secrets-key
    prefix for this entrant's funded Hyperliquid wallet (ARENA_<WALLET>_ADDRESS / _SECRET);
    unused in paper/dry-run (virtual capital)."""
    key: str                      # short id: sonnet / gpt / gemini / deepseek / grok
    model: str                    # routing id: claude-sonnet-4-6 / openai:gpt-5.4 / xai:grok-4.3
    gate: float                   # confidence threshold (the one varied parameter)
    wallet: str = ""              # secrets key prefix for the live wallet (empty = paper only)
    live: bool = False            # go-live: flip True (+ fund the wallet) to trade real money;
    #                               only takes effect if the wallet secrets are present


class ArenaConfig(BaseModel):
    """Live 5-model news-trading arena. Each entrant trades the SAME news with identical params
    except its confidence gate, on its own $capital. Gates are from Phase-1 training (replay +
    OOS + earnings bench). Refine live with scripts/optimize.py walk-forward.
    The arena runs iff `python -m hlbot.main_arena` is the entrypoint — there is
    deliberately no enable flag here (an unread one would be an operator trap)."""
    capital_per_model_usd: float = 10000.0
    ui_port: int = 8001
    # Thinking models (DeepSeek V4 Pro) truncate->empty at the 1024 default; arena needs >=2048.
    max_tokens: int = 2048
    entrants: list[ArenaEntrant] = [
        ArenaEntrant(key="sonnet",   model="claude-sonnet-4-6",        gate=0.80, wallet="sonnet"),
        ArenaEntrant(key="gpt",      model="openai:gpt-5.4",           gate=0.80, wallet="gpt"),
        ArenaEntrant(key="gemini",   model="google:gemini-3.5-flash",  gate=0.85, wallet="gemini"),
        ArenaEntrant(key="deepseek", model="deepseek:deepseek-v4-pro", gate=0.85, wallet="deepseek"),
        ArenaEntrant(key="grok",     model="xai:grok-4.3",             gate=0.85, wallet="grok"),
    ]


class AppConfig(BaseModel):
    dry_run: bool = True
    poll_interval_seconds: float = 3.0
    universe_refresh_seconds: int = 600
    # Telegram source channels to ingest (usernames or t.me links), e.g. ["tradfi", "AggrNews"].
    telegram_channels: list[str] = []
    enable_tree_feed: bool = True   # set false to run Telegram-channels-only (no Tree)
    # Catch-up poll for the Telegram source: every N seconds, re-fetch each channel's newest
    # messages as a BACKSTOP to the live handler (which silently dies after a network/VPN drop)
    # and to actively verify + repair the connection. Keep well under max_news_age_seconds so a
    # polled item is still fresh enough to trade. 0 disables (fragile live-handler-only mode).
    telegram_poll_seconds: int = 45
    # Markdown file with a current macro/political "regime" brief, injected into the
    # analyzer prompt so news is judged against the prevailing climate. Built at startup
    # with `python scripts/build_regime.py`, then auto-refreshed while running.
    regime_context_file: str = "data/regime.md"
    # Hand-curated few-shot exemplars injected into the CACHED analyzer prefix (between the
    # instructions and the regime). Empty/absent file = no block (prompt unchanged). Curate
    # candidates with scripts/mine_exemplars.py — ONLY from history before your validation
    # window, and A/B with a FRESH analysis-cache path (prompt changes invalidate the cache).
    exemplars_file: str = "data/exemplars.md"
    # Hand-written "knowledge bridge": dated, durable post-training-cutoff facts (policy,
    # geopolitics, structural market events) injected into the CACHED prefix between the
    # exemplars and the regime, grounding staleness/novelty judgments the 14d regime brief
    # can't cover. Facts only — no levels, no advice. Empty/absent file = no block.
    # Prompt change: A/B with a FRESH analysis-cache path before shipping content.
    context_bridge_file: str = "data/context_bridge.md"
    # Live regime auto-refresh: every N seconds, rebuild the brief from the last
    # `regime_live_lookback_days` of persisted news (so mid-session events feed the
    # context for SUBSEQUENT news). 0 disables. Needs >= regime_min_items recent items.
    regime_refresh_seconds: int = 43200      # 12h (Sonnet brief ~$3.5/mo; lower = fresher+pricier)
    regime_live_lookback_days: float = 14.0
    regime_min_items: int = 40
    # On startup, backfill recent channel history into the store so the regime has a full
    # trailing window immediately (matches the backtest). Gap-aware: a warm restart only
    # fetches the news missed while down, so frequent reboots stay fast. 0 disables.
    regime_backfill_days: float = 16.0
    # Anti-re-trade event memory: inject catalysts we've already entered in the last
    # N days into the analyzer prompt so a resurfaced same-event headline is flagged
    # stale (refreshed from the store every catalyst_memory_refresh_seconds, no tokens).
    catalyst_memory_days: float = 7.0
    catalyst_memory_refresh_seconds: int = 300
    # Persistent all-time token-usage ledger; live Claude calls accumulate here across
    # restarts (view with scripts/token_report.py). Backtests do NOT write to it.
    token_ledger_file: str = "data/token_usage.json"
    # Feed-silence watchdog: if NO news arrives for this many seconds, alert (Telegram + UI)
    # that the feeds may be down. Tree/Telegram both auto-reconnect, but a silently-dead feed
    # (dropped MTProto session, a WS that reconnects but emits nothing) otherwise just stops
    # the bot trading with no signal. Alerts once on silence, once again on recovery. 0 disables.
    # Set well above the feeds' normal quiet stretches — the curated aggregators can go many
    # hours between posts, so a day-scale threshold avoids false alarms.
    feed_silence_alert_seconds: int = 86400   # 1 day
    risk: RiskConfig = RiskConfig()
    filters: FiltersConfig = FiltersConfig()
    analyzer: AnalyzerConfig = AnalyzerConfig()
    ui: UIConfig = UIConfig()
    arena: ArenaConfig = ArenaConfig()


def load_app_config(path: str = "config.yaml") -> AppConfig:
    p = pathlib.Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return AppConfig(**(data or {}))


RUNTIME_STATE_FILE = "data/runtime_state.json"


def _utc_day() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class RuntimeState:
    """Mutable flags the UI/kill-switch can toggle at runtime. Persisted to disk so a
    manual halt or a dry/live toggle SURVIVES a restart (a crash must not silently
    resume trading or flip the mode). config.yaml's dry_run is only the initial default
    for a fresh install; once toggled, the persisted value wins (delete the state file
    to reset)."""

    def __init__(self, dry_run: bool, path: str | None = None):
        # path: which state file this process owns. The arena entrypoint passes its own file —
        # production and arena must NEVER share one, or the arena's saves clobber the operator's
        # production dry/live + kill-switch toggles (and vice versa). None -> the module default
        # (read dynamically so tests can redirect RUNTIME_STATE_FILE).
        self._path = path
        self.dry_run = dry_run
        self.trading_halted = False  # kill switch
        self.halt_reason = ""
        self.halt_is_daily = False   # True => set by the daily-loss limit (auto-resumes next UTC day)
        self.halt_day = ""           # UTC day the daily-loss halt was set
        # Arena: PER-MODEL daily-loss halts {model_id: [utc_day, reason]}. One lane hitting its
        # daily limit must NOT freeze the others' independent wallets — only the manual kill
        # switch (halt()/trading_halted) is global. model_id None keeps the global path above.
        self._model_daily: dict = {}
        self._load()

    def _load(self) -> None:
        p = pathlib.Path(self._path or RUNTIME_STATE_FILE)
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            self.dry_run = bool(d.get("dry_run", self.dry_run))
            self.trading_halted = bool(d.get("trading_halted", False))
            self.halt_reason = d.get("halt_reason", "") or ""
            self.halt_is_daily = bool(d.get("halt_is_daily", False))
            self.halt_day = d.get("halt_day", "") or ""
            md = d.get("model_daily")
            self._model_daily = {k: list(v) for k, v in md.items()} if isinstance(md, dict) else {}
        except Exception:  # noqa: BLE001 - corrupt/partial file: fall back to defaults
            pass

    def _save(self) -> None:
        try:
            p = pathlib.Path(self._path or RUNTIME_STATE_FILE)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"dry_run": self.dry_run,
                                     "trading_halted": self.trading_halted,
                                     "halt_reason": self.halt_reason,
                                     "halt_is_daily": self.halt_is_daily,
                                     "halt_day": self.halt_day,
                                     "model_daily": self._model_daily}), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def set_dry_run(self, value: bool) -> None:
        self.dry_run = value
        self._save()

    def halt(self, reason: str) -> None:
        """Manual kill switch: stays halted until explicitly resumed."""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_is_daily = False
        self.halt_day = ""
        self._save()

    def halt_daily(self, reason: str, model_id: str | None = None) -> None:
        """Daily-loss halt: blocks trading for the rest of the current UTC day, then auto-resumes
        (see maybe_auto_resume / is_daily_halted). With model_id (ARENA) it halts ONLY that lane —
        the other lanes' independent wallets keep trading. Without it (production/single-model)
        it halts globally. A manual kill switch is NEVER overwritten."""
        if model_id:
            self._model_daily[model_id] = [_utc_day(), reason]
            self._save()
            return
        if self.trading_halted and not self.halt_is_daily:
            return   # don't downgrade an existing manual kill to an auto-resuming halt
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_is_daily = True
        self.halt_day = _utc_day()
        self._save()

    def is_daily_halted(self, model_id: str | None = None) -> tuple[bool, str]:
        """(halted, reason) for a lane's daily-loss halt; auto-expires on UTC day roll. model_id
        None reflects the GLOBAL daily halt (production)."""
        if not model_id:
            return (self.trading_halted and self.halt_is_daily, self.halt_reason)
        entry = self._model_daily.get(model_id)
        if not entry:
            return (False, "")
        if entry[0] != _utc_day():            # day rolled -> this lane auto-resumes
            self._model_daily.pop(model_id, None)
            self._save()
            return (False, "")
        return (True, entry[1])

    def maybe_auto_resume(self) -> bool:
        """Auto-clear an EXPIRED global daily-loss halt + prune stale per-model halts once the UTC
        day has rolled. Returns True if the global halt was resumed. (Per-model halts also expire
        lazily in is_daily_halted.)"""
        resumed = False
        if self.trading_halted and self.halt_is_daily and self.halt_day != _utc_day():
            self.trading_halted = False
            self.halt_reason = ""
            self.halt_is_daily = False
            self.halt_day = ""
            resumed = True
        stale = [m for m, e in self._model_daily.items() if e[0] != _utc_day()]
        for m in stale:
            self._model_daily.pop(m, None)
        if resumed or stale:
            self._save()
        return resumed

    def resume(self) -> None:
        """Manual operator resume: clears the global kill AND every per-lane daily halt."""
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_is_daily = False
        self.halt_day = ""
        self._model_daily.clear()
        self._save()


class Config:
    def __init__(self, app_config_path: str = "config.yaml",
                 runtime_state_file: str | None = None):
        self.secrets = Secrets()
        self.app = load_app_config(app_config_path)
        self.runtime = RuntimeState(self.app.dry_run, path=runtime_state_file)

    @property
    def base_url(self) -> str:
        from hyperliquid.utils import constants

        return (
            constants.TESTNET_API_URL
            if self.secrets.hl_network.lower() == "testnet"
            else constants.MAINNET_API_URL
        )
