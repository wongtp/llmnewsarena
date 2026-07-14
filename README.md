# llmnewsarena — news-catalyst trading bot for Hyperliquid & trade.xyz

Automated **news-catalyst** trader for [Hyperliquid](https://hyperliquid.xyz) perps —
standard crypto perps (`BTC`, `ETH`, alts) and [trade.xyz](https://trade.xyz) HIP-3
equity/index/commodity perps (e.g. `xyz:MRVL`).

It ingests real-time news (Telegram channels and/or [Tree of Alpha](https://news.treeofalpha.com)),
uses **Claude** to extract `(ticker, direction, confidence, rationale)`, and — when a
tradable, high-confidence catalyst appears — opens a confidence-scaled position with a
stop-loss, trailing stop, and time-based exit. Every entry/exit fires a **Telegram alert**
and streams to a **local web dashboard**. A full **replay backtester** re-runs the exact
production pipeline over historical news so every change can be A/B-validated before it
touches real money.

> Ships in **DRY-RUN** by default: full pipeline + alerts + UI, but **simulated fills, no
> real orders**. Validate signal quality before risking capital. This is automated trading
> software — it can lose money. Nothing here is financial advice.

## Strategy thesis

A retail bot will never out-speed HFT — the first minute after a headline belongs to
them. The edge this bot targets is **judgment, not latency**: high-conviction,
directionally sound trades on **regime-changing news** (earnings surprises, guidance
resets, M&A, macro shocks), a good-enough (not perfect) entry ~a minute after the print,
and **hours-to-days holds** with asymmetric exits (tight stop, loose trail, long time
horizon) so losses are capped and winners run. Most of the PnL comes from a fat right
tail of regime-change winners — the risk layer exists to keep the bot alive between them.

## How it works

```
Telegram channels ─► dedup ─► Claude analyzer ─► risk gate ─► executor ─► position mgr
 (+ optional Tree)   (id,      (cached prompt,    (haircuts,    (market     (SL / trail /
                      stale,    strict tool       conf gate,    IOC, dry/   time exits,
                      filters)  output)           sizing,       live)       exchange stop)
                                                  caps)
                                └────── event bus ──────► Telegram + Web UI + SQLite
```

- **Feeds** — Telethon-based Telegram channel reader (real-time handler + catch-up poll
  backstop) and/or the Tree of Alpha websocket. Both feed the same pipeline.
- **Dedup & filters** — id-based dedup (restored from DB on boot), stale-news cutoff,
  retweet/reply filters, source white/blacklists, same-story suppression (ticker +
  direction + token-overlap similarity within a window).
- **Analyzer** — a tool-use Claude call with a heavily cached prompt prefix (symbol
  universe, few-shot slots, auto-refreshed macro **regime brief**, recent
  **catalyst memory** so rebroadcasts of an already-traded event get flagged stale).
  Strict schema-enforced output, owned retry loop, 30s timeout, prompt-cache keep-warm.
  Every analysis is stamped with latency and cost for the UI.
- **Risk engine** — hard confidence gate (default 0.80) plus deterministic confidence
  haircuts (illiquidity, listing news, indirect mention, per-symbol penalties, brand-new
  markets), confidence-tiered notional sizing, portfolio caps (max concurrent positions,
  total exposure), per-ticker cooldowns, spread/depth guard on thin books, and a
  daily-loss kill switch (counts realized + open drawdown; auto-resumes next UTC day).
- **Executor** — paper fill (dry-run) or live IOC market order with slippage cap. Live
  entries also rest a **reduce-only exchange-side stop** as a backstop, so the position
  is protected even if the bot process dies.
- **Position manager** — poll loop enforcing stop / trailing stop / take-profit / time
  exits (adaptive per the analyzer's `time_sensitivity`), contrary-news exits,
  exchange-side close reconciliation, and crash recovery (open positions restored from
  the DB on boot).

## Dashboard

`python -m hlbot.main` serves a single-page dashboard on http://127.0.0.1:8000:

- Live news feed: each headline appears instantly, then enriches in place — pulsing
  "analyzing…", then the verdict with a latency/cost badge; traded rows tint green,
  rejected rows show the reject reason.
- Live TradingView-style chart (vendored
  [Lightweight Charts](https://github.com/tradingview/lightweight-charts), Apache-2.0)
  of any Hyperliquid/trade.xyz pair — candles and volume stream from Hyperliquid's
  public websocket, history loads as you scroll back. Fills are marked on the exact
  candle; open positions draw entry and (trailing) stop lines with live PnL.
- Positions panel with exchange-style columns (entry, size, PnL, funding, stop),
  re-marked on every streamed price tick; account panel with balance, margin ratio, and
  aggregate uPnL.
- Controls: DRY/LIVE toggle, kill switch, manual position close.

## Telegram

Two independent integrations:

- **Alerts bot** (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`): entry/exit/halt cards, plus
  an interactive `/start` position browser with two-step manual close.
- **Channel ingestion** (`TELEGRAM_API_ID`/`TELEGRAM_API_HASH`, MTProto via your user
  account): reads news channels in real time — often faster than aggregator APIs, and
  covers equity-news channels Tree doesn't carry.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
cp .env.example .env      # then fill it in
```

On Windows use `./.venv/Scripts/python.exe` in place of `.venv/bin/python` throughout.

Fill `.env` (see `.env.example` for the full commented list):

| Var | What |
|-----|------|
| `HL_ACCOUNT_ADDRESS` | Your main Hyperliquid account address (holds funds) |
| `HL_SECRET_KEY` | API/agent wallet private key (trade-only; **cannot withdraw**) |
| `HL_NETWORK` | `mainnet` (trade.xyz is mainnet) or `testnet` |
| `ANTHROPIC_API_KEY` | For the Claude analyzer |
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | Telegram channel ingestion (my.telegram.org) |
| `TREE_API_KEY` | Tree of Alpha feed (optional; https://news.treeofalpha.com/account) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Alerts bot (optional; @BotFather) |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, … | Arena entrants only (optional) |
| `TWELVEDATA_API_KEY`, `ALPACA_API_KEY/SECRET` | `earnings_bench.py` data only (optional) |

All tunables live in **`config.yaml`** (risk caps, confidence gate, sizing tiers,
exit parameters, feeds, filters, analyzer models, UI port), parsed into pydantic models
in `src/hlbot/config.py`. The yaml comments document *why* each value is set — most were
derived from backtests.

### Telegram channel ingestion (one-time login)

```bash
.venv/bin/python scripts/telegram_login.py   # interactive: code goes to your phone
```

It prints the channels your account can read and saves `data/tg_session`. Then list the
channels in `config.yaml`:

```yaml
telegram_channels: ["some_news_channel", "another_channel"]   # usernames, no @
enable_tree_feed: false                                       # true = also ingest Tree
```

## Verify (do this first)

```bash
.venv/bin/python -m pytest -q               # offline unit suite (no keys needed)
.venv/bin/python scripts/smoke_test.py      # connectivity check (needs .env)
.venv/bin/python scripts/inject_news.py     # e2e acceptance in dry-run: feeds a sample
                                            # headline through the full pipeline;
                                            # expect ANALYSIS -> DECISION -> PAPER FILL
.venv/bin/python scripts/check_account.py   # read-only pre-live account verification
```

## Run

```bash
.venv/bin/python -m hlbot.main              # dashboard at http://127.0.0.1:8000
```

For unattended operation run it under a supervisor (launchd/systemd) with auto-restart —
every long-lived component is already supervised in-process (crash → restart with
backoff), positions/cooldowns/dedup state survive restarts via SQLite, and on live
entries the exchange-side stop protects positions while the process is down.

## Backtesting & research tooling

The replay backtester runs historical news through the **exact production** analyzer,
gate, sizing, and exit logic against historical candles — only the price source differs.
Analyses are cached on disk, so risk-side experiments replay near-free.

```bash
.venv/bin/python scripts/backtest.py --source telegram --days 60   # live-faithful replay
#   --history-file data/<dump>.json  -> replay a saved news set (A/B arms see identical
#       items; also avoids touching the live bot's Telegram session)
#   --cache-tag <variant>            -> fresh analysis cache for prompt variants
.venv/bin/python scripts/edge_report.py        # calibration / OOS gate / MAE-MFE / Kelly sizing
.venv/bin/python scripts/optimize.py           # walk-forward exit-param sweep
.venv/bin/python scripts/gate_sweep.py         # confidence-gate sweep from cached analyses
.venv/bin/python scripts/slippage_report.py    # live fill quality vs the models
.venv/bin/python scripts/missed.py             # would-be PnL of skipped signals
.venv/bin/python scripts/compare_runs.py       # archived-run table / diff two runs
.venv/bin/python scripts/mine_exemplars.py     # propose few-shot exemplar candidates
.venv/bin/python scripts/token_report.py       # all-time Claude spend ledger
.venv/bin/python scripts/earnings_bench.py     # earnings-print classification bench on
                                               # external data (+ minute-level exit walks)
```

**The validation workflow:** every profitability-relevant change ships **default-off**
and must beat the baseline in an A/B replay (same window, same news set, same caches,
one variable) before its config key is flipped — then flipped one key at a time with a
week of live fill-quality watching in between. Prompt changes need a fresh analysis
cache (`--cache-tag`); the cache is keyed by news id only. Backtest PnL includes adverse
slippage, gap-aware stop fills, and funding on multi-hour holds. `scripts/optimize.py`
is walk-forward — trust the validation column and parameter plateaus, never the
in-sample top line.

Hard-won lesson, encoded in the defaults: this strategy's PnL is a fat right tail.
Mechanisms that de-risk around entry (pre-move guards, second-opinion vetoes, breakeven
floors, calibration-tightening prompt additions) consistently cut the tail more than the
losses they save. Several such features exist in the codebase, fully built and tested —
and default-**off** because the replays said no.

## Multi-LLM arena (optional)

```bash
.venv/bin/python -m hlbot.main_arena        # arena dashboard at :8001
```

An [Alpha-Arena](https://nof1.ai)-style competition: N LLM entrants (Claude, GPT,
Gemini, DeepSeek, Grok — see `arena:` in `config.yaml`) each analyze the **same news**
with identical risk/sizing/exits, each on its own virtual (or real) $10k. Going live is
double-gated per lane: `live: true` in the config **and** a funded
`ARENA_<KEY>_ADDRESS`/`_SECRET` wallet in `.env` — otherwise the lane paper-trades. The
arena dashboard shows a leaderboard with equity curves and a per-headline panel
comparing every model's direction/confidence/rationale side-by-side. Research tooling:
`scripts/control_arms.py` (event-selection vs direction skill decomposition),
`scripts/multisample.py` (gate-straddle stability), `scripts/rubric_judge.py`,
`scripts/compare_models.py`.

## Going live (canary first)

1. Run in dry-run for a while; review the dashboard/Telegram — are the signals good?
2. Tighten `config.yaml`: small `base_notional_usd`/`max_notional_usd`, low
   `max_leverage`, a `daily_loss_limit_usd` you're comfortable with.
3. Flip `dry_run: false` (or use the UI toggle). Start with minimal size as a canary;
   watch a real entry + stop behave end-to-end, then scale.
4. The kill switch (UI button, or automatic on the daily-loss limit) halts new entries.

Things to confirm on your account:

- The agent (API) wallet must be **approved** for your account on Hyperliquid.
- HIP-3 (`xyz`) perps use **independent margining** — collateral must be available on
  the `xyz` dex, not just the core perp dex.
- trade.xyz equity perps trade 24/7 but can be **thin off-hours** — the spread guard and
  conservative sizing help, but be aware.
- The exchange-side stop rests at the *initial* stop (trailing/TP/time exits stay
  bot-side and cancel it on a normal close). Validate on testnet first: confirm the stop
  appears after a canary entry, fires correctly, and is cancelled on a bot-side close.
- Margin mode is **cross** — a badly slipped stop can draw on the whole dex balance;
  size accordingly.

## Layout

```
src/hlbot/
  pipeline.py        # the hot path: news -> analyze -> risk -> execute (latency-aware)
  main.py            # wiring + supervised tasks; main_arena.py = arena entrypoint
  bus.py             # pub/sub fan-out to observers (never for pipeline ordering)
  news/              # Telegram source (Telethon), Tree websocket client, dedup
  analysis/          # Claude analyzer, prompts, regime brief, symbol universe,
                     #   pricing sheet, token ledger, multi-provider registry (arena)
  trading/           # HL client (sync SDK via asyncio.to_thread), risk engine,
                     #   executor, position manager
  backtest/          # replay engine, walk-forward optimizer, edge/calibration, reports
  notify/            # Telegram alerts + interactive position browser
  store/             # SQLite (WAL) audit log + crash-recovery state
  ui/                # FastAPI dashboards (main :8000, arena :8001) + static frontends
scripts/             # backtest / research / ops CLIs (see above)
tests/               # offline unit suite (no keys, no network, tmp-dir state)
```

Runtime state lives in `data/` (gitignored): SQLite audit log, regime brief, token
ledger, Telegram session, backtest caches and archives. `data/runtime_state.json`
persists the UI's dry/live and kill-switch toggles across restarts — **persisted state
wins over `config.yaml`'s `dry_run` after the first toggle**; delete the file to reset.

## Safety

This is automated trading software operating with real money when `dry_run: false`. It
can lose money quickly. Start in dry-run, use small size, keep the daily-loss limit on,
use a trade-only agent wallet, and never fund the account with more than you can afford
to lose. Not financial advice.
