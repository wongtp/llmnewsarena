"""Backtest the news strategy over recent Tree-of-Alpha history.

    python scripts/backtest.py                  # ~7 days (all REST history)
    python scripts/backtest.py --days 3
    python scripts/backtest.py --focus saylor   # spotlight a topic (default: saylor)
    python scripts/backtest.py --no-prefilter   # analyze every item (slow/expensive)

DRY analysis only — never places orders. Claude results are cached in
data/bt_analysis_cache.json so re-runs are fast.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.universe import Universe  # noqa: E402
import time  # noqa: E402

from hlbot.backtest.engine import (  # noqa: E402
    fetch_telegram_history,
    forward_returns,
    load_or_build_regime,
    run_backtest,
    run_live_replay,
    simulate,
)
from hlbot.backtest.report import write_html  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    # Arena entrants — routed by the "provider:model_id" prefix (analysis/providers/registry.py).
    # The model ids are best-guess defaults; CONFIRM each against the vendor's live API and
    # update here (the prefix decides the provider, the part after ":" is sent to the SDK).
    "gpt": "openai:gpt-5.4",
    "gemini": "google:gemini-3.5-flash",
    "deepseek": "deepseek:deepseek-v4-pro",
    "glm": "zhipu:glm-5.2",
    "grok": "xai:grok-4.3",
}


def ts(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%m-%d %H:%M")


def report(res: dict) -> None:
    w = res["window"]
    print("\n" + "=" * 78)
    print(f" BACKTEST  {ts(w[0])} -> {ts(w[1])} UTC")
    print("=" * 78)
    print(f" news in window: {res['n_news']}   analyzed: {res['n_analyzed']}   "
          f"signals(passed gate): {res['n_signals']}   simulated trades: {len(res['trades'])}")
    if res.get("n_no_data"):
        print(f" !! {res['n_no_data']} gate-passing signal(s) dropped: no candle data "
              f"(not listed yet / data gap) — see backtest log for symbols")
    fs = res.get("funding_stats") or {}
    if fs.get("failed"):
        print(f" !! funding history unavailable for {fs['failed']} trade(s) (cost 0 assumed)")

    trades = sorted(res["trades"], key=lambda t: t.time_ms)
    if not trades:
        print("\n No trades triggered in this window.")
        return

    wins = [t for t in trades if t.pnl > 0]
    total = sum(t.pnl for t in trades)
    invested = sum(t.notional for t in trades)
    print(f" total PnL: ${total:,.2f}   on ${invested:,.0f} deployed "
          f"({total / invested * 100:+.2f}%)   win rate: {len(wins)}/{len(trades)} "
          f"({len(wins) / len(trades) * 100:.0f}%)")
    from hlbot.backtest.metrics import summary_metrics
    m = summary_metrics(trades, res.get("account_size_usd", 0.0))
    pf = "inf" if m["profit_factor"] is None else f"{m['profit_factor']:.2f}"
    funding_total = sum(getattr(t, "funding_usd", 0.0) for t in trades)
    print(f" profit factor: {pf}   max realized DD: ${m['max_dd_usd']:,.0f}"
          + (f" ({m['max_dd_pct_of_account']:.1f}% of account)"
             if m["max_dd_pct_of_account"] is not None else "")
          + f"   median hold: {m['median_hold_h']:.1f}h"
          + f"   funding paid: ${funding_total:,.2f}")
    print(f" peak exposure: ${m['peak_notional']:,.0f} across {m['peak_concurrent']} concurrent")

    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.reason] = by_reason.get(t.reason, 0) + 1
    print(" exits: " + ", ".join(f"{k}={v}" for k, v in by_reason.items()))

    print("\n  time        symbol  side   conf   entry      exit       pnl      reason")
    print("  " + "-" * 74)
    for t in trades:
        print(f"  {ts(t.time_ms):11} {t.symbol:<7} {t.side:<5} {t.confidence:>4.0%}  "
              f"{t.entry_px:>9.4g}  {t.exit_px:>9.4g}  {t.pnl:>+8.2f}  {t.reason}")
        print(f"               {t.title[:78]}")

    # CSV
    out = pathlib.Path("data/backtest_trades.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["time_utc", "symbol", "side", "confidence", "notional", "entry",
                       "exit", "pnl", "funding", "mae_pct", "mfe_pct", "interval",
                       "reason", "title", "link"])
        for t in trades:
            wcsv.writerow([ts(t.time_ms), t.symbol, t.side, f"{t.confidence:.2f}",
                           f"{t.notional:.2f}", f"{t.entry_px:.6g}", f"{t.exit_px:.6g}",
                           f"{t.pnl:.2f}", f"{t.funding_usd:.2f}", f"{t.mae_pct:.4f}",
                           f"{t.mfe_pct:.4f}", t.candle_interval,
                           t.reason, t.title, t.link or ""])
    print(f"\n  trade log -> {out}")


async def focus(res: dict, hl: HLClient, keyword: str) -> None:
    kw = keyword.lower()
    items = [i for i in res["all_items"] if kw in (i.title or "").lower()]
    if not items:
        print(f"\n No items matched focus '{keyword}'.")
        return
    print("\n" + "=" * 78)
    print(f" FOCUS: '{keyword}'  ({len(items)} matching news items) — what the bot thought")
    print("=" * 78)
    analyses = res["analyses"]
    traded_ids = {(t.time_ms, t.symbol) for t in res["trades"]}
    for it in items[:30]:
        a = analyses.get(it.id)
        verdict = "(not analyzed)"
        if a:
            verdict = (f"{a.ticker or '-'} {a.direction} conf={a.confidence:.0%}"
                       f"{' STALE' if a.is_stale else ''} :: {a.rationale[:90]}")
        print(f"  {ts(it.time_ms)} | {(it.source or 'TW')[:8]:8} | {it.title[:64]}")
        print(f"     -> {verdict}")

    # Forward returns for BTC from the earliest 'sell/sold' headline in the focus set.
    sell_items = [i for i in items if any(k in i.title.lower()
                                          for k in ("sell", "sold", "sells"))]
    anchor = (sell_items or items)[0]
    print("\n" + "-" * 78)
    print(f" BTC forward move from breaking headline @ {ts(anchor.time_ms)} UTC:")
    print(f"   \"{anchor.title[:88]}\"")
    fr = await forward_returns(hl, "BTC", anchor.time_ms)
    if fr:
        print(f"   price at news: {fr.get('base_px'):.0f}")
        for k in ("+1h", "+6h", "+24h", "+72h"):
            if k in fr:
                px, pct = fr[k]
                short_pnl = -pct  # a short profits from a price drop
                print(f"   {k:>5}: {px:>8.0f}  ({pct:+.2f}%)   short P&L: {short_pnl:+.2f}%")
    print("-" * 78)


async def annotate_would_pnl(res: dict, hl, universe, cfg, cache_path: str,
                             min_conf: float = 0.50) -> int:
    """Fill row['would_pnl'] for SKIPPED directional reads (conf>=min_conf, not blacklisted,
    tradable, price existed) = what they'd have returned if taken. Traded rows keep their
    real pnl. Free (candles only). Returns how many rows were annotated."""
    r = cfg.app.risk
    blacklist = {b.upper() for b in cfg.app.filters.market_blacklist}
    try:
        acache = json.load(open(cache_path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        acache = {}
    n = 0
    for row in res.get("rows", []):
        if row.get("status") == "traded" or row.get("direction") not in ("long", "short"):
            continue
        if (row.get("confidence") or 0) < min_conf:
            continue
        tk = (row.get("ticker") or "").upper()
        if not tk or tk in blacklist:
            continue
        ac = (acache.get(row["id"]) or {}).get("asset_class")
        mkt = universe.resolve(row["ticker"], ac)
        if not mkt or mkt.symbol.upper() in blacklist:
            continue
        tr = await simulate(hl, mkt, row["direction"], row["time_ms"], row["confidence"],
                            row["time_sensitivity"], r, news_id=row["id"])
        if tr:
            row["would_pnl"] = round(tr.pnl, 2)
            n += 1
    return n


def archive_run(res: dict, cfg, report_path: str, cache_path: str | None, keep: int = 5):
    """Snapshot this run (report HTML + trades CSV + analysis cache + a meta.json summary)
    into data/bt_history/<timestamp>/ so prior runs stay available for comparison. Keeps
    the most recent `keep` runs and prunes older ones."""
    import json
    import shutil

    from hlbot.backtest.metrics import summary_metrics
    from hlbot.backtest.report import _cost
    hist = pathlib.Path("data/bt_history")
    hist.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = hist / stamp
    dest.mkdir(exist_ok=True)
    for p in [report_path, "data/backtest_trades.csv", cache_path]:
        if p and pathlib.Path(p).exists():
            shutil.copy2(p, dest / pathlib.Path(p).name)
    trades = res.get("trades", [])
    wins = sum(1 for t in trades if t.pnl > 0)
    r = cfg.app.risk
    m = summary_metrics(trades, r.account_size_usd)
    meta = {
        "stamp": stamp, "window": res.get("window"), "model": res.get("model"),
        "threshold": res.get("threshold"), "trades": len(trades), "wins": wins,
        "win_rate": round(wins / len(trades), 3) if trades else 0.0,
        "total_pnl": round(sum(t.pnl for t in trades), 2),
        "profit_factor": m["profit_factor"],
        "max_dd_usd": m["max_dd_usd"],
        "median_hold_h": m["median_hold_h"],
        "n_no_data": res.get("n_no_data", 0),
        "cost_usd": round(_cost(res.get("usage", {})).get("total", 0.0), 4),
        "exit_config": res.get("exit_config"),
        "duplicate_window_s": r.duplicate_window_seconds,
        "contrary_floor": r.contrary_exit_min_confidence,
        "cache": pathlib.Path(cache_path).name if cache_path else None,
        "report": pathlib.Path(report_path).name,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for old in sorted(d for d in hist.iterdir() if d.is_dir())[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return str(dest)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-prefilter", action="store_true")
    ap.add_argument("--focus", default="saylor")
    ap.add_argument("--model", default="sonnet", help="haiku | sonnet | opus | <full id> "
                    "(default = the production analyzer; haiku was tested & rejected 2026-06-12)")
    ap.add_argument("--source", default="tree", choices=["tree", "telegram"],
                    help="news source to backtest")
    ap.add_argument("--threshold", type=float, default=None, help="override confidence gate")
    ap.add_argument("--whitelist", default=None,
                    help="comma-separated source whitelist override (e.g. '@AggrNews,THE BLOCK')")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the HTML report")
    ap.add_argument("--no-archive", action="store_true",
                    help="don't snapshot this run into data/bt_history/ (keeps last 5)")
    ap.add_argument("--no-would-pnl", action="store_true",
                    help="skip computing 'would-be PnL' for skipped >=50%% reads (faster)")
    ap.add_argument("--cache-tag", default=None,
                    help="suffix for the analysis-cache file. REQUIRED when A/B-ing a prompt "
                         "change: the cache is keyed by news id only (no prompt fingerprint), "
                         "so a variant run must not reuse the baseline's cached analyses")
    ap.add_argument("--history-file", default=None,
                    help="JSON snapshot of fetched Telegram history: loaded if it exists, "
                         "written after a successful fetch. Lets A/B re-runs replay the "
                         "IDENTICAL news set without touching Telethon again (the live bot "
                         "holds the session file — concurrent use hits 'database is locked')")
    ap.add_argument("--no-regime", action="store_true",
                    help="disable regime-context priming for this run")
    ap.add_argument("--rebuild-regime", action="store_true",
                    help="force-rebuild the regime brief (otherwise reuse the saved one, free)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override analyzer max_tokens for this run. ARENA: set 2048+ so thinking "
                         "models (DeepSeek V4 Pro / GLM 5.2) don't truncate their reasoning into "
                         "an empty verdict. Default keeps config.yaml's value (no effect on Sonnet).")
    ap.add_argument("--request-timeout", type=float, default=None,
                    help="override per-attempt API request timeout (seconds). Raise for slow "
                         "thinking models whose full-prompt responses exceed the 30s default "
                         "(e.g. GLM 5.2 at 120) — otherwise they time out, fail, and retry-storm.")
    ap.add_argument("--regime-key", default=None,
                    help="override the regime-timeline cache key (default '<model>_<days>d'). "
                         "Point every arena model at ONE key (e.g. 'sonnet_250d') so they share "
                         "the IDENTICAL regime brief and skip a per-model regime rebuild.")
    ap.add_argument("--regime-model", default=None,
                    help="model used to BUILD any missing regime briefs (default = analysis "
                         "model). Set 'sonnet' for the arena so regime context is identical across "
                         "entrants. Accepts a MODELS short name or a full id.")
    ap.add_argument("--regime-lookback", type=float, default=90.0,
                    help="days of pre-window news fetched (regime history; >=14 needed for replay)")
    ap.add_argument("--no-live-replay", action="store_true",
                    help="use the old single fixed-regime backtest instead of the live-faithful "
                         "progressive replay (regime auto-refresh + catalyst memory)")
    ap.add_argument("--bridge-file", default=None,
                    help="override context_bridge_file for this replay only (A/B a knowledge-"
                         "bridge draft without touching config.yaml / the live bot; pair with "
                         "--cache-tag — prompt changes invalidate the analysis cache)")
    args = ap.parse_args()

    cfg = Config()
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing()))
        return
    if args.threshold is not None:
        cfg.app.risk.confidence_threshold = args.threshold
    if args.whitelist is not None:
        cfg.app.filters.source_whitelist = [s.strip() for s in args.whitelist.split(",") if s.strip()]
        print(f"Source whitelist: {cfg.app.filters.source_whitelist}")
    if args.bridge_file is not None:
        cfg.app.context_bridge_file = args.bridge_file
        print(f"Context bridge override: {args.bridge_file}")
    if args.max_tokens is not None:
        cfg.app.analyzer.max_tokens = args.max_tokens
        print(f"max_tokens override: {args.max_tokens}")
    if args.request_timeout is not None:
        cfg.app.analyzer.request_timeout_seconds = args.request_timeout
        print(f"request_timeout override: {args.request_timeout}s")

    model_full = MODELS.get(args.model, args.model)
    regime_model_full = (MODELS.get(args.regime_model, args.regime_model)
                         if args.regime_model else None)

    def tag(path: str) -> str:
        return path.replace(".json", f"_{args.cache_tag}.json") if args.cache_tag else path

    cache_path = f"data/bt_cache_{args.model}_{args.source}.json"
    if args.model == "haiku" and args.source == "tree":
        cache_path = "data/bt_analysis_cache.json"      # reuse original cache
    elif args.source == "tree":
        cache_path = f"data/bt_cache_{args.model}.json"  # reuse existing per-model tree cache
    cache_path = tag(cache_path)
    used_cache = cache_path

    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()

    # Curated Telegram channels are already filtered, so analyze every message
    # (no ticker prefilter) unless the user explicitly opts out.
    items = None
    prefilter = not args.no_prefilter
    regime_context = None
    res = None
    if args.source == "telegram":
        # Load-or-fetch the window PLUS lookback (history for the progressive regime).
        # With --history-file: load the dump if it exists (NO Telethon fetch — required
        # while the live bot holds tg_session; two processes on one session can kill the
        # auth key), else fetch once and save it. Loading is field-tolerant so dumps
        # written by other checkouts keep working.
        hist_path = pathlib.Path(args.history_file) if args.history_file else None
        if hist_path and hist_path.exists():
            from dataclasses import fields as dc_fields

            from hlbot.models import NewsItem
            _NI = {f.name for f in dc_fields(NewsItem)}
            all_items = sorted((NewsItem(**{k: v for k, v in d.items() if k in _NI})
                                for d in json.loads(hist_path.read_text(encoding="utf-8"))),
                               key=lambda i: i.time_ms)
            print(f"Loaded {len(all_items)} Telegram items from {hist_path} (no Telethon fetch)")
        else:
            # Enforce CLAUDE.md's rule at runtime, not just in help text: NEVER live-fetch
            # Telethon while a bot process holds tg_session (concurrent use of one session
            # can kill the auth key). Best-effort check; proceeds if pgrep is unavailable.
            try:
                import subprocess
                running = subprocess.run(
                    ["pgrep", "-f", "hlbot.main"], capture_output=True, text=True
                ).stdout.strip()
            except Exception:  # noqa: BLE001 - e.g. Windows dev box; no check possible
                running = ""
            if running:
                print("REFUSING to live-fetch Telegram: a bot process (hlbot.main*) is running "
                      f"(pid {running.splitlines()[0]}) and owns data/tg_session.\n"
                      "Use --history-file <dump.json> (fetch it once while the bot is stopped).")
                return
            all_items = await fetch_telegram_history(
                cfg, cfg.app.telegram_channels, args.days + args.regime_lookback)
            if hist_path and all_items:
                hist_path.write_text(json.dumps([i.to_dict() for i in all_items]),
                                     encoding="utf-8")
                print(f"Saved {len(all_items)} fetched items -> {hist_path}")
        if not all_items:
            print("No Telegram messages fetched (check login / channel usernames / membership).")
            return
        if not args.no_live_replay:
            # Live-faithful: regime auto-refreshes as the window is walked + catalyst memory.
            print(f"Live replay: {len(all_items)} raw items | window {args.days:.0f}d | "
                  f"regime every {cfg.app.regime_refresh_seconds/3600:.0f}h / "
                  f"{cfg.app.regime_live_lookback_days:.0f}d trailing | "
                  f"catalyst memory {cfg.app.catalyst_memory_days:.0f}d")
            used_cache = tag(f"data/bt_cache_replay_{args.model}.json")
            res = await run_live_replay(
                cfg, hl, universe, raw_items=all_items, days=args.days, model=model_full,
                cache_path=used_cache,
                regime_refresh_seconds=cfg.app.regime_refresh_seconds,
                regime_lookback_days=cfg.app.regime_live_lookback_days,
                catalyst_memory_days=cfg.app.catalyst_memory_days,
                regime_key=(args.regime_key or f"{args.model}_{int(args.days)}d"),
                regime_model=regime_model_full)
        else:
            win_cutoff_ms = (time.time() - args.days * 86400) * 1000
            items = [i for i in all_items if i.time_ms >= win_cutoff_ms]
            pre = [i for i in all_items if i.time_ms < win_cutoff_ms]
            print(f"Fetched {len(items)} window messages (+{len(pre)} pre-window)")
            if not items:
                print("No Telegram messages in window."); return
            prefilter = False
            if not args.no_regime:
                key = f"{args.model}_{args.source}_{int(args.days)}d"
                regime_context = await load_or_build_regime(cfg, pre, key,
                                                            rebuild=args.rebuild_regime)
                print(f"Regime brief: {len(regime_context)} chars")

    if res is None:
        res = await run_backtest(cfg, hl, universe, limit=args.limit, days=args.days,
                                 prefilter=prefilter, concurrency=args.concurrency,
                                 cache_path=cache_path, model=model_full, items=items,
                                 regime_context=regime_context)
    if "error" in res:
        print("Backtest error:", res["error"])
        return
    if not args.no_would_pnl:
        nm = await annotate_would_pnl(res, hl, universe, cfg, used_cache)
        print(f"  would-be PnL computed for {nm} skipped reads (conf>=50%)")
    report(res)
    if args.focus:
        await focus(res, hl, args.focus)

    report_path = write_html(res, f"data/backtest_report_{args.model}_{args.source}.html")
    print(f"\n  HTML review -> {report_path}")
    if not args.no_archive:
        snap = archive_run(res, cfg, report_path, used_cache)
        print(f"  archived run -> {snap}  (last 5 kept; compare with scripts/compare_runs.py)")
    if not args.no_open:
        import webbrowser
        webbrowser.open(pathlib.Path(report_path).as_uri())


if __name__ == "__main__":
    asyncio.run(main())
