"""Earnings classification benchmark: large-n test of the analyzer's earnings judgment
on EXTERNAL historical data (stockanalysis.com actual-vs-estimate history + Yahoo daily
reactions), independent of the Telegram window.

WHY: the replay backtest contains ~14 traded earnings events — far too few to validate
any earnings-related change without overfitting. This harness rebuilds trad_fin-style
headlines ("*NVIDIA 3Q ADJ EPS $1.30, EST. $1.18 *NVIDIA 3Q REV. $35.08B, EST. $33.16B
$NVDA") for hundreds of historical prints of the names the bot actually trades, runs
them through the PRODUCTION analyzer path (same prompt, same strict tool), and scores
direction/abstain/confidence against the realized next-session (r1) and 3-session (r3)
reactions.

WHAT IT IS NOT: a PnL backtest. No guidance lines (the single biggest omission — many
reactions are guidance-driven), no regime brief, vendor consensus != the IBES numbers
trad_fin prints, daily closes != perp marks, and the model may recognize famous prints
from training (slice by era to check). Read it as "classification + calibration at scale",
not expected dollars.

    python scripts/earnings_bench.py                 # fetch + analyze + score (~$5)
    python scripts/earnings_bench.py --score-only    # re-score from caches, no API calls
    python scripts/earnings_bench.py --limit 50      # cap analyzer calls (cost control)

Analyses cache to data/bt_cache_earnings_bench.json (keyed bench:<ticker>:<date>) — a
PROMPT change requires a fresh cache path, same rule as the replay. Event/price data
cache to data/earnings_bench/.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.analyzer import Analyzer            # noqa: E402
from hlbot.analysis.universe import Universe            # noqa: E402
from hlbot.backtest.engine import AnalysisCache, BACKTEST_FRESH_AGE_S, walk_candles  # noqa: E402
from hlbot.backtest.exit_variants import (  # noqa: E402
    ARMED_TRAIL_GRID, FAR_TP_GRID, RATCHET_GRID, SCALE_OUT_GRID, walk_variant,
)
from hlbot.config import Config                         # noqa: E402
from hlbot.models import NewsItem                       # noqa: E402
from hlbot.trading.hl_client import HLClient            # noqa: E402

CACHE_DIR = pathlib.Path("data/earnings_bench")
ANALYSIS_CACHE = "data/bt_cache_earnings_bench.json"   # default; --cache for prompt A/Bs
RESULTS_CSV = "data/earnings_bench_results.csv"        # default; --results for prompt A/Bs

# Names the bot actually trades (xyz universe), trad_fin-style company names.
TICKERS = {
    "NVDA": "NVIDIA", "AMD": "AMD", "INTC": "INTEL", "MU": "MICRON",
    "PLTR": "PALANTIR", "TSLA": "TESLA", "MSFT": "MICROSOFT", "AAPL": "APPLE",
    "GOOGL": "ALPHABET", "AMZN": "AMAZON", "META": "META", "AVGO": "BROADCOM",
    "ORCL": "ORACLE", "CRM": "SALESFORCE", "NFLX": "NETFLIX", "COIN": "COINBASE",
    "HOOD": "ROBINHOOD", "MSTR": "STRATEGY", "TSM": "TSMC", "DELL": "DELL",
    "SMCI": "SUPER MICRO", "MRVL": "MARVELL", "LITE": "LUMENTUM",
    "HIMS": "HIMS & HERS", "ARM": "ARM HOLDINGS",
}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as f:
        return json.load(f)


def fetch_events(ticker: str, since: str) -> list[dict]:
    """Confirmed historical prints with actual+estimate EPS, newest-first from the API;
    returned oldest-first. Disk-cached (refetch by deleting data/earnings_bench/)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"events_{ticker}.json"
    if p.exists():
        rows = json.loads(p.read_text(encoding="utf-8"))
    else:
        rows = _get_json(f"https://stockanalysis.com/api/symbol/s/{ticker.lower()}/earnings")["data"]
        p.write_text(json.dumps(rows), encoding="utf-8")
    out = [r for r in rows
           if r.get("confirmed") and r.get("eps_actual") is not None
           and r.get("eps_est") is not None and (r.get("date") or "") >= since]
    return sorted(out, key=lambda r: r["date"])


def fetch_daily(ticker: str) -> list[tuple[int, float]]:
    """Full daily close history 2021->now from Yahoo, disk-cached. [(open_ts, close)]."""
    p = CACHE_DIR / f"yahoo_{ticker}.json"
    if p.exists():
        return [tuple(x) for x in json.loads(p.read_text(encoding="utf-8"))]
    t0 = int(dt.datetime(2021, 10, 1, tzinfo=dt.timezone.utc).timestamp())
    t1 = int(dt.datetime.now(dt.timezone.utc).timestamp())
    j = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                  f"?period1={t0}&period2={t1}&interval=1d")
    res = j["chart"]["result"][0]
    bars = [(t, c) for t, c in zip(res["timestamp"],
                                   res["indicators"]["quote"][0]["close"]) if c is not None]
    p.write_text(json.dumps(bars), encoding="utf-8")
    return bars


def label_reaction(bars: list[tuple[int, float]], date: str, time_s: str) -> tuple[float, float] | None:
    """(r1, r3) close-to-close reactions. AMC (>=15:00 ET, the norm) reacts next session;
    BMO (<10:00 ET) reacts same session. Daily bar timestamps are session opens."""
    hour = int((time_s or "16:00")[:2])
    amc = hour >= 15 or 10 <= hour < 15   # mid-session prints are rare; treat as AMC
    bmo = hour < 10
    dates = [dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d") for t, _ in bars]
    # index of the print's session (last session on/before the print date)
    i = max((k for k, d in enumerate(dates) if d <= date), default=-1)
    if i < 0:
        return None
    prior_i, react_i = (i, i + 1) if amc else (i - 1, i)
    if bmo and dates[i] != date:        # BMO on a non-trading day: react next session
        prior_i, react_i = i, i + 1
    if prior_i < 0 or react_i >= len(bars):
        return None
    prior = bars[prior_i][1]
    r1 = bars[react_i][1] / prior - 1
    r3 = bars[min(react_i + 2, len(bars) - 1)][1] / prior - 1
    return r1, r3


def _eps_fmt(v: float) -> str:
    if v < 0:
        return f"LOSS/SHR ${abs(v):.2f}"
    return f"{round(v * 100)}C" if abs(v) < 1 else f"${v:.2f}"


def _rev_fmt(v: float) -> str:
    return f"${v / 1e9:.2f}B" if abs(v) >= 1e9 else f"${v / 1e6:.0f}M"


def headline(ticker: str, ev: dict) -> str:
    name = TICKERS[ticker]
    q = f"{(ev.get('period') or 'Q?')[-1]}Q"
    parts = [f"*{name} {q} ADJ EPS {_eps_fmt(ev['eps_actual'])}, EST. {_eps_fmt(ev['eps_est'])}"]
    if ev.get("revenue_actual") and ev.get("revenue_est"):
        parts.append(f"*{name} {q} REV. {_rev_fmt(ev['revenue_actual'])}, "
                     f"EST. {_rev_fmt(ev['revenue_est'])}")
    return " ".join(parts) + f" ${ticker}"


async def analyze_all(events: list[dict], concurrency: int, cache_path: str,
                      model: str = "", max_tokens: int = 0) -> list[dict]:
    cfg = Config()
    if model:   # model A/B (e.g. haiku): pair with FRESH --cache/--results paths
        cfg.app.analyzer.model_fast = model
    if max_tokens:   # thinking models (DeepSeek V4 Pro / GLM) need >=2048 or they truncate->empty
        cfg.app.analyzer.max_tokens = max_tokens
    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()
    analyzer = Analyzer(cfg)
    if analyzer.exemplars.strip():
        print(f"  NOTE: exemplars block ACTIVE ({len(analyzer.exemplars)} chars) — "
              f"make sure --cache/--results point at fresh paths")
    cache = AnalysisCache(cache_path)
    sem = asyncio.Semaphore(concurrency)
    n_new = 0

    async def one(ev: dict) -> dict:
        nonlocal n_new
        nid = f"bench:{ev['ticker']}:{ev['date']}"
        a = cache.get(nid)
        if a is None:
            item = NewsItem(id=nid, title="tradfi", body=ev["headline"],
                            source="Telegram:trad_fin", link=None,
                            time_ms=ev["time_ms"], received_ms=ev["time_ms"])
            async with sem:
                a = await analyzer.analyze(item, universe, age_seconds=BACKTEST_FRESH_AGE_S)
            cache.put(a)
            n_new += 1
            if n_new % 25 == 0:
                cache.save()
                print(f"  analyzed {n_new} new ...")
        return {**ev, "direction": a.direction, "confidence": a.confidence,
                "a_ticker": a.ticker, "time_sensitivity": a.time_sensitivity,
                "is_stale": a.is_stale, "rationale": a.rationale}

    out = await asyncio.gather(*(one(e) for e in events))
    cache.save()
    print(f"  analyzer calls: {n_new} new, {len(events) - n_new} cached")
    return list(out)


def fetch_hourly(ticker: str) -> list[dict]:
    """1h candles WITH pre/post-market, chunked back ~725 days (Yahoo's 1h depth limit),
    disk-cached. Engine candle format: {t, T, o, h, l, c} in ms. After-hours bars are what
    make AMC prints time-matchable — the bot's perp entry happens in that session."""
    p = CACHE_DIR / f"yahoo1h_{ticker}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    out: list[dict] = []
    for k in range(5):                       # 5 x 145d chunks ~ 725d
        t0, t1 = now - (k + 1) * 145 * 86400, now - k * 145 * 86400
        try:
            j = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                          f"?period1={t0}&period2={t1}&interval=1h&includePrePost=true")
            res = j["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            for t, o, h, l, c in zip(res.get("timestamp", []), q["open"], q["high"],
                                     q["low"], q["close"]):
                if None in (o, h, l, c):
                    continue
                out.append({"t": t * 1000, "T": (t + 3600) * 1000, "o": o, "h": h,
                            "l": l, "c": c})
        except Exception:  # noqa: BLE001  (chunk beyond history -> skip)
            continue
    out.sort(key=lambda b: b["t"])
    dedup = {b["t"]: b for b in out}
    out = [dedup[t] for t in sorted(dedup)]
    p.write_text(json.dumps(out), encoding="utf-8")
    return out


def _print_ts_ms(date: str, time_s: str) -> int:
    from zoneinfo import ZoneInfo
    h, m = int((time_s or "16:00")[:2]), int((time_s or "16:00")[3:5])
    et = dt.datetime.fromisoformat(date).replace(hour=h, minute=m,
                                                 tzinfo=ZoneInfo("America/New_York"))
    return int(et.timestamp() * 1000)


_TD_IV_MS = {"1min": 60_000, "5min": 300_000}
_td_last_call = [0.0]


def fetch_alpaca_intraday(ticker: str, date: str, key: str, secret: str,
                          interval: str) -> list[dict]:
    """Alpaca v2 minute bars (IEX feed, free tier) WITH extended hours for one earnings
    event: print-day 04:00 ET through +4 days 20:00 ET. Bars exist wherever IEX printed
    trades — dense on megacaps (especially post-earnings AH, the most active AH sessions),
    sparser on small names. split-adjusted. Disk-cached per (interval, ticker, date)."""
    import time as _time
    from zoneinfo import ZoneInfo
    tf = {"1min": "1Min", "5min": "5Min"}[interval]
    p = CACHE_DIR / f"alp_{interval}_{ticker}_{date}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    tz = ZoneInfo("America/New_York")
    d0 = dt.datetime.fromisoformat(date).replace(hour=4, tzinfo=tz)
    d1 = (d0 + dt.timedelta(days=4)).replace(hour=20)
    iv = _TD_IV_MS[interval]
    out, token = [], ""
    while True:
        url = ("https://data.alpaca.markets/v2/stocks/" + ticker + "/bars"
               f"?timeframe={tf}&adjustment=split&feed=iex&limit=10000"
               f"&start={d0.isoformat().replace('+', '%2B')}"
               f"&end={d1.isoformat().replace('+', '%2B')}"
               + (f"&page_token={token}" if token else ""))
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        with urllib.request.urlopen(req, timeout=20) as f:
            j = json.load(f)
        for b in j.get("bars") or []:
            t = int(dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp() * 1000)
            out.append({"t": t, "T": t + iv, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]})
        token = j.get("next_page_token")
        if not token:
            break
        _time.sleep(0.05)
    out.sort(key=lambda b: b["t"])
    p.write_text(json.dumps(out), encoding="utf-8")
    _time.sleep(0.31)            # stay under the free tier's ~200 req/min
    return out


def fetch_td_intraday(ticker: str, date: str, api_key: str, interval: str) -> list[dict]:
    """Twelve Data intraday bars for one earnings event (print-day 04:00 ET -> +4d 20:00
    ET). NOTE: prepost (extended hours) is PRO-plan-only on Twelve Data — the free tier
    serves regular hours only, so this is kept as an RTH cross-check source; the primary
    minute source for the exit walk is fetch_alpaca_intraday (IEX, AH included).
    Free tier: 8 req/min, 800/day. Disk-cached per (interval, ticker, date)."""
    import time as _time
    from zoneinfo import ZoneInfo
    p = CACHE_DIR / f"td_{interval}_{ticker}_{date}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    end = (dt.date.fromisoformat(date) + dt.timedelta(days=4)).isoformat()
    url = ("https://api.twelvedata.com/time_series"
           f"?symbol={ticker}&interval={interval}&order=ASC&outputsize=5000"
           f"&start_date={date}%2004:00:00&end_date={end}%2020:00:00&apikey={api_key}")
    for attempt in (1, 2):
        wait = 60.0 / 8 + 0.1 - (_time.monotonic() - _td_last_call[0])
        if wait > 0:
            _time.sleep(wait)                 # free-tier rate limit: 8 credits/min
        _td_last_call[0] = _time.monotonic()
        j = _get_json(url)
        if j.get("status") == "ok" or "values" in j:
            break
        if int(j.get("code", 0)) == 429 and attempt == 1:
            _time.sleep(62)                   # minute-quota exhausted: wait it out once
            continue
        raise RuntimeError(f"TwelveData {ticker} {date}: {str(j)[:140]}")
    iv = _TD_IV_MS[interval]
    tz = ZoneInfo("America/New_York")
    out = []
    for v in j.get("values", []):
        t = int(dt.datetime.fromisoformat(v["datetime"]).replace(tzinfo=tz).timestamp() * 1000)
        out.append({"t": t, "T": t + iv, "o": float(v["open"]), "h": float(v["high"]),
                    "l": float(v["low"]), "c": float(v["close"])})
    out.sort(key=lambda b: b["t"])
    p.write_text(json.dumps(out), encoding="utf-8")
    return out


def variant_keys() -> list[tuple[str, str, object]]:
    """Same labels as scripts/exit_lab.py so replay and bench tables line up."""
    keys: list[tuple[str, str, object]] = []
    for frac, tp1 in SCALE_OUT_GRID:
        keys.append(("scale_out", f"scale_out {int(frac*100)}%@+{tp1:.0%}", (frac, tp1)))
    for tiers in RATCHET_GRID:
        lbl = "->".join(f"{tr:.0%}@{arm:.0%}" for arm, tr in tiers)
        keys.append(("ratchet", f"ratchet {lbl}", tiers))
    for tiers in ARMED_TRAIL_GRID:
        arm, tr = tiers[0]
        keys.append(("armed", f"armed {tr:.0%} trail @ MFE>={arm:.0%}", tiers))
    for tp in FAR_TP_GRID:
        keys.append(("far_tp", f"far_tp +{tp:.0%} & 8% trail", tp))
    return keys


def score_variants(rows: list[dict], keys: list[tuple[str, str, object]]) -> None:
    """Variant exits vs production on the bench gate-passers (pess entries, gross)."""
    g = [r for r in rows if r["confidence"] >= 0.80 and "capture_pess" in r]
    if not g:
        print("\n=== variants: no gate-passers with intraday data ===")
        return
    base = sum(r["capture_pess"] for r in g)
    bw = sum(r["capture_pess"] for r in g if r["capture_pess"] > 0)
    bl = -sum(r["capture_pess"] for r in g if r["capture_pess"] <= 0)
    print(f"\n=== EXIT VARIANTS on bench gate-passers (n={len(g)}, pess entry, gross) ===")
    print(f"{'variant':38} {'total pp':>9} {'d pp':>8} {'PF':>5} {'win%':>5}")
    print(f"{'production':38} {base*100:>+9.0f} {'':>8} {bw/bl if bl else 0:>5.2f} "
          f"{sum(1 for r in g if r['capture_pess']>0)/len(g)*100:>4.0f}%")
    for _fam, key, _p in keys:
        col = f"v::{key}"
        caps = [r.get(col, r["capture_pess"]) for r in g]
        gw = sum(c for c in caps if c > 0)
        gl = -sum(c for c in caps if c <= 0)
        print(f"{key:38} {sum(caps)*100:>+9.0f} {(sum(caps)-base)*100:>+8.0f} "
              f"{gw/gl if gl else 0:>5.2f} {sum(1 for c in caps if c>0)/len(caps)*100:>4.0f}%")


def exit_walk(rows: list[dict], interval: str = "1h", api_key: str = "",
              variants: list[tuple[str, str, object]] | None = None) -> list[dict]:
    """Time-matched exit simulation with the bot's ACTUAL per-sensitivity exits.

    interval "1h": Yahoo prepost hourly (~725d depth); entry = first bar after the print
    (up to ~60min late -> bracket with the pre-print bound). interval "1min"/"5min":
    Twelve Data prepost bars back to 2022 covering ALL events; entry = first bar opening
    >= print + 60s (the bot's realistic wire->order latency), so the pess/opt bracket
    nearly collapses. Adds exit_reason / capture_pct / entry_lag_min per row."""
    r = Config().app.risk
    times = {}
    for tkr in TICKERS:
        p = CACHE_DIR / f"events_{tkr}.json"
        if p.exists():
            for ev in json.loads(p.read_text(encoding="utf-8")):
                times[(tkr, ev["date"])] = ev.get("time") or "16:00"
    bars_by_tkr: dict[str, list[dict]] = {}
    out, skipped, fetch_fail = [], 0, 0
    for row in rows:
        if row["direction"] not in ("long", "short"):
            continue
        tkr = row["ticker"]
        ts = row.get("time_sensitivity") or "days"
        if ts not in ("immediate", "hours", "days"):
            ts = "days"
        pms = _print_ts_ms(row["date"], times.get((tkr, row["date"]), "16:00"))
        if interval == "1h":
            if tkr not in bars_by_tkr:
                try:
                    bars_by_tkr[tkr] = fetch_hourly(tkr)
                except Exception:  # noqa: BLE001
                    bars_by_tkr[tkr] = []
            bars = bars_by_tkr[tkr]
            entry_cut = pms
        else:
            try:
                bars = fetch_alpaca_intraday(tkr, row["date"], api_key[0], api_key[1],
                                             interval)
            except Exception as exc:  # noqa: BLE001
                fetch_fail += 1
                if fetch_fail <= 3:
                    print(f"  ({tkr} {row['date']}: {str(exc)[:90]})")
                continue
            entry_cut = pms + 60_000          # bot latency: ~60s after the wire line
        i = next((k for k, b in enumerate(bars) if b["t"] >= entry_cut), None)
        if i is None or not bars or bars[0]["t"] > pms:   # print predates bar history
            skipped += 1
            continue
        if interval != "1h" and bars[i]["t"] - pms > 120 * 60_000:
            skipped += 1     # no usable bars near the print (thin IEX AH) — measurement
            continue         # failure, not bot behavior; don't fake a next-morning entry
        stop = r.stop_loss_by_sensitivity.get(ts, r.stop_loss_pct)
        trail = r.trail_pct_by_sensitivity.get(ts, 0.0)
        tp = 0.0 if trail > 0 else r.take_profit_pct
        sgn = 1 if row["direction"] == "long" else -1
        rec = dict(row)
        # The bot enters ~1min after the print; hourly bars can't represent that, so
        # bracket it: PESSIMISTIC = next bar open (up to ~60min late, post-pop);
        # OPTIMISTIC = print-bar open (pre-print price, walk includes the print bar).
        for mode, j in (("pess", i), ("opt", max(i - 1, 0))):
            entry = bars[j]["o"]
            stop_px = entry * (1 - sgn * stop)
            tp_px = entry * (1 + sgn * tp) if tp else 0.0
            w = walk_candles(row["direction"], entry, stop_px, tp_px, trail, bars, j,
                             pms + r.exit_horizons.get(ts, 259200) * 1000)
            rec[f"exit_reason_{mode}"] = w.reason
            rec[f"capture_{mode}"] = sgn * (w.exit_px / entry - 1)
        rec["exit_reason"] = rec["exit_reason_pess"]
        rec["capture_pct"] = rec["capture_pess"]
        rec["entry_lag_min"] = (bars[i]["t"] - pms) / 60000
        if variants:
            # Variant exit shapes on the SAME bars, realistic (pess) entry only. Same
            # composition as scripts/exit_lab.py; applies where production trails.
            entry = bars[i]["o"]
            stop_px = entry * (1 - sgn * stop)
            h_end = pms + r.exit_horizons.get(ts, 259200) * 1000
            prod_tiers = ((0.0, trail),)
            for fam, key, p in variants:
                if trail <= 0:                       # immediate: production TP exits
                    rec[f"v::{key}"] = rec["capture_pess"]
                    continue
                if fam == "scale_out":
                    frac, tp1 = p
                    l1 = walk_variant(row["direction"], entry, stop_px, bars, i, h_end,
                                      tp_px=entry * (1 + sgn * tp1))
                    l2 = walk_variant(row["direction"], entry, stop_px, bars, i, h_end,
                                      trail_tiers=prod_tiers)
                    rec[f"v::{key}"] = (frac * sgn * (l1.exit_px / entry - 1)
                                        + (1 - frac) * sgn * (l2.exit_px / entry - 1))
                elif fam in ("ratchet", "armed"):
                    w2 = walk_variant(row["direction"], entry, stop_px, bars, i, h_end,
                                      trail_tiers=p)
                    rec[f"v::{key}"] = sgn * (w2.exit_px / entry - 1)
                elif fam == "far_tp":
                    w2 = walk_variant(row["direction"], entry, stop_px, bars, i, h_end,
                                      trail_tiers=prod_tiers, tp_px=entry * (1 + sgn * p))
                    rec[f"v::{key}"] = sgn * (w2.exit_px / entry - 1)
        out.append(rec)
    if skipped:
        print(f"  (exit-walk: {skipped} events predate the {interval} bar history)")
    if fetch_fail:
        print(f"  (exit-walk: {fetch_fail} events failed to fetch from Alpaca)")
    return out


def score_exits(rows: list[dict], label: str) -> None:
    """Exit-aware scorecard over the time-matched subset (gate-passers), reporting the
    pessimistic (next-bar entry) and optimistic (pre-print entry) bounds side by side —
    the bot's real ~1min entry lies between them."""
    g = [r for r in rows if r["confidence"] >= 0.80]
    if not g:
        print(f"\n=== {label}: no gate-passers with intraday data ===")
        return
    import collections
    print(f"\n=== {label} (time-matched, bot exits, GROSS of fees) ===")
    for mode in ("pess", "opt"):
        cap = f"capture_{mode}"
        reasons = collections.Counter(r[f"exit_reason_{mode}"].replace(" (gap)", "")
                                      for r in g)
        wins = [r for r in g if r[cap] > 0]
        gross_w = sum(r[cap] for r in wins)
        gross_l = -sum(r[cap] for r in g if r[cap] <= 0)
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        tag = "entry next bar (pess)" if mode == "pess" else "entry pre-print (opt)"
        print(f"  [{tag}] n={len(g)}  win {len(wins) / len(g) * 100:.0f}%  PF {pf:.2f}  "
              f"avg {sum(r[cap] for r in g) / len(g) * 100:+.2f}%  "
              f"total {sum(r[cap] for r in g) * 100:+.0f}pp")
        print(f"    exits: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
        for nm, cell in [("defect cell (0.85+ x 5-15%)",
                          [r for r in g if r["confidence"] >= 0.85
                           and 5 <= abs(r["eps_surprise_pct"]) < 15]),
                         ("edge cell (0.90+ x >=15%)",
                          [r for r in g if r["confidence"] >= 0.90
                           and abs(r["eps_surprise_pct"]) >= 15])]:
            if cell:
                cw = sum(1 for r in cell if r[cap] > 0) / len(cell)
                print(f"    {nm}: n={len(cell)} win {cw * 100:.0f}% "
                      f"avg {sum(r[cap] for r in cell) / len(cell) * 100:+.2f}% "
                      f"total {sum(r[cap] for r in cell) * 100:+.0f}pp")


def bucket(x: float, edges: list[float], labels: list[str]) -> str:
    for e, lab in zip(edges, labels):
        if x < e:
            return lab
    return labels[-1]


def score(rows: list[dict]) -> None:
    scored = [r for r in rows if r.get("r1") is not None]
    print(f"\n{'=' * 74}\n EARNINGS BENCH  ·  {len(scored)} prints scored "
          f"({len(rows) - len(scored)} unlabelable)\n{'=' * 74}")
    directional = [r for r in scored if r["direction"] in ("long", "short")]
    abstain = len(scored) - len(directional)
    print(f" directional: {len(directional)}  ·  abstain(none): {abstain} "
          f"({abstain / max(len(scored), 1) * 100:.0f}%)")

    def hit(r, k):  # direction matches reaction sign
        return (1 if r[k] > 0 else -1) == (1 if r["direction"] == "long" else -1)

    def table(title, groups):
        print(f"\n=== {title} ===")
        print(f"  {'':22s} {'n':>4} {'hit r1':>6} {'hit r3':>6} {'avg s·r1':>9} {'avg s·r3':>9}")
        for lab, sub in groups:
            if not sub:
                continue
            h1 = sum(hit(r, 'r1') for r in sub) / len(sub)
            h3 = sum(hit(r, 'r3') for r in sub) / len(sub)
            s1 = sum((1 if r['direction'] == 'long' else -1) * r['r1'] for r in sub) / len(sub)
            s3 = sum((1 if r['direction'] == 'long' else -1) * r['r3'] for r in sub) / len(sub)
            print(f"  {lab:22s} {len(sub):>4} {h1 * 100:>5.0f}% {h3 * 100:>5.0f}% "
                  f"{s1 * 100:>+8.2f}% {s3 * 100:>+8.2f}%")

    gate = [r for r in directional if r["confidence"] >= 0.80]
    table("HEADLINE: GATE-PASSERS (conf >= 0.80) vs all directional",
          [("gate-passers (>=0.80)", gate), ("all directional", directional)])

    # naive baseline the user asked about: beat = long, miss = short, always
    naive = [{**r, "direction": "long" if r["eps_surprise_pct"] > 0 else "short"} for r in scored]
    table("NAIVE 'beat=long miss=short' BASELINE (every print)", [("naive all", naive)])

    conf_groups = [(lab, [r for r in directional
                          if bucket(r["confidence"], [0.7, 0.8, 0.85, 0.9],
                                    ["<0.70", "0.70-0.80", "0.80-0.85", "0.85-0.90", "0.90+"]) == lab])
                   for lab in ["<0.70", "0.70-0.80", "0.80-0.85", "0.85-0.90", "0.90+"]]
    table("CALIBRATION: does confidence rank-order outcomes?", conf_groups)

    surp = lambda r: abs(r["eps_surprise_pct"])
    sg = [(lab, [r for r in gate if bucket(surp(r), [2, 5, 15, 40],
                                           ["<2%", "2-5%", "5-15%", "15-40%", ">=40%"]) == lab])
          for lab in ["<2%", "2-5%", "5-15%", "15-40%", ">=40%"]]
    table("GATE-PASSERS BY |EPS SURPRISE| (is big surprise where the edge is?)", sg)

    eras = sorted({r["date"][:4] for r in scored})
    table("GATE-PASSERS BY ERA (training-leakage check: pre-cutoff years suspiciously good?)",
          [(y, [r for r in gate if r["date"][:4] == y]) for y in eras])

    table("GATE-PASSERS BY TICKER (top losers/winners need eyeballing)",
          sorted([(t, [r for r in gate if r["ticker"] == t]) for t in TICKERS],
                 key=lambda kv: -len(kv[1]))[:12])

    print("\n CAVEATS: no guidance lines (often the real driver), vendor consensus != IBES,")
    print(" daily closes != perp fills, famous prints may be in model training data.")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2022-01-01")
    ap.add_argument("--limit", type=int, default=0, help="cap analyzer calls (0 = all)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--score-only", action="store_true",
                    help="score from existing results CSV (no fetching/analysis)")
    ap.add_argument("--cache", default=ANALYSIS_CACHE,
                    help="analysis cache path (use a FRESH one for any prompt change)")
    ap.add_argument("--results", default=RESULTS_CSV, help="results CSV path")
    ap.add_argument("--exits", action="store_true",
                    help="time-matched exit simulation (prepost candles, bot exit params)")
    ap.add_argument("--intraday", default="1h", choices=["1h", "1min", "5min"],
                    help="bar source for --exits: 1h = Yahoo (~725d depth); 1min/5min = "
                         "Alpaca IEX incl. after-hours (needs ALPACA_API_KEY/SECRET in .env)")
    ap.add_argument("--variants", action="store_true",
                    help="with --exits: also walk the exit-lab variant grid "
                         "(scale-out / ratchet / armed trail / far TP) on the same bars")
    ap.add_argument("--model", default="",
                    help="analyzer model override (alias or full id) — pair with fresh "
                         "--cache and --results paths")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="override analyzer max_tokens (arena: 2048 so thinking models like "
                         "DeepSeek V4 Pro / GLM don't truncate their reasoning into empty output)")
    args = ap.parse_args()
    _ALIAS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6",
              "opus": "claude-opus-4-8"}
    if args.model:
        args.model = _ALIAS.get(args.model, args.model)

    td_key = ""
    if args.intraday != "1h":
        from hlbot.config import Secrets
        s = Secrets()
        if not (s.alpaca_api_key and s.alpaca_api_secret):
            print("ALPACA_API_KEY / ALPACA_API_SECRET missing from .env — free paper "
                  "account: https://alpaca.markets (minute bars incl. after-hours).")
            return
        td_key = (s.alpaca_api_key, s.alpaca_api_secret)

    if args.score_only:
        with open(args.results, encoding="utf-8") as f:
            rows = []
            for r in csv.DictReader(f):
                for k in ("confidence", "eps_surprise_pct"):
                    r[k] = float(r[k])
                for k in ("r1", "r3"):
                    r[k] = float(r[k]) if r[k] else None
                r["is_stale"] = r["is_stale"] == "True"
                rows.append(r)
        score(rows)
        if args.exits:
            vks = variant_keys() if args.variants else None
            walked = exit_walk(rows, args.intraday, td_key, variants=vks)
            score_exits(walked, f"EXIT-AWARE ({args.intraday}): {args.results}")
            if vks:
                score_variants(walked, vks)
        return

    events = []
    for tkr in TICKERS:
        try:
            evs = fetch_events(tkr, args.since)
            bars = fetch_daily(tkr)
        except Exception as exc:  # noqa: BLE001
            print(f"  {tkr}: fetch failed ({exc}) — skipped")
            continue
        for ev in evs:
            rx = label_reaction(bars, ev["date"], ev.get("time") or "16:00")
            t_ms = int(dt.datetime.fromisoformat(ev["date"] + "T21:00:00+00:00").timestamp() * 1000)
            events.append({
                "ticker": tkr, "date": ev["date"], "period": ev.get("period"),
                "eps_surprise_pct": 100.0 * (ev["eps_actual"] - ev["eps_est"])
                                    / max(abs(ev["eps_est"]), 1e-9),
                "headline": headline(tkr, ev), "time_ms": t_ms,
                "r1": rx[0] if rx else None, "r3": rx[1] if rx else None,
            })
        await asyncio.sleep(0.3)   # be polite to the free APIs
    print(f"events: {len(events)} prints across {len(TICKERS)} tickers since {args.since}")
    if args.limit:
        events = events[-args.limit:]
        print(f"  --limit {args.limit}: analyzing the most recent {len(events)}")

    rows = await analyze_all(events, args.concurrency, args.cache, model=args.model,
                             max_tokens=args.max_tokens)

    with open(args.results, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "date", "period", "eps_surprise_pct",
                                          "direction", "confidence", "time_sensitivity",
                                          "is_stale", "r1", "r3", "headline", "rationale"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"results -> {args.results}")
    score(rows)
    if args.exits:
        vks = variant_keys() if args.variants else None
        walked = exit_walk(rows, args.intraday, td_key, variants=vks)
        score_exits(walked, f"EXIT-AWARE ({args.intraday}): {args.results}")
        if vks:
            score_variants(walked, vks)


if __name__ == "__main__":
    asyncio.run(main())
