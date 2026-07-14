"""Pure aggregation helpers for the edge report: source expectancy + confidence calibration.

These operate on a backtest report's ``rows`` (see backtest/report.py): each row is a
directional read that either traded (has real ``pnl``) or was skipped (and may carry a
precomputed ``would_pnl`` from scripts/backtest.py ``annotate_would_pnl``). No network —
the slicing the HTML report doesn't do, so you can see WHICH sources and WHICH confidence
levels actually make money (and where the gate belongs). The single I/O helper is
``load_report`` (shared by edge_report / mine_exemplars / compare tooling). Unit-tested.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Optional

_TW_HANDLE = re.compile(r"\(@([A-Za-z0-9_]+)\)")


def load_report(path: str) -> dict:
    """Extract the embedded ``const DATA = {...}`` payload from a backtest report HTML."""
    html = pathlib.Path(path).read_text(encoding="utf-8")
    marker = "const DATA = "
    data, _ = json.JSONDecoder().raw_decode(html, html.index(marker) + len(marker))
    return data


def source_key(row: dict) -> str:
    """A whitelist-ready source label: the @handle for Twitter items (pulled from the title,
    like scripts/list_sources.py), the channel for Telegram, else the raw source."""
    src = (row.get("source") or "?").strip()
    if src == "Twitter":
        m = _TW_HANDLE.search(row.get("title") or "")
        return f"@{m.group(1)}" if m else "Twitter"
    return src or "?"


def market_type(row: dict) -> str:
    """Coarse asset bucket from the resolved dex: trade.xyz equities vs HL crypto perps."""
    return "xyz (equity/index/cmdty)" if (row.get("dex") or "").strip() == "xyz" else "crypto"


def outcome_pnl(row: dict) -> Optional[float]:
    """The trade outcome for a directional read: realized pnl if it actually traded, else the
    precomputed would-be pnl. None if neither is available (e.g. would_pnl wasn't computed)."""
    if row.get("status") == "traded" and row.get("pnl") is not None:
        return float(row["pnl"])
    if row.get("would_pnl") is not None:
        return float(row["would_pnl"])
    return None


def outcomes(rows: list[dict], min_conf: float = 0.0) -> list[dict]:
    """Flatten report rows to scored outcomes: directional reads with a known (real or would-be)
    pnl and confidence >= min_conf. ``confidence`` here is the effective (post-haircut) value the
    gate sees, so calibration and the gate sweep are apples-to-apples with live behavior."""
    out: list[dict] = []
    for r in rows:
        if r.get("direction") not in ("long", "short"):
            continue
        conf = r.get("confidence")
        if conf is None or conf < min_conf:
            continue
        pnl = outcome_pnl(r)
        if pnl is None:
            continue
        out.append({
            "conf": float(conf),
            "pnl": pnl,
            "traded": r.get("status") == "traded",
            "source": source_key(r),
            "market_type": market_type(r),
            "time_sensitivity": r.get("time_sensitivity") or "none",
            "ticker": (r.get("ticker") or "?").upper(),
            "time_ms": int(r.get("time_ms") or 0),
            "mae": r.get("mae_pct"),
            "mfe": r.get("mfe_pct"),
            "pre_move": r.get("pre_move_pct"),
            "pre_move_bucket": pre_move_bucket(r.get("pre_move_pct")),
        })
    return out


def pre_move_bucket(pm) -> str:
    """Bucket the in-direction pre-news->entry move for the already-moved-guard slices.
    Edges match the suggested guard thresholds (1.25% haircut / 3% reject)."""
    if pm is None:
        return "(unmeasured)"
    if pm <= 0:
        return "flat / against us"
    if pm < 0.0125:
        return "0 to +1.25%"
    if pm < 0.03:
        return "+1.25% to +3%"
    return ">= +3% (reject zone)"


def summarize(items: list[dict]) -> dict:
    """n / wins / win_rate / total & avg pnl / profit factor / how many actually traded.
    Profit factor (gross wins / gross losses) separates a +$900 slice with PF 1.05 from
    one with PF 3 — total alone can't."""
    n = len(items)
    wins = sum(1 for o in items if o["pnl"] > 0)
    total = sum(o["pnl"] for o in items)
    traded = sum(1 for o in items if o["traded"])
    gross_win = sum(o["pnl"] for o in items if o["pnl"] > 0)
    gross_loss = -sum(o["pnl"] for o in items if o["pnl"] < 0)
    return {"n": n, "wins": wins, "win_rate": (wins / n) if n else 0.0,
            "total": total, "avg": (total / n) if n else 0.0, "traded": traded,
            "gross_win": gross_win, "gross_loss": gross_loss,
            "pf": (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win else 0.0)}


def group_by(items: list[dict], key: str) -> list[tuple[str, dict]]:
    """Group outcomes by a key, summarized, sorted by total pnl descending."""
    buckets: dict[str, list] = {}
    for o in items:
        buckets.setdefault(o[key], []).append(o)
    rows = [(k, summarize(v)) for k, v in buckets.items()]
    rows.sort(key=lambda kv: kv[1]["total"], reverse=True)
    return rows


def calibration(items: list[dict], width: float = 0.05,
                lo: float = 0.5) -> list[tuple[tuple[float, float], dict]]:
    """Bucket outcomes by confidence into [lo, lo+width, ...) bins; return non-empty bins in
    ascending order with their summaries — the core 'is confidence monotonic with edge?' view.
    Anything below ``lo`` collapses into a single (0, lo) bin."""
    bins: dict[int, list] = {}
    for o in items:
        # +1e-9 so a value sitting exactly on a boundary (e.g. 0.60) lands in the bucket it
        # STARTS, not the one below it, despite float error in the subtraction/division.
        idx = int((o["conf"] - lo) / width + 1e-9) if o["conf"] >= lo else -1
        bins.setdefault(idx, []).append(o)
    out = []
    for idx in sorted(bins):
        rng = (0.0, lo) if idx < 0 else (lo + idx * width, lo + (idx + 1) * width)
        out.append((rng, summarize(bins[idx])))
    return out


def suggest_gate(items: list[dict], lo: float = 0.5, hi: float = 0.95, step: float = 0.01,
                 min_trades: int = 5) -> dict:
    """Sweep candidate gates; for each, summarize outcomes with conf >= gate, and report the
    gate that maximizes TOTAL pnl (subject to >= min_trades surviving). IN-SAMPLE and
    descriptive only — it overfits one window; suggest_gate_oos is the honest version."""
    best: dict = {}
    steps = int(round((hi - lo) / step)) + 1
    for i in range(steps):
        t = round(lo + i * step, 4)   # rounded so the sweep doesn't drift off clean thresholds
        s = summarize([o for o in items if o["conf"] >= t])
        if s["n"] >= min_trades and (not best or s["total"] > best["total"]):
            best = {"gate": t, **s}
    return best


def suggest_gate_oos(items: list[dict], train_frac: float = 0.7, lo: float = 0.5,
                     hi: float = 0.95, step: float = 0.01, min_trades: int = 5) -> dict:
    """Chronological out-of-sample gate check: fit the PnL-max gate on the FIRST
    train_frac of outcomes (by time), then report how that gate performs on the held-out
    later slice. A gate that only wins in-sample shows up as a weak/negative `val`.
    Returns {} when there's too little data to split."""
    ordered = sorted(items, key=lambda o: o["time_ms"])
    k = int(round(len(ordered) * train_frac))
    train, val = ordered[:k], ordered[k:]
    if len(train) < min_trades or not val:
        return {}
    best = suggest_gate(train, lo, hi, step, min_trades)
    if not best:
        return {}
    gate = best["gate"]
    return {"gate": gate,
            "train": summarize([o for o in train if o["conf"] >= gate]),
            "val": summarize([o for o in val if o["conf"] >= gate]),
            "n_train": len(train), "n_val": len(val)}


def gate_sweep_oos(items: list[dict], gates: list[float],
                   train_frac: float = 0.7) -> list[tuple[float, dict, dict]]:
    """Train-vs-validation summaries for each candidate gate (same chronological split as
    suggest_gate_oos) — shows whether the gate's edge is stable out-of-sample."""
    ordered = sorted(items, key=lambda o: o["time_ms"])
    k = int(round(len(ordered) * train_frac))
    train, val = ordered[:k], ordered[k:]
    return [(g,
             summarize([o for o in train if o["conf"] >= g]),
             summarize([o for o in val if o["conf"] >= g])) for g in gates]


def kelly_table(rows: list[dict], tiers: list, account_usd: float,
                max_notional: float, min_n: int = 20) -> list[dict]:
    """Quarter-Kelly sizing check per confidence tier, from TRADED rows only (real
    fills, real PnL — would-be PnL ignores caps/cooldowns and would flatter the edge).
    REPORT-ONLY: tiers with n < min_n get no suggestion; sizing changes on small
    samples are how a good 180 days buys a bad next 90. The suggested notional puts
    quarter-Kelly (f*/4) of the account at risk given the tier's empirical
    loss-per-dollar-of-notional, capped at max_notional."""
    edges = sorted(float(c) for c, _ in tiers)
    cur_by_lo = {float(c): float(n) for c, n in tiers}
    traded = [r for r in rows if r.get("status") == "traded"
              and r.get("pnl") is not None and r.get("confidence") is not None]
    out = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else 1.01
        sub = [r for r in traded if lo <= float(r["confidence"]) < hi]
        n = len(sub)
        row = {"lo": lo, "hi": hi, "n": n, "cur_notional": cur_by_lo[lo],
               "win": None, "kelly": None, "suggest": None}
        if n:
            pnls = [float(r["pnl"]) for r in sub]
            wins = [p for p in pnls if p > 0]
            losses = [-p for p in pnls if p <= 0]
            row["win"] = len(wins) / n
            if wins and losses:
                w_avg = sum(wins) / len(wins)
                # exact-0.0 PnLs count as losses; all-zero losses would divide by zero below
                l_avg = max(sum(losses) / len(losses), 1e-9)
                row["kelly"] = row["win"] - (1 - row["win"]) / (w_avg / l_avg)
                if n >= min_n:
                    notional_avg = sum(float(r["notional"]) for r in sub) / n
                    loss_frac = l_avg / max(notional_avg, 1e-9)
                    sugg = max(0.0, row["kelly"] / 4.0) * account_usd / max(loss_frac, 1e-9)
                    row["suggest"] = min(sugg, max_notional)
        out.append(row)
    return out


def excursion_stats(items: list[dict]) -> dict:
    """MAE/MFE percentiles for winners vs losers (traded rows that carry excursions).
    Directly answers stop/trail placement: 'how deep do eventual winners draw down first?'
    (stop width) and 'how far do losers run favorably before dying?' (breakeven trigger)."""
    def pcts(vals: list[float]) -> dict:
        if not vals:
            return {}
        s = sorted(vals)
        at = lambda q: s[min(len(s) - 1, int(q * len(s)))]
        return {"p50": at(0.50), "p75": at(0.75), "p90": at(0.90), "n": len(s)}

    scored = [o for o in items if o.get("mae") is not None and o.get("mfe") is not None]
    winners = [o for o in scored if o["pnl"] > 0]
    losers = [o for o in scored if o["pnl"] <= 0]
    return {
        "winners_mae": pcts([o["mae"] for o in winners]),
        "winners_mfe": pcts([o["mfe"] for o in winners]),
        "losers_mae": pcts([o["mae"] for o in losers]),
        "losers_mfe": pcts([o["mfe"] for o in losers]),
        "n_scored": len(scored),
    }
