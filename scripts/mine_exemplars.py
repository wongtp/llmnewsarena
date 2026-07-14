"""Propose few-shot exemplar CANDIDATES for data/exemplars.md from archived backtests.

Prints the top winners (candidate "correct call" examples), the high-confidence losers
(candidate "should have scored lower" examples), and chased entries (big pre-move), each
pre-formatted as an exemplar block. This is a CURATION AID — a human edits the WHY line
to state the transferable principle and keeps only 4-6 diverse examples; do not paste
the output wholesale (script-generated exemplars encode window-specific noise).

    python scripts/mine_exemplars.py                 # newest archived run
    python scripts/mine_exemplars.py --all           # pool every archived run (deduped)
    python scripts/mine_exemplars.py --top 8

LEAKAGE RULE: curate only from history STRICTLY BEFORE the window you'll validate on
(e.g. mine from an old archive, validate the newest 30d with a fresh analysis cache).
The exemplar block is part of the cached prompt prefix — any change requires a fresh
analysis-cache path in scripts/backtest.py or the replay silently reuses stale verdicts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.backtest.edge import load_report  # noqa: E402

HIST = pathlib.Path("data/bt_history")


def collect_rows(pool_all: bool) -> tuple[list[dict], str]:
    runs = sorted(d for d in HIST.iterdir() if d.is_dir()) if HIST.exists() else []
    if not runs:
        return [], "(no archived runs — run scripts/backtest.py first)"
    picked = runs if pool_all else runs[-1:]
    rows: dict[str, dict] = {}   # dedup by row id (archived windows overlap)
    for d in picked:
        html = next(iter(d.glob("*.html")), None)
        if not html:
            continue
        for r in load_report(str(html)).get("rows", []):
            rows.setdefault(r["id"], r)
    label = f"{len(picked)} run(s): {picked[0].name} .. {picked[-1].name}"
    return list(rows.values()), label


def fmt_block(r: dict, verdict_note: str) -> str:
    text = (r.get("body") or r.get("title") or "").replace("\n", " ").strip()[:220]
    when = dt.datetime.fromtimestamp(r["time_ms"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
    pnl = r.get("pnl")
    pm = r.get("pre_move_pct")
    meta = [f"{when}", f"pnl ${pnl:+.0f}" if pnl is not None else "skipped"]
    if pm is not None:
        meta.append(f"pre-move {pm:+.1%}")
    return (
        f"# candidate · {' · '.join(meta)}  {verdict_note}\n"
        f"EXAMPLE:\n"
        f"NEWS: \"{text}\"\n"
        f"-> ticker={r.get('ticker')} direction={r.get('direction')} "
        f"confidence={r.get('confidence'):.2f} time_sensitivity={r.get('time_sensitivity')} "
        f"is_stale={str(bool(r.get('is_stale'))).lower()}\n"
        f"WHY: {(r.get('rationale') or '').strip()[:200]}\n"
        f"#   ^^ EDIT the WHY to state the transferable principle; fix the labels if the\n"
        f"#      outcome shows they were wrong (e.g. a confident loser -> lower confidence).\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="pool all archived runs (deduped)")
    ap.add_argument("--top", type=int, default=5, help="candidates per section")
    args = ap.parse_args()

    rows, label = collect_rows(args.all)
    print(f"Mining exemplar candidates from {label}\n")
    if not rows:
        return
    traded = [r for r in rows if r.get("status") == "traded" and r.get("pnl") is not None]

    winners = sorted(traded, key=lambda r: -r["pnl"])[:args.top]
    print("=" * 74)
    print(" TOP WINNERS — candidate 'correct call' exemplars (keep labels, write WHY)")
    print("=" * 74)
    for r in winners:
        print(fmt_block(r, "(verdict was RIGHT)"))

    losers = sorted((r for r in traded if (r.get("confidence") or 0) >= 0.80
                     and r["pnl"] < 0), key=lambda r: r["pnl"])[:args.top]
    print("=" * 74)
    print(" HIGH-CONFIDENCE LOSERS — candidate 'should have scored lower' exemplars")
    print(" (LOWER the confidence / fix time_sensitivity in the block before using)")
    print("=" * 74)
    for r in losers:
        print(fmt_block(r, "(verdict was WRONG — relabel before using)"))

    chased = sorted((r for r in traded if (r.get("pre_move_pct") or 0) >= 0.0125
                     and r["pnl"] < 0), key=lambda r: r["pnl"])[:args.top]
    if chased:
        print("=" * 74)
        print(" CHASED ENTRIES THAT LOST (pre-move >= 1.25%) — evidence for the move guard,")
        print(" and candidate 'already repriced -> is_stale/low conf' exemplars")
        print("=" * 74)
        for r in chased:
            print(fmt_block(r, "(market had already repriced)"))

    print("Curate 4-6 DIVERSE blocks (one true regime-change winner, one plausible chop")
    print("loser, one priced-in/result headline, one trap category) into data/exemplars.md,")
    print("then A/B: scripts/backtest.py --source telegram with a FRESH cache path.")


if __name__ == "__main__":
    main()
