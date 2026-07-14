"""Exit-strategy lab: re-walk the live-replay trade set through alternative exit shapes.

Replays the production gate/sizing (cached analyses — near-free) to get the real trade
set, then for each trade re-fetches the SAME candles (disk cache) and:
  1. reproduces the production exit exactly (walk_candles + slippage + funding must match
     the recorded PnL — a per-trade sanity gate; mismatches are excluded and held constant)
  2. walks every variant exit shape on the identical bars.

Variant families (see hlbot/backtest/exit_variants.py):
  scale_out  — close `frac` at +tp1, ride the rest on production trail
  ratchet    — trail tightens as MFE grows (8% -> 5% -> 3%)
  armed      — NO trail until MFE >= arm (hard 3% stop only), then a tight trail
  far_tp     — production 8% trail PLUS a far take-profit (monetize blowoffs)
  extend     — at the 72h time exit, winners (close PnL >= thresh) ride to 120/168h

Variants apply to trail-managed sensitivities (hours/days); `immediate` trades and
contrary-news exits are held constant across all arms. Walk-forward: best combo per
family on the first train_frac of trades (by time), judged on the validation tail.

Usage:
  .venv/bin/python scripts/exit_lab.py                # 250d replay set, all variants
  .venv/bin/python scripts/exit_lab.py --no-extend    # skip extension (no new HL fetches)
"""
import argparse
import asyncio
import json
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, "src")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from hlbot.backtest.engine import (  # noqa: E402
    fetch_candles, funding_cost, pick_entry, pnl_usd, run_live_replay, slip_exit,
    walk_candles,
)
from hlbot.backtest.exit_variants import (  # noqa: E402
    ARMED_TRAIL_GRID, EXTEND_GRID, FAR_TP_GRID, RATCHET_GRID, SCALE_OUT_GRID, walk_variant,
)
from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402
from hlbot.trading.risk import exit_params  # noqa: E402
from hlbot.analysis.universe import Universe  # noqa: E402


def profit_factor(pnls: list[float]) -> float:
    gw = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    return gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)


@dataclass
class Walked:
    trade: object
    sens: str
    candles: list = field(default_factory=list)
    ext_candles: list = field(default_factory=list)
    idx: int = -1
    horizon_end: int = 0
    pnl: dict = field(default_factory=dict)        # variant key -> pnl USD
    reason: dict = field(default_factory=dict)     # variant key -> exit reason
    constant: bool = False                          # held identical in every arm


def variant_keys(no_extend: bool) -> list[tuple[str, str, object]]:
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
    if not no_extend:
        for thresh, hours in EXTEND_GRID:
            keys.append(("extend", f"extend->{hours}h if >={thresh:.0%}", (thresh, hours)))
    return keys


async def walk_all(hl, w: Walked, r, keys, no_extend: bool) -> None:
    t = w.trade
    side, entry, size = t.side, t.entry_px, t.size
    horizon_s, stop_pct, _tp, trail_pct = exit_params(w.sens, r)
    stop_px = t.stop_px

    async def price_pnl(walk, frac=1.0) -> float:
        px = slip_exit(side, walk.exit_px, walk.reason, r)
        fund = await funding_cost(hl, t.market, side, t.notional * frac,
                                  t.time_ms, walk.exit_ms)
        return pnl_usd(side, entry, px, size * frac) - fund

    prod_tiers = ((0.0, trail_pct),)
    for fam, key, p in keys:
        if fam == "scale_out":
            frac, tp1 = p
            tp_px = entry * (1 + tp1) if side == "long" else entry * (1 - tp1)
            leg1 = walk_variant(side, entry, stop_px, w.candles, w.idx, w.horizon_end,
                                tp_px=tp_px)
            leg2 = walk_variant(side, entry, stop_px, w.candles, w.idx, w.horizon_end,
                                trail_tiers=prod_tiers)
            w.pnl[key] = await price_pnl(leg1, frac) + await price_pnl(leg2, 1 - frac)
            w.reason[key] = f"{leg1.reason}/{leg2.reason}"
        elif fam in ("ratchet", "armed"):
            walk = walk_variant(side, entry, stop_px, w.candles, w.idx, w.horizon_end,
                                trail_tiers=p)
            w.pnl[key] = await price_pnl(walk)
            w.reason[key] = walk.reason
        elif fam == "far_tp":
            tp_px = entry * (1 + p) if side == "long" else entry * (1 - p)
            walk = walk_variant(side, entry, stop_px, w.candles, w.idx, w.horizon_end,
                                trail_tiers=prod_tiers, tp_px=tp_px)
            w.pnl[key] = await price_pnl(walk)
            w.reason[key] = walk.reason
        elif fam == "extend":
            thresh, hours = p
            cs = w.ext_candles or w.candles
            walk = walk_variant(side, entry, stop_px, cs, w.idx, w.horizon_end,
                                trail_tiers=prod_tiers, extend_min_pnl_pct=thresh,
                                extend_end_ms=t.time_ms + hours * 3600_000)
            w.pnl[key] = await price_pnl(walk)
            w.reason[key] = walk.reason


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-file", default="data/bt_tg_history_270d.json")
    ap.add_argument("--days", type=float, default=250.0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--no-extend", action="store_true",
                    help="skip extension variants (no new HL candle fetches)")
    args = ap.parse_args()

    cfg = Config()
    if cfg.secrets.missing():
        print("Missing secrets:", ", ".join(cfg.secrets.missing()))
        return
    r = cfg.app.risk

    from dataclasses import fields as dc_fields
    from hlbot.models import NewsItem
    _NI = {f.name for f in dc_fields(NewsItem)}
    items = sorted((NewsItem(**{k: v for k, v in d.items() if k in _NI})
                    for d in json.loads(pathlib.Path(args.history_file).read_text(
                        encoding="utf-8"))), key=lambda i: i.time_ms)
    print(f"Loaded {len(items)} items from {args.history_file}")

    hl = HLClient(cfg)
    await hl.connect()
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    await universe.refresh()

    res = await run_live_replay(
        cfg, hl, universe, raw_items=items, days=args.days, model="claude-sonnet-4-6",
        cache_path="data/bt_cache_replay_sonnet.json",
        regime_refresh_seconds=cfg.app.regime_refresh_seconds,
        regime_lookback_days=cfg.app.regime_live_lookback_days,
        catalyst_memory_days=cfg.app.catalyst_memory_days,
        regime_key=f"sonnet_{int(args.days)}d")
    trades = res["trades"]
    print(f"\nReplay: {len(trades)} trades, baseline total "
          f"${sum(t.pnl for t in trades):,.2f}")

    sens_by_id = {k: (v.get("time_sensitivity") or "none")
                  for k, v in json.loads(pathlib.Path(
                      "data/bt_cache_replay_sonnet.json").read_text(
                      encoding="utf-8")).items()}
    keys = variant_keys(args.no_extend)
    max_ext_h = 0 if args.no_extend else max(h for _, h in EXTEND_GRID)

    walked: list[Walked] = []
    n_const = n_mismatch = 0
    for t in trades:
        sens = sens_by_id.get(t.news_id, "days")
        w = Walked(trade=t, sens=sens)
        walked.append(w)
        horizon_s, stop_pct, tp_pct, trail_pct = exit_params(sens, r)
        if t.reason.startswith("contrary news") or trail_pct <= 0:
            w.constant = True
            n_const += 1
            continue
        start = t.time_ms - 60_000
        end = t.time_ms + horizon_s * 1000 + 120_000
        ivs = [t.candle_interval] if t.candle_interval else (
            ["1m", "5m", "15m"] if horizon_s <= 6 * 3600 else ["5m", "15m", "1h"])
        candles = []
        for iv in ivs:
            candles = await fetch_candles(hl, t.market, start, end, iv)
            if candles:
                break
        idx, raw_entry = pick_entry(candles, t.time_ms) if candles else (None, None)
        if idx is None:
            w.constant = True
            n_const += 1
            continue
        w.candles, w.idx, w.horizon_end = candles, idx, t.time_ms + horizon_s * 1000
        # Sanity gate: reproduce the recorded production exit on these bars.
        base = walk_candles(t.side, t.entry_px, t.stop_px, t.tp_px, trail_pct,
                            candles, idx, w.horizon_end,
                            breakeven_arm_pct=getattr(r, "breakeven_arm_pct", 0.0),
                            breakeven_offset_pct=getattr(r, "breakeven_offset_pct", 0.0))
        px = slip_exit(t.side, base.exit_px, base.reason, r)
        fund = await funding_cost(hl, t.market, t.side, t.notional, t.time_ms, base.exit_ms)
        repro = pnl_usd(t.side, t.entry_px, px, t.size) - fund
        if abs(repro - t.pnl) > 0.05:
            print(f"  !! baseline mismatch {t.symbol} {t.time_ms}: repro {repro:.2f} "
                  f"vs recorded {t.pnl:.2f} — held constant")
            w.constant = True
            n_mismatch += 1
            continue
        if max_ext_h and horizon_s < max_ext_h * 3600:
            ext = await fetch_candles(hl, t.market, start,
                                      t.time_ms + max_ext_h * 3600_000 + 120_000,
                                      t.candle_interval or "1h")
            if ext and int(ext[0]["t"]) <= int(candles[w.idx]["t"]):
                ei, _ = pick_entry(ext, t.time_ms)
                if ei is not None and abs(float(ext[ei]["o"]) - float(
                        candles[w.idx]["o"])) < 1e-9:
                    w.ext_candles = ext
        await walk_all(hl, w, r, keys, args.no_extend)

    applied = [w for w in walked if not w.constant]
    print(f"variants applied to {len(applied)} trades "
          f"({n_const} held constant: immediate/contrary/no-data, "
          f"{n_mismatch} repro mismatches)")

    def arm_pnls(ws: list[Walked], key: str) -> list[float]:
        return [w.pnl.get(key, w.trade.pnl) if not w.constant else w.trade.pnl
                for w in ws]

    walked.sort(key=lambda w: w.trade.time_ms)
    n_train = int(len(walked) * args.train_frac)
    train, val = walked[:n_train], walked[n_train:]
    base_total = sum(w.trade.pnl for w in walked)
    base_train = sum(w.trade.pnl for w in train)
    base_val = sum(w.trade.pnl for w in val)
    print(f"\nBASELINE  total ${base_total:,.0f}  "
          f"PF {profit_factor([w.trade.pnl for w in walked]):.2f}  "
          f"train ${base_train:,.0f} (n={len(train)})  val ${base_val:,.0f} (n={len(val)})")

    print(f"\n{'variant':38} {'dTotal':>9} {'PF':>5} {'dTrain':>9} {'dVal':>9}  exit mix")
    fam_best: dict[str, tuple[float, str]] = {}
    for fam, key, _ in keys:
        tot = sum(arm_pnls(walked, key))
        dtr = sum(arm_pnls(train, key)) - base_train
        dva = sum(arm_pnls(val, key)) - base_val
        pf = profit_factor(arm_pnls(walked, key))
        mix = Counter(w.reason.get(key, w.trade.reason).replace(" (gap)", "")
                      for w in applied)
        mixs = " ".join(f"{k.split('/')[0][:5]}:{v}" for k, v in mix.most_common(4))
        print(f"{key:38} {tot - base_total:>+9,.0f} {pf:>5.2f} {dtr:>+9,.0f} "
              f"{dva:>+9,.0f}  {mixs}")
        if fam not in fam_best or dtr > fam_best[fam][0]:
            fam_best[fam] = (dtr, key)

    print("\nWALK-FORWARD (per family: combo picked on TRAIN, judged on VAL)")
    for fam, (dtr, key) in fam_best.items():
        dva = sum(arm_pnls(val, key)) - base_val
        print(f"  {fam:10} pick: {key:38} dTrain {dtr:>+8,.0f}  dVal {dva:>+8,.0f}")
        movers = sorted((w for w in val if not w.constant),
                        key=lambda w: abs(w.pnl.get(key, w.trade.pnl) - w.trade.pnl),
                        reverse=True)[:3]
        for w in movers:
            d = w.pnl.get(key, w.trade.pnl) - w.trade.pnl
            if abs(d) > 1:
                print(f"      {w.trade.symbol:7} {w.trade.side:5} base "
                      f"{w.trade.pnl:>+8,.0f} ({w.trade.reason}) -> "
                      f"{w.pnl[key]:>+8,.0f} ({w.reason[key]})  d {d:+,.0f}")

    out = {"baseline": {"total": base_total, "train": base_train, "val": base_val,
                        "n": len(walked), "n_applied": len(applied)},
           "variants": {key: {"total": sum(arm_pnls(walked, key)),
                              "d_train": sum(arm_pnls(train, key)) - base_train,
                              "d_val": sum(arm_pnls(val, key)) - base_val}
                        for _, key, _2 in keys}}
    pathlib.Path("data/exit_lab_results.json").write_text(json.dumps(out, indent=1),
                                                          encoding="utf-8")
    print("\nresults -> data/exit_lab_results.json")


if __name__ == "__main__":
    asyncio.run(main())
