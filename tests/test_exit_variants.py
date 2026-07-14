"""walk_variant must reproduce walk_candles EXACTLY under production parameters —
that equivalence is what makes exit-lab variant deltas trustworthy — and its
generalizations (trail tiers, far TP with trail, conditional time-extension) must
behave per spec on hand-built paths."""
from hlbot.backtest.engine import walk_candles
from hlbot.backtest.exit_variants import walk_variant


def bars(specs, t0=1_000_000, step=60_000):
    """Build candles from (o, h, l, c) tuples."""
    out = []
    for i, (o, h, l, c) in enumerate(specs):
        out.append({"t": t0 + i * step, "T": t0 + (i + 1) * step,
                    "o": o, "h": h, "l": l, "c": c})
    return out


# Deterministic zoo of paths: pops, dips, gaps through stops, reversals, drifts.
PATHS = [
    [(100, 101, 99, 100), (100, 108, 100, 107), (107, 110, 104, 105), (105, 106, 96, 97)],
    [(100, 100.5, 99.5, 100), (100, 101, 95, 96)],                      # straight stop-out
    [(100, 104, 99.8, 103), (103, 105, 102, 104), (90, 91, 89, 90)],    # gap through stop
    [(100, 112, 100, 111), (111, 113, 108, 109), (109, 110, 101, 102)], # big pop, give-back
    [(100, 101, 100, 101)] * 50,                                        # drift to time exit
    [(100, 103, 97.2, 98), (98, 109, 98, 108), (108, 109, 99, 100)],    # dip then rip
]


def test_walk_variant_reproduces_walk_candles_with_production_params():
    for side in ("long", "short"):
        for tr, tp in ((0.08, 0.0), (0.0, 0.03), (0.05, 0.0)):
            for path in PATHS:
                cs = bars(path)
                entry = cs[0]["o"]
                stop = entry * (1 - 0.03) if side == "long" else entry * (1 + 0.03)
                tppx = 0.0
                if tp:
                    tppx = entry * (1 + tp) if side == "long" else entry * (1 - tp)
                h_end = cs[2]["T"] if len(cs) > 3 else cs[-1]["T"]
                a = walk_candles(side, entry, stop, tppx, tr, cs, 0, h_end)
                b = walk_variant(side, entry, stop, cs, 0, h_end,
                                 trail_tiers=((0.0, tr),) if tr > 0 else (),
                                 tp_px=tppx)
                assert (a.exit_px, a.reason, a.exit_ms) == (b.exit_px, b.reason, b.exit_ms), \
                    (side, tr, tp, path)
                assert abs(a.mae_pct - b.mae_pct) < 1e-12
                assert abs(a.mfe_pct - b.mfe_pct) < 1e-12


def test_ratchet_tightens_after_arming():
    # Pop to +10% then retrace: production 8% trail exits at 110*0.92=101.2;
    # ratchet (8% -> 4% once +6% armed) exits at 110*0.96 = 105.6.
    cs = bars([(100, 110, 100, 109), (109, 110, 100, 101)])
    w = walk_variant("long", 100, 97, cs, 0, cs[-1]["T"],
                     trail_tiers=((0.0, 0.08), (0.06, 0.04)))
    assert w.reason == "trailing stop"
    assert abs(w.exit_px - 110 * 0.96) < 1e-9
    base = walk_candles("long", 100, 97, 0.0, 0.08, cs, 0, cs[-1]["T"])
    assert abs(base.exit_px - 110 * 0.92) < 1e-9


def test_armed_trail_inactive_below_arm():
    # +2% pop then dip to -2.9%: armed-at-3% trail must NOT engage (hard stop only,
    # survives), while a production 5% trail would have exited at 102*0.95=96.9... which
    # is below the dip low 97.1, so production survives too — use a 4% trail to contrast.
    cs = bars([(100, 102, 100, 101.5), (101.5, 101.6, 97.15, 98), (98, 99, 97.5, 98.5)])
    armed = walk_variant("long", 100, 97, cs, 0, cs[-1]["T"], trail_tiers=((0.03, 0.04),))
    assert armed.reason == "time exit"            # never armed, never stopped
    always = walk_variant("long", 100, 97, cs, 0, cs[-1]["T"], trail_tiers=((0.0, 0.04),))
    assert always.reason == "trailing stop"       # 102*0.96=97.92 > 97.15 low -> binds


def test_far_tp_with_trail_takes_profit_at_target():
    cs = bars([(100, 105, 100, 104), (104, 116, 104, 112), (112, 113, 100, 101)])
    w = walk_variant("long", 100, 97, cs, 0, cs[-1]["T"],
                     trail_tiers=((0.0, 0.08),), tp_px=115.0)
    assert w.reason == "take profit" and w.exit_px == 115.0
    base = walk_candles("long", 100, 97, 0.0, 0.08, cs, 0, cs[-1]["T"])
    assert base.reason == "trailing stop"         # gives back to 116*0.92 = 106.7
    assert w.exit_px > base.exit_px


def test_extension_rides_winners_and_cuts_losers_on_time():
    flat_win = bars([(100, 106, 100, 105)] + [(105, 106, 104, 105)] * 5
                    + [(105, 112, 105, 111)] * 3)
    h_end = flat_win[3]["T"]
    ext_end = flat_win[-1]["T"]
    w = walk_variant("long", 100, 97, flat_win, 0, h_end,
                     extend_min_pnl_pct=0.02, extend_end_ms=ext_end)
    assert w.reason == "time exit (extended)" and w.exit_px == 111
    base = walk_variant("long", 100, 97, flat_win, 0, h_end)
    assert base.reason == "time exit" and base.exit_px == 105
    loser = bars([(100, 101, 99, 99.5)] + [(99.5, 100, 98.5, 99)] * 8)
    wl = walk_variant("long", 100, 97, loser, 0, loser[3]["T"],
                      extend_min_pnl_pct=0.02, extend_end_ms=loser[-1]["T"])
    assert wl.reason == "time exit"               # not extended: below threshold


def test_gap_through_armed_trail_fills_at_open():
    cs = bars([(100, 110, 100, 109), (95, 96, 94, 95)])
    w = walk_variant("long", 100, 97, cs, 0, cs[-1]["T"], trail_tiers=((0.0, 0.08),))
    assert w.reason == "trailing stop (gap)" and w.exit_px == 95
    base = walk_candles("long", 100, 97, 0.0, 0.08, cs, 0, cs[-1]["T"])
    assert base.exit_px == 95 and "gap" in base.reason
