"""Read-only pre-live account check — a 'canary' that spends NOTHING and places NO orders.

Verifies the live path's preconditions against your real account:
  * the agent wallet connects and derives an address;
  * collateral / account value is present on each enabled dex (HIP-3 `xyz` uses INDEPENDENT
    margin, so it needs its own collateral — not just the core perp dex);
  * the exact coin-name keys the exchange returns for open positions, and whether the
    exchange-stop reconciliation's name match would recognize them;
  * live mid + top-of-book spread/depth for a sample market (the entry liquidity guard's path);
  * a PREVIEW of the reduce-only stop order the bot WOULD rest on a live entry — computed and
    printed, never sent.

    python scripts/check_account.py
    python scripts/check_account.py --symbol MRVL
    python scripts/check_account.py --symbol BTC --notional 200
"""
from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, "src")

try:  # robust output on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.analysis.universe import Universe  # noqa: E402
from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402
from hlbot.trading.risk import compute_sl_tp, exit_params  # noqa: E402

OK, WARN, BAD = "[OK ]", "[WARN]", "[FAIL]"


async def account_summary(hl: HLClient, dex: str) -> dict:
    st = await asyncio.to_thread(hl.info.user_state, hl.address, dex)
    ms = st.get("marginSummary", {}) if isinstance(st, dict) else {}
    return {
        "account_value": float(ms.get("accountValue", 0) or 0),
        "margin_used": float(ms.get("totalMarginUsed", 0) or 0),
        "withdrawable": float(st.get("withdrawable", 0) or 0) if isinstance(st, dict) else 0.0,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="sample symbol to inspect (e.g. MRVL, BTC)")
    ap.add_argument("--notional", type=float, default=None,
                    help="notional USD for the stop preview (default: base_notional_usd)")
    ap.add_argument("--wallet", default=None,
                    help="arena entrant key (gemini|gpt|sonnet|deepseek|grok) — check ITS funded "
                         "wallet (ARENA_<KEY>_ADDRESS/_SECRET) instead of the default HL_ACCOUNT. "
                         "Run once per funded model as a go-live pre-flight.")
    args = ap.parse_args()

    cfg = Config()
    missing = cfg.secrets.missing()
    if missing:
        print(f"{BAD} Missing secrets in .env: {', '.join(missing)}")
        return
    # Arena pre-flight: point the check at a specific entrant's funded wallet.
    wallet_addr = wallet_sec = None
    if args.wallet:
        wallet_addr, wallet_sec = cfg.secrets.arena_wallet(args.wallet)
        if not (wallet_addr and wallet_sec):
            print(f"{BAD} arena wallet '{args.wallet}' not configured — set "
                  f"ARENA_{args.wallet.upper()}_ADDRESS and ARENA_{args.wallet.upper()}_SECRET in .env")
            return
        print(f" checking ARENA wallet '{args.wallet}' (not the default HL_ACCOUNT_ADDRESS)")

    print("=" * 72)
    print(" hlbot — read-only pre-live account check (no orders placed)")
    print("=" * 72)
    net = cfg.secrets.hl_network.lower()
    print(f" network: {net}   dry_run(runtime): {cfg.runtime.dry_run}   "
          f"allowed_dexes: {cfg.app.filters.allowed_dexes}")
    if net == "testnet":
        print(f" {WARN} testnet has NO trade.xyz equities (xyz is mainnet-only) — only crypto "
              "perps are inspectable here.")

    # ---- connect ---------------------------------------------------------------
    hl = HLClient(cfg, address=wallet_addr, secret_key=wallet_sec) if args.wallet else HLClient(cfg)
    try:
        agent = await hl.connect()
        print(f"{OK} connected — account {hl.address}")
        print(f"      agent (API) wallet: {agent}")
        if agent.lower() != (hl.address or "").lower():
            print(f"      {WARN} agent != account (normal for an agent wallet) — it must be "
                  "APPROVED for this account on Hyperliquid, or live orders will fail.")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} connect failed: {exc}")
        return

    # ---- universe --------------------------------------------------------------
    universe = Universe(hl, cfg.app.filters.allowed_dexes)
    try:
        n = await universe.refresh()
        print(f"{OK} universe: {n} symbols  ({len(universe.equity_symbols())} equity / "
              f"{len(universe.crypto_symbols())} crypto)")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} universe refresh failed: {exc}")
        return
    if n == 0:
        print(f"{BAD} no tradable markets — check HL_NETWORK / allowed_dexes.")
        return

    # ---- collateral ------------------------------------------------------------
    # Hyperliquid uses a UNIFIED margin model: USDC held in the spot account collateralizes
    # perps (core + HIP-3 dexes), so per-dex perp accountValue can read $0 while the real
    # buying power sits in spot. Sum both so we don't false-alarm on a funded account.
    print("\n--- collateral (unified margin) ---")
    spot_usdc = 0.0
    try:
        sp = await asyncio.to_thread(hl.info.spot_user_state, hl.address)
        bals = {b.get("coin"): float(b.get("total", 0) or 0) for b in (sp or {}).get("balances", [])}
        spot_usdc = bals.get("USDC", 0.0)
        nonzero = {c: v for c, v in bals.items() if v > 0}
        print(f"{OK if spot_usdc > 0 else WARN} spot balance (unified collateral): "
              f"USDC ${spot_usdc:,.2f}" + (f"  other: {nonzero}" if len(nonzero) > 1 else ""))
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} could not read spot balance: {exc}")
    perp_total = 0.0
    for dex in cfg.app.filters.allowed_dexes:
        label = "crypto" if dex == "" else f"{dex!r}"
        try:
            summ = await account_summary(hl, dex)
            perp_total += summ["account_value"]
            print(f"      perp dex {label:8} account_value=${summ['account_value']:,.2f}  "
                  f"margin_used=${summ['margin_used']:,.2f}  withdrawable=${summ['withdrawable']:,.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{BAD} dex {label}: could not read perp state: {exc}")
    total = spot_usdc + perp_total
    if total > 0:
        print(f"{OK} total collateral (spot USDC + perp): ${total:,.2f} — unified margin "
              "covers both the core and the xyz dex.")
    else:
        print(f"{BAD} no collateral found (spot USDC + perp = $0) — fund the account before live.")

    # ---- open positions per dex (coin-name format for reconciliation) ----------
    print("\n--- open positions (exchange-stop reconciliation key format) ---")
    for dex in cfg.app.filters.allowed_dexes:
        label = "crypto" if dex == "" else f"{dex!r}"
        try:
            pos = await hl.positions(dex)
            if pos:
                print(f"      dex {label}: {pos}")
                for coin in pos:
                    print(f"        key {coin!r} -> bare {coin.split(':')[-1]!r} "
                          "(reconciliation matches this against position.symbol)")
            else:
                print(f"      dex {label}: flat (no open positions to inspect)")
        except Exception as exc:  # noqa: BLE001
            print(f"      {WARN} could not read positions for {label}: {exc}")

    # ---- sample market: mid, spread/depth, stop preview ------------------------
    sym = (args.symbol
           or ("MRVL" if "MRVL" in universe.equity_symbols() else None)
           or ("BTC" if "BTC" in universe.crypto_symbols() else None)
           or (universe.equity_symbols() or universe.crypto_symbols() or [None])[0])
    if not sym:
        print("\n(no sample symbol available to inspect)")
        await _bye()
        return
    ac = "equity" if sym.upper() in universe.equity_symbols() else "crypto"
    market = universe.resolve(sym, ac)
    print(f"\n--- sample market: {sym} ({ac}) ---")
    if not market:
        print(f"{WARN} {sym} not resolvable on enabled dexes.")
        await _bye()
        return
    print(f"      market={market.name}  dex={market.dex!r}  sz_decimals={market.sz_decimals}  "
          f"max_leverage={market.max_leverage}")

    mid = await hl.mid(market)
    if not mid or mid <= 0:
        print(f"{BAD} no live mid for {market.name} — cannot price an entry/stop.")
        await _bye()
        return
    print(f"{OK} live mid = {mid}")

    # spread / depth guard (only meaningful on the guarded dexes, e.g. xyz)
    try:
        book = await hl.l2_book(market)
        sp = HLClient.book_spread(book)
        if sp:
            bid, ask, bmid, spread_pct = sp
            depth = HLClient.top_depth_usd(book)   # cumulative near-touch (top 10 levels)
            n_lv = len((book.get("levels") or [[]])[0])
            cap = cfg.app.risk.max_spread_pct
            guarded = market.dex in cfg.app.risk.spread_guard_dexes
            min_depth = cfg.app.risk.min_top_depth_usd
            spread_bad = guarded and cap > 0 and spread_pct > cap
            depth_bad = guarded and min_depth > 0 and 0 < depth < min_depth
            tag = WARN if (spread_bad or depth_bad) else OK
            print(f"{tag} book: bid {bid} / ask {ask}  spread {spread_pct:.2%} "
                  f"(guard {'on' if guarded else 'off'}, cap {cap:.2%})")
            print(f"      near-touch depth (thinner side, top 10 of {n_lv} levels): ${depth:,.0f}  "
                  f"(min_top_depth_usd guard: ${min_depth:,.0f}{' OFF' if min_depth <= 0 else ''})")
        else:
            print(f"      {WARN} book empty/one-sided — guard fails OPEN (entry would proceed).")
    except Exception as exc:  # noqa: BLE001
        print(f"      {WARN} l2_book failed ({exc}) — guard fails OPEN (entry would proceed).")

    # ---- exchange-stop PREVIEW (computed, NOT sent) ---------------------------
    notional = args.notional or cfg.app.risk.base_notional_usd
    size = round(notional / mid, market.sz_decimals)
    _, stop_pct, _, _ = exit_params("immediate", cfg.app.risk)
    stop_loss, _ = compute_sl_tp("long", mid, stop_pct, 0.0)
    is_buy_to_close = False  # closing a LONG = sell
    try:
        trig = hl.exchange._slippage_price(market.name, is_buy_to_close, 0.0, stop_loss)
        limit = hl.exchange._slippage_price(market.name, is_buy_to_close, cfg.app.risk.slippage_pct,
                                            stop_loss)
        print("\n--- exchange-stop PREVIEW for a hypothetical LONG entry (NOT sent) ---")
        print(f"      entry≈{mid}  size={size}  stop_pct={stop_pct:.1%}  stop≈{stop_loss:.6g}")
        print(f"      would place: order(coin={market.name!r}, is_buy={is_buy_to_close} (sell to "
              f"close long), sz={size}, limit_px={limit}, reduce_only=True,")
        print(f"                         order_type={{'trigger': {{'triggerPx': {trig}, "
              "'isMarket': True, 'tpsl': 'sl'}})")
        print(f"{OK} stop price rounding resolves cleanly on this market.")
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} could not compute stop preview ({exc}) — check price rounding for {market.name}.")

    await _bye()


async def _bye() -> None:
    print("\nDone. This was READ-ONLY — no orders were placed and nothing was spent.")
    print("To validate the exchange stop for real: flip ONE symbol to live at minimal size,")
    print("watch the resting stop appear on Hyperliquid, then close from the dashboard and")
    print("confirm the stop is cancelled.")


if __name__ == "__main__":
    asyncio.run(main())
