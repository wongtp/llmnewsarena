"""Connectivity smoke tests (Milestone 1). Verifies each external dependency
independently so you can diagnose setup before running the bot.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import asyncio
import sys

import websockets

sys.path.insert(0, "src")

try:  # make stdout robust to non-ASCII on Windows consoles (cp1252)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.config import Config  # noqa: E402
from hlbot.trading.hl_client import HLClient  # noqa: E402

OK, BAD, WARN = "OK ", "FAIL", "WARN"


def line(mark: str, msg: str) -> None:
    print(f"  [{mark}] {msg}")


async def check_hyperliquid(cfg: Config) -> None:
    print("Hyperliquid / trade.xyz:")
    try:
        hl = HLClient(cfg)
        await hl.connect()
        for dex, sample in (("", "BTC"), ("xyz", "MRVL")):
            if dex not in cfg.app.filters.allowed_dexes:
                continue
            meta = await hl.meta(dex)
            names = [a["name"] for a in meta.get("universe", [])]
            label = "crypto perps" if dex == "" else f"{dex} perps"
            line(OK, f"{label}: {len(names)} markets")
            hit = [n for n in names if n.split(':')[-1] == sample]
            if hit:
                mids = await hl.all_mids(dex)
                px = mids.get(hit[0]) or mids.get(sample)
                line(OK, f"  e.g. {hit[0]} mid={px}")
            else:
                line(WARN, f"  sample {sample} not found in {label}")
    except Exception as exc:  # noqa: BLE001
        line(BAD, f"Hyperliquid failed: {exc}")


async def check_tree(cfg: Config) -> None:
    print("Tree of Alpha news websocket:")
    try:
        async with websockets.connect("wss://news.treeofalpha.com/ws", ping_interval=20) as ws:
            await ws.send(f"login {cfg.secrets.tree_api_key}")
            line(OK, "connected + login sent; waiting up to 20s for a news message…")
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=20)
                preview = str(raw)[:160].replace("\n", " ")
                line(OK, f"received: {preview}")
            except asyncio.TimeoutError:
                line(WARN, "connected but no message within 20s (quiet feed?)")
    except Exception as exc:  # noqa: BLE001
        line(BAD, f"Tree WS failed: {exc}")


async def check_claude(cfg: Config) -> None:
    print("Anthropic / Claude:")
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=cfg.secrets.anthropic_api_key)
        resp = await client.messages.create(
            model=cfg.app.analyzer.model_fast, max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        line(OK, f"{cfg.app.analyzer.model_fast} replied: {text.strip()[:40]}")
    except Exception as exc:  # noqa: BLE001
        line(BAD, f"Claude failed: {exc}")


async def check_telegram(cfg: Config) -> None:
    print("Telegram:")
    if not (cfg.secrets.telegram_bot_token and cfg.secrets.telegram_chat_id):
        line(WARN, "not configured (optional) - skipping")
        return
    try:
        import httpx

        url = f"https://api.telegram.org/bot{cfg.secrets.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, json={"chat_id": cfg.secrets.telegram_chat_id,
                                        "text": "✅ hlbot smoke test: Telegram is wired up."})
        line(OK if r.status_code == 200 else BAD, f"sendMessage -> HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        line(BAD, f"Telegram failed: {exc}")


async def main() -> None:
    cfg = Config()
    missing = cfg.secrets.missing()
    if missing:
        print(f"[!] Missing secrets: {', '.join(missing)} (some checks will fail)\n")
    await check_hyperliquid(cfg)
    await check_tree(cfg)
    await check_claude(cfg)
    await check_telegram(cfg)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
