"""One-time Telegram login for channel ingestion (creates data/tg_session).

Run this ONCE in a terminal; it will ask for your phone number, the code Telegram
sends you, and your 2FA password if set. After that, the bot can read channels
non-interactively.

    python scripts/telegram_login.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - Windows console: dialog titles are routinely emoji-laden
    pass

from hlbot.config import Config  # noqa: E402
from hlbot.news.telegram_source import SESSION_PATH  # noqa: E402


def main() -> None:
    cfg = Config()
    api_id = cfg.secrets.telegram_api_id
    api_hash = cfg.secrets.telegram_api_hash
    if not (api_id and api_hash):
        print("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env first "
              "(get them at https://my.telegram.org).")
        return

    from telethon.sync import TelegramClient

    with TelegramClient(SESSION_PATH, int(api_id), api_hash) as client:
        me = client.get_me()
        uname = getattr(me, "username", None) or getattr(me, "first_name", "?")
        print(f"Logged in as: {uname}")
        print(f"Session saved to {SESSION_PATH}.session — you're set.")
        print("\nChannels you can read (a sample of your dialogs):")
        for d in client.iter_dialogs(limit=40):
            if d.is_channel:
                u = getattr(d.entity, "username", None)
                print(f"  - {d.name}" + (f"  (@{u})" if u else "  (private/no username)"))
        print("\nAdd the channel usernames you want to config.yaml -> telegram_channels.")


if __name__ == "__main__":
    main()
