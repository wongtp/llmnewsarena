"""Print the persistent all-time token-usage ledger with $ cost. No API calls — it
just reads data/token_usage.json and applies the price sheet, so it's free to run.

    python scripts/token_report.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from hlbot.backtest.report import PRICING, _DEFAULT_PRICE  # noqa: E402
from hlbot.config import Config  # noqa: E402


def main() -> None:
    path = Config().app.token_ledger_file
    p = pathlib.Path(path)
    if not p.exists():
        print(f"No ledger yet at {path} — the live bot writes it as it makes Claude calls.")
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    total = 0.0
    print(f"All-time Claude token usage  ({path})\n")
    print(f"  {'model':28} {'calls':>6} {'input':>10} {'output':>8} {'cache_rd':>10} "
          f"{'cache_wr':>9} {'$cost':>9}")
    print("  " + "-" * 88)
    for model, u in sorted(data.items()):
        pin, pout, pcw, pcr = PRICING.get(model, _DEFAULT_PRICE)
        cost = (u["input"] * pin + u["output"] * pout
                + u["cache_creation"] * pcw + u["cache_read"] * pcr) / 1e6
        total += cost
        print(f"  {model:28} {u['calls']:>6} {u['input']:>10} {u['output']:>8} "
              f"{u['cache_read']:>10} {u['cache_creation']:>9} ${cost:>8.4f}")
    print("  " + "-" * 88)
    print(f"  {'TOTAL':28} {'':>6} {'':>10} {'':>8} {'':>10} {'':>9} ${total:>8.4f}")


if __name__ == "__main__":
    main()
