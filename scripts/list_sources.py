"""List the news sources currently flowing through the Tree feed, so you can build
a source_whitelist in config.yaml. (Tree has no server-side source filter on the
API; the bot filters client-side via filters.source_whitelist / source_blacklist.)

    python scripts/list_sources.py            # top accounts + outlets
    python scripts/list_sources.py --all      # every source
"""
from __future__ import annotations

import argparse
import re
import sys

sys.path.insert(0, "src")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import httpx  # noqa: E402

URL = "https://news.treeofalpha.com/api/news"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="show every source, not just the top")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    arr = httpx.get(URL, params={"limit": args.limit}, timeout=60).json()
    cat: dict[str, int] = {}
    tw: dict[str, int] = {}
    blog: dict[str, int] = {}
    for x in arr:
        s = x.get("source") or "?"
        cat[s] = cat.get(s, 0) + 1
        title = x.get("title") or ""
        if s == "Twitter":
            m = re.search(r"\(@([A-Za-z0-9_]+)\)", title)
            if m:
                tw[m.group(1)] = tw.get(m.group(1), 0) + 1
        elif s == "Blogs":
            sn = x.get("sourceName") or (title.split(":")[0] if ":" in title else "?")
            blog[sn] = blog.get(sn, 0) + 1

    n = 10**9 if args.all else 40
    print(f"Sample of {len(arr)} recent items\n")
    print("=== categories ===")
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5}  {k}")
    print(f"\n=== Twitter accounts: {len(tw)} ===")
    for k, v in sorted(tw.items(), key=lambda kv: -kv[1])[:n]:
        print(f"  {v:4}  @{k}")
    print(f"\n=== Blog outlets: {len(blog)} ===")
    for k, v in sorted(blog.items(), key=lambda kv: -kv[1])[:n]:
        print(f"  {v:4}  {k}")
    print("\nTo act on only some, set filters.source_whitelist in config.yaml, e.g.:")
    print('  source_whitelist: ["@WuBlockchain", "@arkham", "THE BLOCK", "BLOOMBERG", "WSJ"]')


if __name__ == "__main__":
    main()
