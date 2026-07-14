"""Persistent token-usage ledger: accumulate per-model Claude token counts to a JSON
file so the LIVE bot's cost survives restarts. Best-effort and side-effect-only — any
failure is swallowed so it can never affect trading. Backtests don't use this (they
report per-run cost from the in-memory Analyzer.usage)."""
from __future__ import annotations

import json
import os
import pathlib
import threading

# Serializes the read-modify-write so concurrent calls can't lose an update. (Today all
# callers run on the asyncio loop so they're already serialized, but this keeps it correct
# if record() is ever dispatched to a worker thread.)
_LOCK = threading.Lock()

_FIELDS = ("input", "output", "cache_read", "cache_creation")
_USAGE_ATTRS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_creation": "cache_creation_input_tokens",
}


def record(path: str | None, model: str, resp) -> None:
    """Add one call's token usage (from an Anthropic response) to the ledger at `path`."""
    usage = getattr(resp, "usage", None)
    if not path or usage is None:
        return
    try:
        with _LOCK:
            p = pathlib.Path(path)
            try:
                data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            except Exception:  # noqa: BLE001 - corrupt file: start fresh rather than crash
                data = {}
            entry = data.setdefault(model, {"calls": 0, **{f: 0 for f in _FIELDS}})
            entry["calls"] += 1
            for field, attr in _USAGE_ATTRS.items():
                entry[field] += getattr(usage, attr, 0) or 0
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = f"{p}.{os.getpid()}.tmp"
            pathlib.Path(tmp).write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, str(p))   # atomic swap so a crash mid-write can't corrupt it
    except Exception:  # noqa: BLE001 - ledger is best-effort, never break the caller
        pass
