"""SQLite persistence (audit log + dashboard history + crash recovery state)."""
from __future__ import annotations

import json
import logging
import pathlib
import statistics
import time
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from ..models import Analysis, Decision, NewsItem, Position

log = logging.getLogger("hlbot.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY, ts INTEGER, source TEXT, title TEXT, body TEXT, link TEXT, json TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    news_id TEXT, ts INTEGER, ticker TEXT, asset_class TEXT, direction TEXT,
    confidence REAL, is_stale INTEGER, rationale TEXT, model TEXT, json TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    news_id TEXT, ts INTEGER, action TEXT, reason TEXT, symbol TEXT, side TEXT,
    notional REAL, leverage INTEGER, entry_px REAL, stop_loss REAL, take_profit REAL, json TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY, news_id TEXT, opened_ms INTEGER, closed_ms INTEGER,
    symbol TEXT, dex TEXT, side TEXT, size REAL, entry_px REAL, exit_px REAL,
    stop_loss REAL, take_profit REAL, leverage INTEGER, notional REAL,
    status TEXT, pnl REAL, exit_reason TEXT, dry_run INTEGER, model_id TEXT, json TEXT
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_model ON positions(model_id);
CREATE INDEX IF NOT EXISTS idx_news_ts ON news(ts);
CREATE INDEX IF NOT EXISTS idx_analyses_ts ON analyses(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_positions_opened ON positions(opened_ms);
CREATE INDEX IF NOT EXISTS idx_analyses_news ON analyses(news_id);
CREATE INDEX IF NOT EXISTS idx_decisions_news ON decisions(news_id);
"""


class Store:
    def __init__(self, path: str = "data/hlbot.sqlite"):
        self.path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> "Store":
        pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        # WAL + NORMAL: a commit on the news hot path no longer pays a full
        # rollback-journal fsync, and readers (dashboard snapshot) don't block writers.
        # busy_timeout retries brief writer contention instead of raising SQLITE_BUSY.
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        # Defensive migration for DBs created before model_id existed — MUST run before the
        # schema script: SCHEMA indexes positions(model_id), which errors on a pre-arena
        # table that still lacks the column. Fails harmlessly on a fresh DB (no table yet)
        # or when the column already exists.
        try:
            await self._db.execute("ALTER TABLE positions ADD COLUMN model_id TEXT")
        except Exception:  # noqa: BLE001 - fresh DB or column already exists
            pass
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_news(self, item: NewsItem) -> None:
        await self._db.execute(
            "INSERT OR IGNORE INTO news (id, ts, source, title, body, link, json) VALUES (?,?,?,?,?,?,?)",
            (item.id, item.time_ms, item.source, item.title, item.body, item.link,
             json.dumps(item.to_dict())),
        )
        await self._db.commit()

    async def save_analysis(self, a: Analysis) -> None:
        await self._db.execute(
            "INSERT INTO analyses (news_id, ts, ticker, asset_class, direction, confidence,"
            " is_stale, rationale, model, json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (a.news_id, int(time.time() * 1000), a.ticker, a.asset_class, a.direction,
             a.confidence, int(a.is_stale), a.rationale, a.model, json.dumps(a.to_dict())),
        )
        await self._db.commit()

    async def save_decision(self, d: Decision) -> None:
        sym = d.market.symbol if d.market else None
        await self._db.execute(
            "INSERT INTO decisions (news_id, ts, action, reason, symbol, side, notional,"
            " leverage, entry_px, stop_loss, take_profit, json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (d.news_id, int(time.time() * 1000), d.action, d.reason, sym, d.side, d.notional_usd,
             d.leverage, d.entry_px, d.stop_loss, d.take_profit, json.dumps(d.to_dict())),
        )
        await self._db.commit()

    async def upsert_position(self, p: Position) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO positions (id, news_id, opened_ms, closed_ms, symbol, dex,"
            " side, size, entry_px, exit_px, stop_loss, take_profit, leverage, notional, status,"
            " pnl, exit_reason, dry_run, model_id, json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.id, p.news_id, p.opened_ms, p.closed_ms, p.symbol, p.dex, p.side, p.size,
             p.entry_px, p.exit_px, p.stop_loss, p.take_profit, p.leverage, p.notional_usd,
             p.status, p.pnl_usd, p.exit_reason, int(p.dry_run), p.model_id, json.dumps(p.to_dict())),
        )
        await self._db.commit()

    async def lane_stats(self, model_id: str, dry_run: bool) -> dict:
        """One-row aggregate for a lane's leaderboard entry: closed-trade count, wins, realized
        PnL. SQL-side so the 30s capital loop and every snapshot don't fetch (and re-sum) the
        lane's entire closed-trade history as it grows."""
        cur = await self._db.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(pnl>0),0) AS wins, COALESCE(SUM(pnl),0) AS s"
            " FROM positions WHERE status='closed' AND model_id=? AND dry_run=?",
            (model_id, int(dry_run)))
        row = await cur.fetchone()
        return {"n": int(row["n"] or 0), "wins": int(row["wins"] or 0),
                "realized": float(row["s"] or 0.0)}

    async def open_positions(self, model_id: Optional[str] = None) -> list[dict]:
        if model_id is not None:
            cur = await self._db.execute(
                "SELECT json FROM positions WHERE status='open' AND model_id=? ORDER BY opened_ms",
                (model_id,))
        else:
            cur = await self._db.execute(
                "SELECT json FROM positions WHERE status='open' ORDER BY opened_ms")
        out: list[dict] = []
        for r in await cur.fetchall():
            try:
                out.append(json.loads(r["json"]))
            except Exception:  # noqa: BLE001 - one bad row must not abort restart
                log.error("Skipping unparseable open-position row")
        return out

    async def recent_news_ids(self, limit: int = 20000) -> list[str]:
        """The most recent news ids (newest first) — used to reseed the in-memory dedup
        on startup so a restart doesn't re-analyze news it already processed."""
        cur = await self._db.execute("SELECT id FROM news ORDER BY ts DESC LIMIT ?", (limit,))
        return [r["id"] for r in await cur.fetchall()]

    async def latest_news_ms(self) -> Optional[int]:
        """Most recent news timestamp in the store (for gap-aware startup backfill)."""
        cur = await self._db.execute("SELECT MAX(ts) AS m FROM news")
        row = await cur.fetchone()
        return int(row["m"]) if row and row["m"] else None

    async def last_entries(self, since_ms: int, model_id: Optional[str] = None) -> dict[str, int]:
        """symbol -> most recent entry time (opened_ms) since `since_ms`; rebuilds the
        per-ticker cooldown after a restart. model_id restricts to one arena lane's entries
        (None = all, the production behavior)."""
        sql = "SELECT symbol, MAX(opened_ms) AS m FROM positions WHERE opened_ms>=?"
        params: list = [int(since_ms)]
        if model_id is not None:
            sql += " AND model_id=?"
            params.append(model_id)
        cur = await self._db.execute(sql + " GROUP BY symbol", params)
        return {r["symbol"]: int(r["m"]) for r in await cur.fetchall() if r["symbol"] and r["m"]}

    async def news_detail(self, news_id: str) -> dict:
        """One news item + every model's analysis and decision rows for it — the dashboard
        modal drill-in for items that have scrolled out of the snapshot window."""
        cur = await self._db.execute("SELECT json FROM news WHERE id=?", (news_id,))
        row = await cur.fetchone()
        out: dict = {"news": None, "analyses": [], "decisions": []}
        if row:
            try:
                out["news"] = json.loads(row["json"])
            except Exception:  # noqa: BLE001
                pass
        for table in ("analyses", "decisions"):
            cur = await self._db.execute(
                f"SELECT json FROM {table} WHERE news_id=? ORDER BY ts", (news_id,))  # noqa: S608
            for r in await cur.fetchall():
                try:
                    out[table].append(json.loads(r["json"]))
                except Exception:  # noqa: BLE001 - one bad row must not blank the modal
                    pass
        return out

    async def analysis_usage(self) -> list[dict]:
        """Per-model analysis counters for the usage panel: row count, mean + median
        latency, summed per-analysis cost and first-analysis ts (latency_ms / cost_usd
        live in the json blob; rows that predate those fields don't skew AVG/median)."""
        cur = await self._db.execute(
            "SELECT model, COUNT(*) AS n,"
            " AVG(json_extract(json,'$.latency_ms')) AS avg_ms,"
            " SUM(COALESCE(json_extract(json,'$.cost_usd'),0)) AS cost,"
            " MIN(ts) AS since_ms"
            " FROM analyses GROUP BY model ORDER BY n DESC")
        rows = [dict(r) for r in await cur.fetchall()]
        # SQLite has no MEDIAN aggregate: pull the latency column (modal-open only,
        # two slim columns) and take the per-model p50 in Python.
        cur = await self._db.execute(
            "SELECT model, json_extract(json,'$.latency_ms') AS ms FROM analyses"
            " WHERE json_extract(json,'$.latency_ms') IS NOT NULL")
        lat: dict[str, list[float]] = {}
        for r in await cur.fetchall():
            lat.setdefault(r["model"], []).append(float(r["ms"]))
        for row in rows:
            v = lat.get(row["model"])
            row["med_ms"] = statistics.median(v) if v else None
        return rows

    async def arena_started_ms(self) -> Optional[int]:
        """First analysis timestamp — when the arena actually started scoring models
        (news rows can predate it via backfill). Shown as the competition start date."""
        cur = await self._db.execute("SELECT MIN(ts) AS m FROM analyses")
        row = await cur.fetchone()
        return int(row["m"]) if row and row["m"] else None

    async def news_page(self, before_ms: int, limit: int = 50) -> dict:
        """Older ANALYZED news (strictly before `before_ms`, newest first) plus their
        analysis/decision rows — the dashboard feed's "load older news" pagination.
        Un-analyzed rows (filtered / deduped items) are skipped: the feed renders one
        row per analyzed item, so they would consume page slots invisibly."""
        cur = await self._db.execute(
            "SELECT id, json FROM news WHERE ts<? AND EXISTS"
            " (SELECT 1 FROM analyses a WHERE a.news_id = news.id)"
            " ORDER BY ts DESC LIMIT ?", (int(before_ms), int(limit)))
        out: dict = {"news": [], "analyses": [], "decisions": []}
        ids: list[str] = []
        for r in await cur.fetchall():
            try:
                out["news"].append(json.loads(r["json"]))
                ids.append(r["id"])
            except Exception:  # noqa: BLE001 - one bad row must not blank the page
                pass
        if ids:
            ph = ",".join("?" * len(ids))
            for table in ("analyses", "decisions"):
                cur = await self._db.execute(
                    f"SELECT json FROM {table} WHERE news_id IN ({ph}) ORDER BY ts",  # noqa: S608
                    ids)
                for r in await cur.fetchall():
                    try:
                        out[table].append(json.loads(r["json"]))
                    except Exception:  # noqa: BLE001
                        pass
        return out

    async def recent_closed_positions(self, limit: int = 200) -> list[dict]:
        """Most recent closed positions across all lanes, newest first — seeds the
        dashboard's trade-history panel deeper than the general snapshot's row limit."""
        cur = await self._db.execute(
            "SELECT json FROM positions WHERE status='closed' ORDER BY closed_ms DESC LIMIT ?",
            (int(limit),))
        out: list[dict] = []
        for r in await cur.fetchall():
            try:
                out.append(json.loads(r["json"]))
            except Exception:  # noqa: BLE001 - one bad row must not blank the panel
                log.error("Skipping unparseable closed-position row")
        return out

    async def model_closed_positions(self, model_id: str, limit: int = 1000) -> list[dict]:
        """ONE lane's closed positions, newest first — the leaderboard chip's per-model
        trade-history modal (goes deeper than the cross-model recent_closed_positions seed)."""
        cur = await self._db.execute(
            "SELECT json FROM positions WHERE status='closed' AND model_id=?"
            " ORDER BY closed_ms DESC LIMIT ?", (model_id, int(limit)))
        out: list[dict] = []
        for r in await cur.fetchall():
            try:
                out.append(json.loads(r["json"]))
            except Exception:  # noqa: BLE001 - one bad row must not blank the modal
                log.error("Skipping unparseable closed-position row")
        return out

    async def recent_directional_analyses(self, since_ms: int,
                                          model: Optional[str] = None) -> list[dict]:
        """Directional (long/short) analyses since `since_ms`, joined to their news text —
        rebuilds the duplicate-signal window after a restart. `model` scopes to one arena
        lane (None = all, the production behavior)."""
        sql = ("SELECT a.ticker AS ticker, a.direction AS direction, a.ts AS ts,"
               " n.title AS title, n.body AS body"
               " FROM analyses a JOIN news n ON n.id = a.news_id"
               " WHERE a.ts >= ? AND a.direction IN ('long','short') AND a.ticker IS NOT NULL")
        params: list = [int(since_ms)]
        if model is not None:
            sql += " AND a.model = ?"
            params.append(model)
        cur = await self._db.execute(sql + " ORDER BY a.ts", params)
        return [dict(r) for r in await cur.fetchall()]

    async def realized_pnl_today(self, dry_run: bool, model_id: Optional[str] = None) -> float:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_ms = int(start.timestamp() * 1000)
        sql = ("SELECT COALESCE(SUM(pnl),0) AS s FROM positions WHERE status='closed'"
               " AND closed_ms>=? AND dry_run=?")
        params: list = [start_ms, int(dry_run)]
        if model_id is not None:   # arena: this lane's daily loss only
            sql += " AND model_id=?"
            params.append(model_id)
        cur = await self._db.execute(sql, tuple(params))
        row = await cur.fetchone()
        return float(row["s"] or 0.0)

    async def recent_news(self, days: float, limit: int = 1500) -> list[str]:
        """Headline strings (title + body, oldest first) from the last `days` —
        feeds the periodic regime-brief refresh. When the window holds more than
        `limit` items we must keep the MOST RECENT ones (the current climate), not the
        oldest — so take the newest `limit` by ts DESC, then return chronological."""
        cutoff = int((time.time() - days * 86400) * 1000)
        cur = await self._db.execute(
            "SELECT title, body FROM (SELECT ts, title, body FROM news WHERE ts>=?"
            " ORDER BY ts DESC LIMIT ?) ORDER BY ts ASC", (cutoff, limit))
        return [f"{(r['title'] or '')} {(r['body'] or '')}".strip()[:200]
                for r in await cur.fetchall()]

    async def recent_entered_catalysts(self, days: float, limit: int = 25) -> list[dict]:
        """Catalysts we've ALREADY entered trades on in the last `days` — injected into
        the analyzer prompt so it flags a resurfaced same-event headline as stale."""
        cutoff = int((time.time() - days * 86400) * 1000)
        cur = await self._db.execute(
            "SELECT ts, symbol, side, reason FROM decisions WHERE action='enter' AND ts>=?"
            " ORDER BY ts DESC LIMIT ?", (cutoff, limit))
        return [{"ts": r["ts"], "symbol": r["symbol"], "side": r["side"], "reason": r["reason"] or ""}
                for r in await cur.fetchall()]

    async def snapshot(self, limit: int = 50) -> dict:
        """Initial state for the dashboard."""
        async def rows(sql, n):
            cur = await self._db.execute(sql, (n,))
            return [json.loads(r["json"]) for r in await cur.fetchall()]

        return {
            "news": await rows("SELECT json FROM news ORDER BY ts DESC LIMIT ?", limit),
            "analyses": await rows("SELECT json FROM analyses ORDER BY ts DESC LIMIT ?", limit),
            "decisions": await rows("SELECT json FROM decisions ORDER BY ts DESC LIMIT ?", limit),
            "positions": await rows("SELECT json FROM positions ORDER BY opened_ms DESC LIMIT ?", limit),
        }
