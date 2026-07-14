"""Slippage/latency attribution: signed-bps convention and the DB join."""
import importlib.util
import json
import pathlib
import sqlite3

_SPEC = importlib.util.spec_from_file_location(
    "slippage_report",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "slippage_report.py")
sr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sr)


def test_signed_bps_entry_adverse_positive():
    # Long entry: paid MORE than the decision mid -> adverse (+).
    assert sr.signed_bps("long", 100.0, 100.10, is_exit=False) > 0
    # Short entry: sold LOWER than the decision mid -> adverse (+).
    assert sr.signed_bps("short", 100.0, 99.90, is_exit=False) > 0
    # Favorable fills are negative.
    assert sr.signed_bps("long", 100.0, 99.95, is_exit=False) < 0


def test_signed_bps_exit_adverse_positive():
    # Long exit: received LESS than the trigger mid -> adverse (+).
    assert sr.signed_bps("long", 100.0, 99.90, is_exit=True) > 0
    # Short exit: bought back HIGHER than the trigger mid -> adverse (+).
    assert sr.signed_bps("short", 100.0, 100.10, is_exit=True) > 0


def _mk_db(path):
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE news (id TEXT PRIMARY KEY, ts INTEGER, source TEXT, title TEXT,
                       body TEXT, link TEXT, json TEXT);
    CREATE TABLE decisions (news_id TEXT, ts INTEGER, action TEXT, reason TEXT, symbol TEXT,
                            side TEXT, notional REAL, leverage INTEGER, entry_px REAL,
                            stop_loss REAL, take_profit REAL, json TEXT);
    CREATE TABLE positions (id TEXT PRIMARY KEY, news_id TEXT, opened_ms INTEGER,
                            closed_ms INTEGER, symbol TEXT, dex TEXT, side TEXT, size REAL,
                            entry_px REAL, exit_px REAL, stop_loss REAL, take_profit REAL,
                            leverage INTEGER, notional REAL, status TEXT, pnl REAL,
                            exit_reason TEXT, dry_run INTEGER, json TEXT);
    """)
    con.execute("INSERT INTO news VALUES ('n1', 1000, 'tg', 't', 'b', NULL, '{}')")
    con.execute("INSERT INTO decisions VALUES ('n1', 4000, 'enter', 'r', 'BTC', 'long',"
                " 5000, 5, 100.0, 97.0, 0.0, '{}')")
    con.execute("INSERT INTO decisions VALUES ('n1', 900, 'reject', 'dup', 'BTC', 'long',"
                " 0, 0, 0, 0, 0, '{}')")   # non-enter decision must not join
    con.execute("INSERT INTO positions VALUES ('p1', 'n1', 5000, 9000, 'BTC', '', 'long', 50,"
                " 100.05, 103.0, 97.0, 0.0, 5, 5000, 'closed', 140.0, 'take profit', 0, ?)",
                (json.dumps({"exit_decision_px": 103.1}),))
    con.commit()
    con.close()


def test_fetch_trades_join_and_blob(tmp_path):
    db = tmp_path / "t.sqlite"
    _mk_db(db)
    trades = sr.fetch_trades(str(db), dry_run=False)
    assert len(trades) == 1   # only the enter decision joins
    t = trades[0]
    assert t["news_ms"] == 1000 and t["decision_ms"] == 4000 and t["opened_ms"] == 5000
    assert t["decision_px"] == 100.0 and t["fill_px"] == 100.05
    assert abs(t["exit_decision_px"] - 103.1) < 1e-9
    # 5 bps adverse on entry; ~9.7 bps adverse on exit (103.1 -> 103.0)
    assert abs(sr.signed_bps("long", t["decision_px"], t["fill_px"], is_exit=False) - 5.0) < 0.01
    assert sr.signed_bps("long", t["exit_decision_px"], t["exit_px"], is_exit=True) > 9
    assert sr.fetch_trades(str(db), dry_run=True) == []
