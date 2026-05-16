"""SQLite schema + connection helpers."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty REAL NOT NULL,
        entry_ts TEXT NOT NULL,
        entry_px REAL NOT NULL,
        exit_ts TEXT,
        exit_px REAL,
        pnl_usd REAL,
        pnl_r REAL,
        reason_entry TEXT,
        reason_exit TEXT,
        regime TEXT,
        meta_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS equity_curve (
        ts TEXT PRIMARY KEY,
        equity_usd REAL NOT NULL,
        cash_usd REAL NOT NULL,
        exposure_usd REAL NOT NULL,
        peak_equity_usd REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_versions (
        version TEXT PRIMARY KEY,
        parent TEXT,
        created_ts TEXT NOT NULL,
        spec_json TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT NOT NULL,
        window_start TEXT NOT NULL,
        window_end TEXT NOT NULL,
        oos_flag INTEGER NOT NULL,
        sharpe REAL,
        sortino REAL,
        max_dd REAL,
        n_trades INTEGER,
        equity_final REAL,
        params_hash TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS proposals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        parent_version TEXT NOT NULL,
        prompt_hash TEXT NOT NULL,
        response_json TEXT NOT NULL,
        decision TEXT,
        reject_reason TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS guardrail_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        kind TEXT NOT NULL,
        detail_json TEXT
    )
    """,
]


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        for stmt in SCHEMA:
            conn.execute(stmt)
        conn.commit()


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
