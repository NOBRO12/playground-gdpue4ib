"""Append-only writers for trades, equity points, and guardrail events."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_trade(conn: sqlite3.Connection, **fields: Any) -> int:
    cur = conn.execute(
        """
        INSERT INTO trades (
            version, symbol, side, qty, entry_ts, entry_px, exit_ts, exit_px,
            pnl_usd, pnl_r, reason_entry, reason_exit, regime, meta_json
        ) VALUES (
            :version, :symbol, :side, :qty, :entry_ts, :entry_px, :exit_ts, :exit_px,
            :pnl_usd, :pnl_r, :reason_entry, :reason_exit, :regime, :meta_json
        )
        """,
        {
            "version": fields["version"],
            "symbol": fields["symbol"],
            "side": fields.get("side", "long"),
            "qty": fields["qty"],
            "entry_ts": fields["entry_ts"],
            "entry_px": fields["entry_px"],
            "exit_ts": fields.get("exit_ts"),
            "exit_px": fields.get("exit_px"),
            "pnl_usd": fields.get("pnl_usd"),
            "pnl_r": fields.get("pnl_r"),
            "reason_entry": fields.get("reason_entry"),
            "reason_exit": fields.get("reason_exit"),
            "regime": fields.get("regime"),
            "meta_json": json.dumps(fields.get("meta", {})),
        },
    )
    return int(cur.lastrowid)


def close_open_trade(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    version: str,
    exit_ts: str,
    exit_px: float,
    reason_exit: str,
) -> bool:
    """Close the most recent open trade for ``symbol`` under ``version``.

    Returns True if a row was updated. Used by the live loop where entry and
    exit happen on different ticks; the backtester does not need this (it
    inserts complete rows when a position closes).
    """
    cur = conn.execute(
        """
        SELECT id, entry_px, qty FROM trades
        WHERE symbol = ? AND version = ? AND exit_ts IS NULL
        ORDER BY id DESC LIMIT 1
        """,
        (symbol, version),
    )
    row = cur.fetchone()
    if row is None:
        return False
    pnl_usd = (exit_px - row["entry_px"]) * row["qty"]
    conn.execute(
        """
        UPDATE trades
        SET exit_ts = ?, exit_px = ?, pnl_usd = ?, reason_exit = ?
        WHERE id = ?
        """,
        (exit_ts, exit_px, pnl_usd, reason_exit, row["id"]),
    )
    return True


def log_equity(
    conn: sqlite3.Connection,
    ts: str,
    equity_usd: float,
    cash_usd: float,
    exposure_usd: float,
    peak_equity_usd: float,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO equity_curve
        (ts, equity_usd, cash_usd, exposure_usd, peak_equity_usd)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ts, equity_usd, cash_usd, exposure_usd, peak_equity_usd),
    )


def log_guardrail(conn: sqlite3.Connection, kind: str, detail: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO guardrail_events (ts, kind, detail_json) VALUES (?, ?, ?)",
        (_now(), kind, json.dumps(detail)),
    )
