"""Strategy version log + atomic champion swap."""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..strategy.spec import StrategySpec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_version(
    conn: sqlite3.Connection,
    spec: StrategySpec,
    status: str,
    notes: str = "",
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO strategy_versions
        (version, parent, created_ts, spec_json, status, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (spec.version, spec.parent, _now(), spec.to_json(), status, notes),
    )


def set_status(conn: sqlite3.Connection, version: str, status: str) -> None:
    conn.execute(
        "UPDATE strategy_versions SET status = ? WHERE version = ?",
        (status, version),
    )


def promote(
    conn: sqlite3.Connection,
    challenger: StrategySpec,
    strategies_dir: str | Path,
) -> None:
    """Atomic swap: write champion.json, archive previous champion."""
    strategies_dir = Path(strategies_dir)
    champion_path = strategies_dir / "champion.json"
    archive_dir = strategies_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if champion_path.exists():
        prev = json.loads(champion_path.read_text())
        prev_version = prev.get("version", "unknown")
        shutil.copy2(champion_path, archive_dir / f"{prev_version}.json")
        set_status(conn, prev_version, "archived")

    tmp = champion_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(challenger.model_dump(), indent=2, sort_keys=True))
    tmp.replace(champion_path)
    set_status(conn, challenger.version, "champion")


def reject(
    conn: sqlite3.Connection,
    challenger: StrategySpec,
    strategies_dir: str | Path,
    reason: str,
) -> None:
    strategies_dir = Path(strategies_dir)
    archive_dir = strategies_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{challenger.version}_rejected.json"
    path.write_text(json.dumps(challenger.model_dump(), indent=2, sort_keys=True))
    set_status(conn, challenger.version, "rejected")
    conn.execute(
        "UPDATE strategy_versions SET notes = ? WHERE version = ?",
        (f"rejected: {reason}", challenger.version),
    )
