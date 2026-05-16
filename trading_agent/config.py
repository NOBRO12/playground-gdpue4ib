"""Env-driven settings and hard-coded guardrail constants."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    db_path: Path
    strategies_dir: Path
    is_dir: Path
    oos_dir: Path
    live_state_path: Path
    alpaca_key: str | None
    alpaca_secret: str | None
    anthropic_key: str | None
    proposer_model: str
    log_level: str


def load() -> Settings:
    root = _root()
    return Settings(
        db_path=Path(os.environ.get("AGENT_DB_PATH", root / "data" / "agent.db")),
        strategies_dir=Path(os.environ.get("AGENT_STRATEGIES_DIR", root / "strategies")),
        is_dir=Path(os.environ.get("AGENT_IS_DIR", root / "data" / "is")),
        oos_dir=Path(os.environ.get("AGENT_OOS_DIR", root / "data" / "oos")),
        live_state_path=Path(
            os.environ.get("AGENT_LIVE_STATE_PATH", root / "data" / "live_state.json")
        ),
        alpaca_key=os.environ.get("ALPACA_KEY"),
        alpaca_secret=os.environ.get("ALPACA_SECRET"),
        anthropic_key=os.environ.get("ANTHROPIC_API_KEY"),
        proposer_model=os.environ.get("AGENT_PROPOSER_MODEL", "claude-sonnet-4-6"),
        log_level=os.environ.get("AGENT_LOG_LEVEL", "INFO"),
    )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


# Non-negotiable guardrail constants (never LLM-tunable).
HARD_DD_KILL_PCT = -0.10
MAX_DAILY_LOSS_PCT = -0.03
MAX_TRADES_PER_DAY = 8
MAX_POSITION_PCT = 0.25
LEVERAGE = 1.0
NO_SHORTS = True

# Promotion gate constants.
PROMOTE_MIN_SHARPE_DELTA = 0.10
PROMOTE_MAX_DD_RATIO = 1.20
PROMOTE_MIN_TRADES = 30
PROMOTE_MIN_ABS_SHARPE = 0.20
