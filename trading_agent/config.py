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
    broker: str  # "alpaca" | "mock"
    mock_broker_path: Path
    asset_class: str  # "stock" | "crypto"
    stock_feed: str  # "iex" (free) | "sip" (paid) — stock market data feed
    notify: str  # "none" | "webhook"
    notify_webhook_url: str | None
    live_mode: bool  # False = paper endpoint; True = real-money live endpoint
    alpaca_key: str | None
    alpaca_secret: str | None
    anthropic_key: str | None
    proposer_model: str
    log_level: str


def load() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(_root() / ".env", override=False)
    except ImportError:
        pass
    root = _root()
    return Settings(
        db_path=Path(os.environ.get("AGENT_DB_PATH", root / "data" / "agent.db")),
        strategies_dir=Path(os.environ.get("AGENT_STRATEGIES_DIR", root / "strategies")),
        is_dir=Path(os.environ.get("AGENT_IS_DIR", root / "data" / "is")),
        oos_dir=Path(os.environ.get("AGENT_OOS_DIR", root / "data" / "oos")),
        live_state_path=Path(
            os.environ.get("AGENT_LIVE_STATE_PATH", root / "data" / "live_state.json")
        ),
        broker=os.environ.get("AGENT_BROKER", "alpaca").lower(),
        mock_broker_path=Path(
            os.environ.get("AGENT_MOCK_BROKER_PATH", root / "data" / "mock_broker.json")
        ),
        asset_class=os.environ.get("AGENT_ASSET_CLASS", "stock").lower(),
        stock_feed=os.environ.get("AGENT_STOCK_FEED", "iex").lower(),
        notify=os.environ.get("AGENT_NOTIFY", "none").lower(),
        notify_webhook_url=os.environ.get("AGENT_NOTIFY_WEBHOOK_URL"),
        live_mode=os.environ.get("AGENT_LIVE_MODE", "false").lower() in ("1", "true", "yes"),
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

# Pattern Day Trader rule (US equities): under this equity threshold a margin
# account is capped at MAX_DAY_TRADES_PER_5D day-trades per rolling 5 business
# days. Inert for crypto (the broker reports a day-trade count of 0).
PDT_EQUITY_THRESHOLD_USD = 25_000.0
MAX_DAY_TRADES_PER_5D = 3

# Data-staleness circuit breaker: max age of the latest bar before the live loop
# refuses to trade (market-data outage / gap). The 1d window absorbs weekends and
# holidays; 1h/4h allow a couple of missed bars.
MAX_BAR_AGE_SECONDS = {
    "1h": 2 * 3600,
    "4h": 8 * 3600,
    "1d": 3 * 86_400,
}

# Promotion gate constants.
PROMOTE_MIN_SHARPE_DELTA = 0.10
PROMOTE_MAX_DD_RATIO = 1.20
PROMOTE_MIN_TRADES = 30
PROMOTE_MIN_ABS_SHARPE = 0.20
# Benchmark-relative gate: a challenger must actually MAKE money on the OOS
# window and BEAT buy-and-hold of the same instrument by at least this margin.
# A strategy that underperforms simply holding the index isn't worth the risk it
# takes. Operators can raise PROMOTE_MIN_EXCESS_RETURN to demand a wider edge.
PROMOTE_REQUIRE_POSITIVE_RETURN = True
PROMOTE_MIN_EXCESS_RETURN = 0.0  # excess_return (alpha over buy-and-hold) floor

# Overfitting haircut. A single OOS pass is easy to fit by luck, especially when
# we pick the best of many challengers. Two defenses:
#  1. Walk-forward consistency: the edge must show positive excess return in at
#     least this fraction of rolling folds, not just the one held-out window.
#  2. Best-of-N trials penalty: the max of N noisy Sharpe estimates is biased
#     upward ~sqrt(2*ln N), so raise the required Sharpe delta as N grows. The
#     penalty is 0 at N=1, preserving single-challenger behavior exactly.
PROMOTE_CONSISTENCY_FOLDS = 4
PROMOTE_MIN_POSITIVE_FOLD_FRAC = 0.6
PROMOTE_TRIALS_PENALTY_COEF = 0.05
