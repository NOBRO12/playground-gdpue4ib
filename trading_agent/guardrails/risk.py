"""Pre-trade risk gate. None of these limits are LLM-tunable."""
from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    notional_usd: float


@dataclass(frozen=True)
class PortfolioState:
    equity_usd: float
    peak_equity_usd: float
    day_start_equity_usd: float
    trades_today: int
    current_qty: float  # qty currently held in order.symbol (>=0 for long-only)
    kill_switch_tripped: bool


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str  # "" when allowed


def allow(order: Order, state: PortfolioState) -> Decision:
    if state.kill_switch_tripped:
        return Decision(False, "kill_switch_tripped")

    # Hard drawdown kill.
    if state.peak_equity_usd > 0:
        dd = state.equity_usd / state.peak_equity_usd - 1.0
        if dd <= config.HARD_DD_KILL_PCT:
            return Decision(False, "hard_dd_kill")

    # No-shorts: a sell larger than what we hold would open a short.
    # Enforced before per-side gates so it applies regardless of buy/sell.
    if config.NO_SHORTS and order.side == "sell" and order.qty > state.current_qty:
        return Decision(False, "would_open_short")

    # Daily loss cap (only blocks new entries, not exits).
    if order.side == "buy" and state.day_start_equity_usd > 0:
        day_pnl_pct = state.equity_usd / state.day_start_equity_usd - 1.0
        if day_pnl_pct <= config.MAX_DAILY_LOSS_PCT:
            return Decision(False, "daily_loss_cap")

    if order.side == "buy" and state.trades_today >= config.MAX_TRADES_PER_DAY:
        return Decision(False, "max_trades_per_day")

    if order.side == "buy" and state.equity_usd > 0:
        pct = order.notional_usd / state.equity_usd
        if pct > config.MAX_POSITION_PCT:
            return Decision(False, "max_position_pct")

    return Decision(True, "")
