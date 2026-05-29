"""Shared stop-loss / take-profit exit logic.

Both the vectorized backtester and the live loop must decide position exits the
*same* way, or the live agent stops trading the strategy that was backtested and
promoted. This one helper is the single source of truth for that decision, so
the two paths cannot drift.
"""
from __future__ import annotations

import math


def exit_for_levels(
    low: float, high: float, stop_px: float, take_px: float
) -> tuple[str | None, float | None]:
    """Decide whether an open long exits on this bar's stop or take-profit.

    Intrabar-conservative ordering: the stop is checked before the take-profit
    (assume the worse fill if a bar's range spans both). ``nan`` levels are
    treated as "not set". Returns ``(reason, exit_px)`` or ``(None, None)``.
    """
    if not math.isnan(stop_px) and low <= stop_px:
        return "stop", stop_px
    if not math.isnan(take_px) and high >= take_px:
        return "take_profit", take_px
    return None, None


def take_from_stop(entry_px: float, stop_px: float, take_profit_r: float) -> float:
    """Take-profit price for a long: entry + R-multiple of the per-unit risk."""
    return entry_px + take_profit_r * (entry_px - stop_px)
