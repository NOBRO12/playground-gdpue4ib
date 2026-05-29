"""Shared position-sizing logic (single source of truth for backtest + live).

Default sizing allocates a fixed fraction of equity (``position_pct``). When a
spec sets ``risk_per_trade_pct``, sizing instead targets a constant *risk* per
trade — risking that fraction of equity over the entry-to-stop distance — so
position size shrinks when the stop is far (volatile) and grows when it's tight.
The fixed ``position_pct`` always remains a hard notional cap.
"""
from __future__ import annotations


def size_position(
    equity: float,
    price: float,
    risk_per_unit: float,
    position_pct: float,
    risk_per_trade_pct: float | None = None,
) -> float:
    """Return the (fractional) quantity to buy. Never exceeds the position cap."""
    cap_notional = equity * position_pct
    if risk_per_trade_pct and risk_per_unit > 0:
        qty = (risk_per_trade_pct * equity) / risk_per_unit
        if qty * price > cap_notional:
            qty = cap_notional / price  # clamp to the fixed-allocation cap
    else:
        qty = cap_notional / price
    return qty
