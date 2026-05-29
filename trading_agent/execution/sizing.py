"""Shared position-sizing logic (single source of truth for backtest + live).

Default sizing allocates a fixed fraction of equity (``position_pct``). When a
spec sets ``risk_per_trade_pct``, sizing instead targets a constant *risk* per
trade — risking that fraction of equity over the entry-to-stop distance — so
position size shrinks when the stop is far (volatile) and grows when it's tight.

When a spec sets ``kelly_fraction`` AND a *validated* edge is supplied
(``win_rate`` + ``payoff_ratio`` from realized trades), the per-trade risk
fraction is derived from the edge via fractional Kelly instead — scaling a proven
edge up toward, but never past, the cap. A weak/absent edge sizes down or falls
back to fixed sizing; it never sizes up on hope.

The fixed ``position_pct`` always remains a hard notional cap, so Kelly can only
size *down* from the ceiling enforced by ``MAX_POSITION_PCT``.
"""
from __future__ import annotations


def kelly_risk_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Full-Kelly fraction of equity to risk per trade for a win/loss bet.

    Classic Kelly: f* = p - (1-p)/b, where p = win_rate and b = payoff_ratio
    (average win / average loss). Returns 0.0 when there is no positive edge or
    inputs are degenerate, so the caller never sizes up without an edge.
    """
    p = win_rate
    b = payoff_ratio
    if not (0.0 < p < 1.0) or b <= 0.0:
        return 0.0
    f = p - (1.0 - p) / b
    return f if f > 0.0 else 0.0


def size_position(
    equity: float,
    price: float,
    risk_per_unit: float,
    position_pct: float,
    risk_per_trade_pct: float | None = None,
    *,
    kelly_fraction: float | None = None,
    win_rate: float | None = None,
    payoff_ratio: float | None = None,
) -> float:
    """Return the (fractional) quantity to buy. Never exceeds the position cap."""
    cap_notional = equity * position_pct

    # Fractional-Kelly overlay: when enabled with a real edge, it sets the
    # per-trade risk fraction. Falls through to the existing logic otherwise.
    if (
        kelly_fraction
        and win_rate is not None
        and payoff_ratio is not None
        and risk_per_unit > 0
    ):
        eff_risk = kelly_fraction * kelly_risk_fraction(win_rate, payoff_ratio)
        if eff_risk > 0.0:
            risk_per_trade_pct = eff_risk

    if risk_per_trade_pct and risk_per_unit > 0:
        qty = (risk_per_trade_pct * equity) / risk_per_unit
        if qty * price > cap_notional:
            qty = cap_notional / price  # clamp to the fixed-allocation cap
    else:
        qty = cap_notional / price
    return qty
