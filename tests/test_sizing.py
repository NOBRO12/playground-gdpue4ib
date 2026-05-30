from __future__ import annotations

from trading_agent.execution.sizing import (
    edge_from_pnls,
    kelly_risk_fraction,
    size_position,
)


def test_edge_from_pnls_basic():
    # 3 wins (+20 each), 2 losses (-10 each) -> p=0.6, payoff=20/10=2.0
    n, win_rate, payoff = edge_from_pnls([20, 20, 20, -10, -10])
    assert n == 5
    assert abs(win_rate - 0.6) < 1e-9
    assert abs(payoff - 2.0) < 1e-9


def test_edge_from_pnls_needs_both_sides():
    assert edge_from_pnls([1, 2, 3]) == (3, None, None)   # no losses
    assert edge_from_pnls([-1, -2]) == (2, None, None)    # no wins
    assert edge_from_pnls([]) == (0, None, None)


def test_fixed_sizing_when_no_risk_target():
    # 10% of 100k at $100 → 100 shares.
    assert size_position(100_000, 100.0, 5.0, 0.10, None) == 100.0


def test_vol_target_sizes_by_risk_distance():
    # risk 0.5% of 100k = $500 budget; stop distance $50/unit → 10 units.
    assert size_position(100_000, 100.0, 50.0, 0.10, 0.005) == 10.0


def test_vol_target_clamped_to_position_cap():
    # Tight $1 stop would want 500 units ($50k) — clamped to the 10% cap (100 units).
    assert size_position(100_000, 100.0, 1.0, 0.10, 0.005) == 100.0


def test_nonpositive_risk_unit_falls_back_to_fixed():
    assert size_position(100_000, 100.0, 0.0, 0.10, 0.005) == 100.0


# --- Fractional-Kelly ---------------------------------------------------------

def test_kelly_risk_fraction_classic_formula():
    # p=0.6, b=2 -> f* = 0.6 - 0.4/2 = 0.4
    assert abs(kelly_risk_fraction(0.6, 2.0) - 0.4) < 1e-12


def test_kelly_risk_fraction_zero_without_edge():
    assert kelly_risk_fraction(0.4, 1.0) == 0.0   # p - (1-p)/b = -0.2 -> floored 0
    assert kelly_risk_fraction(0.0, 2.0) == 0.0
    assert kelly_risk_fraction(0.6, 0.0) == 0.0   # degenerate payoff
    assert kelly_risk_fraction(1.0, 2.0) == 0.0   # p>=1 guarded


def test_kelly_sizes_by_edge():
    # half-Kelly of f*=0.4 -> 0.2 risk fraction; $20k risk / $50 stop = 400 units,
    # but that's $40k > 10% cap ($10k) -> clamped to 100 units.
    qty = size_position(
        100_000, 100.0, 50.0, 0.10,
        kelly_fraction=0.5, win_rate=0.6, payoff_ratio=2.0,
    )
    # 0.2 * 100k = 20k risk budget / 50 = 400 units = $40k notional -> clamp 100.
    assert qty == 100.0


def test_kelly_below_cap_sizes_down():
    # Modest edge: p=0.55,b=1.5 -> f*=0.55-0.45/1.5=0.25; quarter via kelly_fraction
    # 0.2 -> 0.05 risk. $5k / $50 = 100 units = $10k notional == 10% cap exactly.
    qty = size_position(
        100_000, 100.0, 50.0, 0.10,
        kelly_fraction=0.2, win_rate=0.55, payoff_ratio=1.5,
    )
    assert abs(qty - 100.0) < 1e-9


def test_kelly_never_exceeds_position_cap_even_with_huge_edge():
    qty = size_position(
        100_000, 100.0, 1.0, 0.10,
        kelly_fraction=0.5, win_rate=0.95, payoff_ratio=5.0,
    )
    assert qty * 100.0 <= 100_000 * 0.10 + 1e-6


def test_kelly_no_edge_falls_back_to_fixed():
    # f* floors to 0 -> kelly overlay disabled -> fixed 10% sizing (100 units).
    qty = size_position(
        100_000, 100.0, 50.0, 0.10,
        kelly_fraction=0.5, win_rate=0.4, payoff_ratio=1.0,
    )
    assert qty == 100.0


def test_kelly_off_when_edge_inputs_missing():
    # kelly_fraction set but no realized edge supplied -> fixed sizing.
    qty = size_position(
        100_000, 100.0, 50.0, 0.10,
        kelly_fraction=0.5, win_rate=None, payoff_ratio=None,
    )
    assert qty == 100.0


def test_backtest_live_parity_same_inputs_same_qty():
    # Single source of truth: identical args -> identical qty regardless of caller.
    args = dict(
        equity=100_000, price=123.45, risk_per_unit=7.0, position_pct=0.10,
        risk_per_trade_pct=0.005,
    )
    assert size_position(**args) == size_position(**args)
