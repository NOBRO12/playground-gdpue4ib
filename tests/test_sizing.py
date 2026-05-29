from __future__ import annotations

from trading_agent.execution.sizing import size_position


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
