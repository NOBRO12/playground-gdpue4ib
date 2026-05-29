from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_agent.strategy.spec import StrategySpec


def _base() -> dict:
    return {
        "type": "donchian",
        "symbol": "BTC/USD",
        "timeframe": "1h",
        "params": {"entry_lookback": 20, "exit_lookback": 10},
        "filters": {},
        "risk": {"atr_mult_stop": 2.0, "take_profit_r": 3.0, "position_pct": 0.10},
        "version": "v-test",
    }


def test_valid_donchian():
    spec = StrategySpec.model_validate(_base())
    assert spec.type == "donchian"
    assert spec.params["entry_lookback"] == 20


def test_stock_symbol_accepted():
    spec = StrategySpec.model_validate(_base() | {"symbol": "SPY", "timeframe": "1d"})
    assert spec.symbol == "SPY"


def test_malformed_symbol_rejected():
    # Lowercase / too long / punctuation don't match the ticker-or-pair pattern.
    for bad in ("spy", "TOOLONG", "BAD!", "AA/EUR"):
        with pytest.raises(ValidationError):
            StrategySpec.model_validate(_base() | {"symbol": bad})


def test_position_pct_capped():
    raw = _base()
    raw["risk"]["position_pct"] = 0.5
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_risk_per_trade_pct_optional_and_bounded():
    assert StrategySpec.model_validate(_base()).risk.risk_per_trade_pct is None
    raw = _base()
    raw["risk"]["risk_per_trade_pct"] = 0.005
    assert StrategySpec.model_validate(raw).risk.risk_per_trade_pct == 0.005
    raw["risk"]["risk_per_trade_pct"] = 0.5  # above the 0.05 cap
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_kelly_fraction_optional_and_capped_at_half():
    assert StrategySpec.model_validate(_base()).risk.kelly_fraction is None
    raw = _base()
    raw["risk"]["kelly_fraction"] = 0.25
    assert StrategySpec.model_validate(raw).risk.kelly_fraction == 0.25
    raw["risk"]["kelly_fraction"] = 0.75  # above the 0.5 half-Kelly cap
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)
    raw["risk"]["kelly_fraction"] = 0.0  # must be > 0 when set
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_ema_fast_lt_slow():
    raw = _base() | {"type": "ema_cross", "params": {"fast": 50, "slow": 10}}
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_donchian_params_required_for_donchian_type():
    raw = _base() | {"params": {"fast": 5, "slow": 20}}  # ema params on donchian
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_valid_mean_reversion():
    raw = _base() | {
        "type": "mean_reversion",
        "params": {"lookback": 20, "entry_z": 2.0, "exit_z": 0.0},
    }
    spec = StrategySpec.model_validate(raw)
    assert spec.type == "mean_reversion"
    assert spec.params["entry_z"] == 2.0


def test_mean_reversion_exit_must_be_above_entry():
    # exit_z <= -entry_z would exit below entry and never capture reversion.
    raw = _base() | {
        "type": "mean_reversion",
        "params": {"lookback": 20, "entry_z": 2.0, "exit_z": -2.0},
    }
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_mean_reversion_type_in_tool_schema():
    # The Claude tool input_schema must expose the new family so the proposer
    # can actually emit it.
    from trading_agent.strategy.spec import claude_tool_input_schema

    dumped = str(claude_tool_input_schema())
    assert "mean_reversion" in dumped
