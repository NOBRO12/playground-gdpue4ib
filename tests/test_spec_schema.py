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


def test_invalid_symbol_rejected():
    raw = _base() | {"symbol": "DOGE/USD"}
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(raw)


def test_position_pct_capped():
    raw = _base()
    raw["risk"]["position_pct"] = 0.5
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
