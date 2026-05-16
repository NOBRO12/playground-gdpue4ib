"""Shared fixtures: deterministic synthetic crypto bars and seed specs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.strategy.spec import StrategySpec


def _bars(seed: int, n: int = 600, drift: float = 0.0005, vol: float = 0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    returns = rng.normal(drift, vol, size=n)
    close = 30_000 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.004, size=n))
    low = close * (1 - rng.uniform(0, 0.004, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(50, 200, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


@pytest.fixture
def bars_trend() -> pd.DataFrame:
    return _bars(seed=7, drift=0.001, vol=0.012)


@pytest.fixture
def bars_chop() -> pd.DataFrame:
    return _bars(seed=11, drift=0.0, vol=0.02)


@pytest.fixture
def donchian_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "type": "donchian",
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "params": {"entry_lookback": 20, "exit_lookback": 10},
            "filters": {},
            "risk": {"atr_mult_stop": 2.0, "take_profit_r": 3.0, "position_pct": 0.10},
            "version": "v-test",
            "parent": None,
        }
    )


@pytest.fixture
def ema_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "type": "ema_cross",
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "params": {"fast": 12, "slow": 48},
            "filters": {},
            "risk": {"atr_mult_stop": 2.0, "take_profit_r": 3.0, "position_pct": 0.10},
            "version": "v-test-ema",
            "parent": None,
        }
    )
