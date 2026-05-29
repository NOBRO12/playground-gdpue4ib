from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent.improvement import backtester
from trading_agent.strategy.mean_reversion import MeanReversionStrategy
from trading_agent.strategy.spec import StrategySpec


def _spec(**params_override) -> StrategySpec:
    params = {"lookback": 10, "entry_z": 2.0, "exit_z": 0.0}
    params.update(params_override)
    return StrategySpec.model_validate(
        {
            "type": "mean_reversion",
            "symbol": "SPY",
            "timeframe": "1d",
            "params": params,
            "filters": {},
            "risk": {"atr_mult_stop": 2.0, "take_profit_r": 3.0, "position_pct": 0.10},
            "version": "v-mr",
        }
    )


def _osc(n: int, level: float = 100.0, amp: float = 0.5) -> list[float]:
    """A gently oscillating base so the rolling std is well-defined and nonzero
    (a perfectly flat series gives std=0 -> NaN z, which no real bar produces)."""
    return [level + (amp if i % 2 == 0 else -amp) for i in range(n)]


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1D", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 100.0},
        index=idx,
    )


def test_entry_fires_on_dip_below_mean():
    # 15 oscillating bars, then one sharp dip pushes z below -2, then recovery.
    closes = _osc(15) + [90.0] + _osc(6)
    sig = MeanReversionStrategy(_spec()).signals(_bars(closes))
    assert bool(sig["entry"].iloc[15]) is True  # the dip bar
    assert sig["entry"].sum() == 1  # the oscillation alone never triggers
    assert sig["reason"].iloc[15] == "mean_reversion_dip"


def test_exit_fires_on_reversion_to_mean():
    closes = _osc(15) + [90.0] + _osc(6)
    sig = MeanReversionStrategy(_spec()).signals(_bars(closes))
    # Price reverts back up through the mean the bar after the dip -> exit.
    assert bool(sig["exit"].iloc[16]) is True


def test_entry_is_an_event_not_a_sustained_state():
    # Price stays oversold for several bars; we open once on the crossing, not
    # on every bar below the threshold.
    closes = _osc(15) + [90.0, 90.0, 90.0, 90.0]
    sig = MeanReversionStrategy(_spec()).signals(_bars(closes))
    assert sig["entry"].sum() == 1
    assert bool(sig["entry"].iloc[15]) is True


def test_stop_is_below_close():
    closes = _osc(15) + [90.0] + _osc(6)
    sig = MeanReversionStrategy(_spec()).signals(_bars(closes))
    valid = sig["stop_px"].dropna()
    assert (valid < pd.Series(closes, index=sig.index)[valid.index]).all()


def test_runs_through_backtester(bars_chop, mean_reversion_spec):
    # Integration: the registry-dispatched strategy runs end-to-end and the
    # equity curve stays well-defined (no NaNs, positive).
    res = backtester.run(mean_reversion_spec, bars_chop)
    assert res.equity.iloc[-1] > 0
    assert not np.isnan(res.equity.iloc[-1])
