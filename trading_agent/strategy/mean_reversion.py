"""Bollinger mean reversion: buy when price is stretched below its rolling
mean, exit on reversion back toward the mean.

This is the counter-trend complement to the trend-following templates
(``donchian``/``ema_cross``): it aims to profit in range-bound regimes where
breakouts whipsaw. Like the others it emits a long-only ``entry``/``exit``
signal plus an ATR ``stop_px`` below the close, so position sizing, the shared
exit logic, and every guardrail apply identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .donchian import _atr
from .spec import MeanReversionParams


class MeanReversionStrategy(Strategy):
    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = MeanReversionParams.model_validate(self.spec.params)
        close = bars["close"]
        mean = close.rolling(p.lookback).mean()
        sd = close.rolling(p.lookback).std()
        # z-score of price relative to its rolling mean. Negative = below mean.
        z = (close - mean) / sd.replace(0.0, np.nan)
        prev_z = z.shift(1)

        # Enter as price crosses DOWN through -entry_z (an event, not a state,
        # so we open once per dip rather than every bar we stay oversold).
        entry = (z <= -p.entry_z) & (prev_z > -p.entry_z)
        # Exit once price has reverted up to the exit level.
        exit_ = (z >= p.exit_z) & (prev_z < p.exit_z)

        f = self.spec.filters
        if f.vol_max is not None:
            rv = close.pct_change().rolling(24).std() * np.sqrt(365 * 24)
            entry = entry & (rv <= f.vol_max)

        atr = _atr(bars)
        stop_px = close - self.spec.risk.atr_mult_stop * atr

        return pd.DataFrame(
            {
                "entry": entry.fillna(False),
                "exit": exit_.fillna(False),
                "stop_px": stop_px,
                "reason": np.where(entry, "mean_reversion_dip", ""),
            },
            index=bars.index,
        )
