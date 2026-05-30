"""EMA crossover: enter on fast > slow, exit on fast < slow."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .donchian import _atr
from .spec import EmaCrossParams


class EmaCrossStrategy(Strategy):
    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = EmaCrossParams.model_validate(self.spec.params)
        fast = bars["close"].ewm(span=p.fast, adjust=False).mean()
        slow = bars["close"].ewm(span=p.slow, adjust=False).mean()
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)

        entry = (fast > slow) & (prev_fast <= prev_slow)
        exit_ = (fast < slow) & (prev_fast >= prev_slow)

        atr = _atr(bars)
        stop_px = bars["close"] - self.spec.risk.atr_mult_stop * atr

        return pd.DataFrame(
            {
                "entry": entry.fillna(False),
                "exit": exit_.fillna(False),
                "stop_px": stop_px,
                "reason": np.where(entry, "ema_cross_up", ""),
            },
            index=bars.index,
        )
