"""Donchian breakout: long when close > N-bar high, exit on M-bar low."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .spec import DonchianParams


def _atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _adx(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(bars, n)
    plus_di = 100 * pd.Series(plus_dm, index=bars.index).rolling(n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=bars.index).rolling(n).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.rolling(n).mean()


class DonchianStrategy(Strategy):
    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = DonchianParams.model_validate(self.spec.params)
        upper = bars["high"].rolling(p.entry_lookback).max().shift(1)
        lower = bars["low"].rolling(p.exit_lookback).min().shift(1)

        entry = bars["close"] > upper
        exit_ = bars["close"] < lower

        # Filters: ADX gate dampens chop, vol_max caps extreme realized vol.
        f = self.spec.filters
        if f.adx_min is not None:
            adx = _adx(bars)
            entry = entry & (adx >= f.adx_min)
        if f.vol_max is not None:
            rv = bars["close"].pct_change().rolling(24).std() * np.sqrt(365 * 24)
            entry = entry & (rv <= f.vol_max)

        atr = _atr(bars)
        stop_px = bars["close"] - self.spec.risk.atr_mult_stop * atr

        out = pd.DataFrame(
            {
                "entry": entry.fillna(False),
                "exit": exit_.fillna(False),
                "stop_px": stop_px,
                "reason": np.where(entry, "donchian_breakout", ""),
            },
            index=bars.index,
        )
        return out
