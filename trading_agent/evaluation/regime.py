"""Crude regime classifier: trend vs chop vs high-vol."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy.donchian import _adx


def classify(bars: pd.DataFrame) -> pd.Series:
    """Return a Series of labels in {trend, chop, highvol} per bar."""
    adx = _adx(bars)
    rv = bars["close"].pct_change().rolling(24).std() * np.sqrt(365 * 24)

    label = pd.Series("chop", index=bars.index, dtype="object")
    label = label.mask(adx >= 25, "trend")
    label = label.mask(rv >= 1.5, "highvol")
    return label
