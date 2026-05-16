"""Strategy ABC: spec in, signal series out."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .spec import StrategySpec


class Strategy(ABC):
    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec

    @abstractmethod
    def signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame indexed by bars.index with columns:

        - ``entry`` (bool): True on the bar that should open a new long.
        - ``exit`` (bool): True on the bar that should flatten an open long.
        - ``stop_px`` (float): suggested initial stop price (NaN if none).
        - ``reason`` (str): short label persisted to the trade log.

        Pure function of ``bars`` and ``self.spec``; no side effects, no I/O.
        """
        raise NotImplementedError
