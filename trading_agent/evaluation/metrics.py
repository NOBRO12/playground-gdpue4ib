"""Performance metrics. Pure functions over pandas/numpy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scorecard:
    sharpe: float
    sortino: float
    max_dd: float
    n_trades: int
    win_rate: float
    expectancy_r: float
    equity_final: float


def sharpe(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def score(equity: pd.Series, trades: pd.DataFrame, periods_per_year: int) -> Scorecard:
    returns = equity.pct_change().dropna()
    if len(trades):
        wins = trades[trades["pnl_r"] > 0]
        win_rate = len(wins) / len(trades)
        expectancy_r = float(trades["pnl_r"].mean())
    else:
        win_rate = 0.0
        expectancy_r = 0.0
    return Scorecard(
        sharpe=sharpe(returns, periods_per_year),
        sortino=sortino(returns, periods_per_year),
        max_dd=max_drawdown(equity),
        n_trades=int(len(trades)),
        win_rate=win_rate,
        expectancy_r=expectancy_r,
        equity_final=float(equity.iloc[-1]) if len(equity) else 0.0,
    )
