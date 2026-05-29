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
    # Return-based metrics (defaults keep positional/legacy construction valid).
    total_return: float = 0.0  # equity end/start - 1 over the scored window
    cagr: float = 0.0  # annualized total return
    benchmark_return: float = 0.0  # buy-and-hold of the same instrument/window
    excess_return: float = 0.0  # total_return - benchmark_return (the alpha)


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


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    """Annualized compound return over the equity curve's span."""
    n = len(equity) - 1  # number of return periods
    if n <= 0 or equity.iloc[0] <= 0:
        return 0.0
    end = float(equity.iloc[-1])
    if end <= 0:
        return -1.0  # total loss
    return float((end / equity.iloc[0]) ** (periods_per_year / n) - 1.0)


def score(
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int,
    benchmark_return: float | None = None,
) -> Scorecard:
    returns = equity.pct_change().dropna()
    if len(trades):
        wins = trades[trades["pnl_r"] > 0]
        win_rate = len(wins) / len(trades)
        expectancy_r = float(trades["pnl_r"].mean())
    else:
        win_rate = 0.0
        expectancy_r = 0.0
    tot = total_return(equity)
    bench = float(benchmark_return) if benchmark_return is not None else 0.0
    return Scorecard(
        sharpe=sharpe(returns, periods_per_year),
        sortino=sortino(returns, periods_per_year),
        max_dd=max_drawdown(equity),
        n_trades=int(len(trades)),
        win_rate=win_rate,
        expectancy_r=expectancy_r,
        equity_final=float(equity.iloc[-1]) if len(equity) else 0.0,
        total_return=tot,
        cagr=cagr(equity, periods_per_year),
        benchmark_return=bench,
        excess_return=tot - bench,
    )
