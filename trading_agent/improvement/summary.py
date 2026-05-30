"""Build the in-sample trade summary the proposer is allowed to see.

OOS isolation: this only ever summarizes IS-window results. Richer context
(drawdown duration, losing streaks, per-regime expectancy/win-rate) helps the
LLM make grounded proposals without exposing any held-out data.
"""
from __future__ import annotations

import pandas as pd

from ..evaluation.metrics import Scorecard


def _longest_losing_streak(trades: pd.DataFrame) -> int:
    if "pnl_r" not in trades or len(trades) == 0:
        return 0
    worst = run = 0
    for pnl in trades["pnl_r"]:
        run = run + 1 if pnl < 0 else 0
        worst = max(worst, run)
    return worst


def _max_dd_duration_bars(equity: pd.Series) -> int:
    """Longest run (in bars) the equity curve spent below a prior peak."""
    if equity is None or len(equity) == 0:
        return 0
    peak = equity.iloc[0]
    worst = run = 0
    for v in equity:
        if v >= peak:
            peak = v
            run = 0
        else:
            run += 1
            worst = max(worst, run)
    return worst


def _by_regime(trades: pd.DataFrame) -> dict:
    if "regime" not in trades or len(trades) == 0:
        return {}
    out: dict[str, dict[str, float]] = {}
    for regime_label, grp in trades.groupby("regime"):
        pnl = grp["pnl_r"]
        out[str(regime_label)] = {
            "count": int(len(grp)),
            "expectancy_r": round(float(pnl.mean()), 3),
            "win_rate": round(float((pnl > 0).mean()), 3),
        }
    return out


def build_trade_summary(score: Scorecard, trades: pd.DataFrame, equity: pd.Series) -> dict:
    """Compact IS-window summary for the proposer prompt."""
    return {
        "n_trades": score.n_trades,
        "sharpe": round(score.sharpe, 3),
        "sortino": round(score.sortino, 3),
        "max_dd": round(score.max_dd, 3),
        "max_dd_duration_bars": _max_dd_duration_bars(equity),
        "win_rate": round(score.win_rate, 3),
        "expectancy_r": round(score.expectancy_r, 3),
        "longest_losing_streak": _longest_losing_streak(trades),
        "by_regime": _by_regime(trades),
    }
