from __future__ import annotations

import pandas as pd

from trading_agent.evaluation.metrics import Scorecard
from trading_agent.improvement import summary as summary_mod


def _score() -> Scorecard:
    return Scorecard(
        sharpe=0.5,
        sortino=0.4,
        max_dd=-0.08,
        n_trades=5,
        win_rate=0.4,
        expectancy_r=0.1,
        equity_final=101_000.0,
    )


def test_longest_losing_streak():
    trades = pd.DataFrame({"pnl_r": [1.0, -1.0, -0.5, -0.2, 2.0, -1.0]})
    assert summary_mod._longest_losing_streak(trades) == 3


def test_max_dd_duration_counts_bars_below_peak():
    equity = pd.Series([100, 110, 105, 102, 108, 115])  # below peak 110 for bars 2,3,4
    assert summary_mod._max_dd_duration_bars(equity) == 3


def test_by_regime_has_expectancy_and_win_rate():
    trades = pd.DataFrame(
        {"pnl_r": [1.0, -1.0, 2.0], "regime": ["trend", "trend", "chop"]}
    )
    out = summary_mod._by_regime(trades)
    assert out["trend"]["count"] == 2
    assert out["trend"]["win_rate"] == 0.5
    assert out["chop"]["expectancy_r"] == 2.0


def test_build_trade_summary_keys_present():
    trades = pd.DataFrame({"pnl_r": [1.0, -1.0], "regime": ["trend", "chop"]})
    equity = pd.Series([100.0, 99.0, 101.0])
    out = summary_mod.build_trade_summary(_score(), trades, equity)
    for key in (
        "n_trades", "sharpe", "sortino", "max_dd", "max_dd_duration_bars",
        "win_rate", "expectancy_r", "longest_losing_streak", "by_regime",
    ):
        assert key in out


def test_empty_trades_safe():
    out = summary_mod.build_trade_summary(_score(), pd.DataFrame(), pd.Series(dtype=float))
    assert out["longest_losing_streak"] == 0
    assert out["by_regime"] == {}
