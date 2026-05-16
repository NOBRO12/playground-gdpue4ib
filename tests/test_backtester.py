from __future__ import annotations

import pandas as pd

from trading_agent.improvement import backtester


def test_deterministic(bars_trend, donchian_spec):
    a = backtester.run(donchian_spec, bars_trend)
    b = backtester.run(donchian_spec, bars_trend)
    assert a.score.sharpe == b.score.sharpe
    assert a.score.max_dd == b.score.max_dd
    assert a.score.n_trades == b.score.n_trades
    pd.testing.assert_series_equal(a.equity, b.equity)


def test_equity_starts_at_starting_capital(bars_trend, donchian_spec):
    res = backtester.run(donchian_spec, bars_trend, starting_equity=50_000.0)
    assert res.equity.iloc[0] > 0
    # First bar can't enter (no lookback) — equity == cash.
    assert abs(res.equity.iloc[0] - 50_000.0) < 1e-6


def test_max_dd_nonpositive(bars_trend, donchian_spec):
    res = backtester.run(donchian_spec, bars_trend)
    assert res.score.max_dd <= 0


def test_trades_have_pnl(bars_trend, donchian_spec):
    res = backtester.run(donchian_spec, bars_trend)
    if len(res.trades) > 0:
        assert "pnl_usd" in res.trades.columns
        assert "pnl_r" in res.trades.columns


def test_walk_forward_runs(bars_trend, donchian_spec):
    folds = backtester.walk_forward(donchian_spec, bars_trend, n_folds=3)
    assert len(folds) == 3
    for f in folds:
        assert f.score.max_dd <= 0


def test_ema_strategy_runs(bars_trend, ema_spec):
    res = backtester.run(ema_spec, bars_trend)
    assert res.equity.iloc[-1] > 0
