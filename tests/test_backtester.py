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


def _kelly_spec(donchian_spec):
    raw = donchian_spec.model_dump()
    raw["risk"]["kelly_fraction"] = 0.5
    raw["version"] = "v-kelly"
    from trading_agent.strategy.spec import StrategySpec

    return StrategySpec.model_validate(raw)


def test_kelly_curve_is_opt_in(bars_trend, donchian_spec):
    kspec = _kelly_spec(donchian_spec)
    # Default: never computed, even when the spec sets kelly_fraction.
    assert backtester.run(kspec, bars_trend).kelly_equity is None
    # Opt-in: present and well-formed.
    res = backtester.run(kspec, bars_trend, kelly_curve=True)
    assert res.kelly_equity is not None
    assert len(res.kelly_equity) == len(res.equity)
    assert res.kelly_equity.iloc[-1] > 0


def test_kelly_fraction_does_not_change_the_gate_score(bars_trend, donchian_spec):
    # Adding Kelly sizing must not move Sharpe/DD/return — the gate is computed on
    # the unsized base curve regardless of kelly_fraction or kelly_curve.
    base = backtester.run(donchian_spec, bars_trend).score
    kspec = _kelly_spec(donchian_spec)
    with_kelly = backtester.run(kspec, bars_trend, kelly_curve=True).score
    assert with_kelly.sharpe == base.sharpe
    assert with_kelly.max_dd == base.max_dd
    assert with_kelly.equity_final == base.equity_final
    assert with_kelly.total_return == base.total_return


def test_no_kelly_curve_when_fraction_unset(bars_trend, donchian_spec):
    # kelly_curve=True but the spec has no kelly_fraction -> still None.
    assert backtester.run(donchian_spec, bars_trend, kelly_curve=True).kelly_equity is None
