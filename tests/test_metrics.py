from __future__ import annotations

import pandas as pd

from trading_agent.evaluation import metrics


def _equity(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float, name="equity")


def test_total_return_basic():
    assert abs(metrics.total_return(_equity([100.0, 110.0])) - 0.10) < 1e-9
    # Too short / degenerate -> 0.
    assert metrics.total_return(_equity([100.0])) == 0.0
    assert metrics.total_return(_equity([0.0, 50.0])) == 0.0


def test_cagr_annualizes():
    # 10% over 1 period, 365 periods/year -> enormous annualization. Just check
    # it compounds in the right direction and dwarfs the period return.
    eq = _equity([100.0, 110.0])
    assert metrics.cagr(eq, periods_per_year=365) > metrics.total_return(eq)
    # Total wipeout annualizes to -100%.
    assert metrics.cagr(_equity([100.0, 0.0]), 365) == -1.0


def test_score_populates_returns_and_excess():
    eq = _equity([100.0, 101.0, 103.0])
    sc = metrics.score(eq, pd.DataFrame(), periods_per_year=365, benchmark_return=0.01)
    assert round(sc.total_return, 6) == 0.03
    assert sc.benchmark_return == 0.01
    assert round(sc.excess_return, 6) == 0.02  # 3% strategy - 1% benchmark


def test_score_benchmark_defaults_to_zero_when_omitted():
    # Backward compatibility: legacy callers that don't pass a benchmark still work.
    eq = _equity([100.0, 105.0])
    sc = metrics.score(eq, pd.DataFrame(), periods_per_year=365)
    assert sc.benchmark_return == 0.0
    assert sc.excess_return == sc.total_return


def test_backtester_sets_benchmark_to_buy_and_hold(bars_trend, donchian_spec):
    from trading_agent.improvement import backtester

    res = backtester.run(donchian_spec, bars_trend)
    closes = bars_trend["close"]
    expected_bh = float(closes.iloc[-1] / closes.iloc[0] - 1.0)
    assert abs(res.score.benchmark_return - expected_bh) < 1e-9
    assert abs(res.score.excess_return - (res.score.total_return - expected_bh)) < 1e-9
