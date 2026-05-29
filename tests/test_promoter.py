from __future__ import annotations

from trading_agent.evaluation.metrics import Scorecard
from trading_agent.improvement.promoter import _gate


def _sc(
    sharpe: float,
    max_dd: float,
    n: int = 50,
    total_return: float = 0.20,
    excess_return: float = 0.10,
) -> Scorecard:
    # Defaults make money and beat buy-and-hold so the benchmark gate passes;
    # tests that probe the benchmark checks override these.
    return Scorecard(
        sharpe=sharpe,
        sortino=sharpe,
        max_dd=max_dd,
        n_trades=n,
        win_rate=0.5,
        expectancy_r=0.1,
        equity_final=100_000.0,
        total_return=total_return,
        excess_return=excess_return,
    )


def test_promotes_on_clear_improvement():
    accepted, reason = _gate(_sc(0.5, -0.05), _sc(0.8, -0.05))
    assert accepted and reason == "promoted"


def test_rejects_below_absolute_sharpe_floor():
    # Negative champion -> +0.10 delta clears the relative gate but absolute
    # gate must still floor at 0.20.
    accepted, reason = _gate(_sc(-0.1, -0.05), _sc(0.15, -0.05))
    assert not accepted and reason == "absolute_sharpe_floor"


def test_rejects_on_sharpe_regression():
    accepted, reason = _gate(_sc(0.8, -0.05), _sc(0.6, -0.05))
    assert not accepted and reason == "oos_sharpe_regression"


def test_rejects_when_sharpe_delta_too_small():
    accepted, reason = _gate(_sc(0.5, -0.05), _sc(0.55, -0.05))
    assert not accepted and reason == "oos_sharpe_regression"


def test_rejects_on_dd_blowup():
    accepted, reason = _gate(_sc(0.5, -0.05), _sc(0.9, -0.10))
    assert not accepted and reason == "oos_max_dd_regression"


def test_rejects_when_too_few_trades():
    accepted, reason = _gate(_sc(0.5, -0.05), _sc(0.9, -0.05, n=10))
    assert not accepted and reason == "insufficient_trades"


def test_rejects_negative_absolute_return():
    # Clears Sharpe/DD/trades but actually lost money on OOS.
    chal = _sc(0.9, -0.05, total_return=-0.03, excess_return=0.05)
    accepted, reason = _gate(_sc(0.5, -0.05), chal)
    assert not accepted and reason == "negative_absolute_return"


def test_rejects_when_underperforms_buy_and_hold():
    # Made money in absolute terms but trailed buy-and-hold -> not worth the risk.
    chal = _sc(0.9, -0.05, total_return=0.04, excess_return=-0.02)
    accepted, reason = _gate(_sc(0.5, -0.05), chal)
    assert not accepted and reason == "underperforms_benchmark"


def test_promotes_when_beats_benchmark_and_sharpe():
    chal = _sc(0.9, -0.05, total_return=0.12, excess_return=0.06)
    accepted, reason = _gate(_sc(0.5, -0.05), chal)
    assert accepted and reason == "promoted"
