from __future__ import annotations

from trading_agent.evaluation.metrics import Scorecard
from trading_agent.improvement.promoter import _gate


def _sc(sharpe: float, max_dd: float, n: int = 50) -> Scorecard:
    return Scorecard(
        sharpe=sharpe,
        sortino=sharpe,
        max_dd=max_dd,
        n_trades=n,
        win_rate=0.5,
        expectancy_r=0.1,
        equity_final=100_000.0,
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
