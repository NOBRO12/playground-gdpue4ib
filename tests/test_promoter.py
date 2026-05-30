from __future__ import annotations

import math

from trading_agent import config
from trading_agent.evaluation.metrics import Scorecard
from trading_agent.improvement import promoter
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


def test_trials_penalty_zero_at_one_and_grows_with_n():
    from trading_agent import config

    assert promoter.trials_penalty(1) == 0.0
    assert promoter.trials_penalty(0) == 0.0
    p5 = promoter.trials_penalty(5)
    p20 = promoter.trials_penalty(20)
    assert p5 > 0 and p20 > p5  # wider search -> higher bar
    assert math.isclose(p5, config.PROMOTE_TRIALS_PENALTY_COEF * math.sqrt(2 * math.log(5)))


def test_trials_penalty_raises_effective_delta_floor():
    # A challenger that clears the base delta but not the penalized delta is
    # rejected once we account for having searched many candidates.
    champ = _sc(0.50, -0.05)
    chal = _sc(0.65, -0.05)  # +0.15 delta: clears base 0.10
    assert _gate(champ, chal)[0] is True
    penalized = config.PROMOTE_MIN_SHARPE_DELTA + promoter.trials_penalty(50)
    accepted, reason = _gate(champ, chal, sharpe_delta_floor=penalized)
    assert not accepted and reason == "oos_sharpe_regression"


def test_consistency_counts_positive_excess_folds(bars_trend, donchian_spec):
    frac = promoter.consistency(donchian_spec, bars_trend)
    assert frac is None or (0.0 <= frac <= 1.0)


def test_consistency_none_when_too_few_bars(donchian_spec, bars_trend):
    assert promoter.consistency(donchian_spec, bars_trend.iloc[:20]) is None


def test_decide_rejects_inconsistent_oos(monkeypatch, bars_trend, donchian_spec):
    import pandas as pd

    good = _sc(0.9, -0.05)
    monkeypatch.setattr(
        promoter.backtester, "run",
        lambda spec, bars, **k: promoter.backtester.BacktestResult(
            equity=pd.Series([1.0, 1.1]), trades=pd.DataFrame(), score=good
        ),
    )
    # Force the single-window gate to pass, then make walk-forward consistency fail.
    monkeypatch.setattr(promoter, "_gate", lambda c, ch, sharpe_delta_floor=None: (True, "promoted"))
    monkeypatch.setattr(promoter, "consistency", lambda spec, bars: 0.25)  # below 0.6
    decision = promoter.decide(
        donchian_spec, donchian_spec, bars_trend, consistency_bars=bars_trend
    )
    assert not decision.accepted and decision.reason == "inconsistent_oos"


def test_decide_promotes_when_consistent(monkeypatch, bars_trend, donchian_spec):
    import pandas as pd

    good = _sc(0.9, -0.05)
    monkeypatch.setattr(
        promoter.backtester, "run",
        lambda spec, bars, **k: promoter.backtester.BacktestResult(
            equity=pd.Series([1.0, 1.1]), trades=pd.DataFrame(), score=good
        ),
    )
    monkeypatch.setattr(promoter, "_gate", lambda c, ch, sharpe_delta_floor=None: (True, "promoted"))
    monkeypatch.setattr(promoter, "consistency", lambda spec, bars: 0.75)  # above 0.6
    decision = promoter.decide(
        donchian_spec, donchian_spec, bars_trend, consistency_bars=bars_trend
    )
    assert decision.accepted and decision.reason == "promoted"
    assert decision.challenger_stress_score is not None  # stress test was reached


def _stub_runs(monkeypatch, base_score, stress_score):
    """Make backtester.run return base_score at 1x cost and stress_score at >1x."""
    import pandas as pd

    def fake_run(spec, bars, starting_equity=100_000.0, fee_bps=10.0, slippage_bps=5.0):
        sc = stress_score if fee_bps > config.BACKTEST_FEE_BPS else base_score
        return promoter.backtester.BacktestResult(
            equity=pd.Series([1.0, 1.1]), trades=pd.DataFrame(), score=sc
        )

    monkeypatch.setattr(promoter.backtester, "run", fake_run)


def test_decide_rejects_when_edge_dies_at_2x_cost(monkeypatch, donchian_spec):
    base = _sc(0.9, -0.05, total_return=0.10, excess_return=0.05)
    stressed = _sc(0.4, -0.05, total_return=-0.01, excess_return=-0.06)  # gone at 2x
    _stub_runs(monkeypatch, base, stressed)
    monkeypatch.setattr(promoter, "_gate", lambda c, ch, sharpe_delta_floor=None: (True, "promoted"))
    decision = promoter.decide(donchian_spec, donchian_spec, oos_bars=None)
    assert not decision.accepted and decision.reason == "fails_cost_stress"
    assert decision.challenger_stress_score.total_return == -0.01


def test_decide_promotes_when_edge_survives_2x_cost(monkeypatch, donchian_spec):
    base = _sc(0.9, -0.05, total_return=0.10, excess_return=0.05)
    stressed = _sc(0.8, -0.05, total_return=0.07, excess_return=0.03)  # still good
    _stub_runs(monkeypatch, base, stressed)
    monkeypatch.setattr(promoter, "_gate", lambda c, ch, sharpe_delta_floor=None: (True, "promoted"))
    decision = promoter.decide(donchian_spec, donchian_spec, oos_bars=None)
    assert decision.accepted and decision.reason == "promoted"
