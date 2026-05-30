from __future__ import annotations

from trading_agent.evaluation.metrics import Scorecard
from trading_agent.improvement import promoter
from trading_agent.strategy.spec import StrategySpec


def _spec(version: str) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "type": "donchian",
            "symbol": "SPY",
            "timeframe": "1d",
            "params": {"entry_lookback": 20, "exit_lookback": 10},
            "filters": {},
            "risk": {"atr_mult_stop": 2.0, "take_profit_r": 3.0, "position_pct": 0.10},
            "version": version,
        }
    )


def _score(sharpe: float) -> Scorecard:
    return Scorecard(sharpe, sharpe, -0.05, 40, 0.5, 0.1, 100_000.0)


def test_rank_puts_passers_first_then_by_sharpe(monkeypatch):
    champ = _spec("champ")
    a, b, c = _spec("v-a"), _spec("v-b"), _spec("v-c")
    table = {
        "v-a": (False, 0.9),  # highest sharpe but rejected
        "v-b": (True, 0.3),   # passes, lower sharpe
        "v-c": (True, 0.5),   # passes, higher sharpe → should win
    }

    def fake_decide(champion, challenger, oos_bars):
        accepted, sharpe = table[challenger.version]
        return promoter.Decision(accepted, "x", _score(0.2), _score(sharpe))

    monkeypatch.setattr(promoter, "decide", fake_decide)
    ranked = promoter.rank_challengers(champ, [a, b, c], oos_bars=None)
    assert [spec.version for spec, _ in ranked] == ["v-c", "v-b", "v-a"]
    assert ranked[0][1].accepted is True
