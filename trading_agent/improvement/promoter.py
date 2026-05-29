"""Champion vs challenger promotion gate.

The proposer never sees the OOS window. The gate evaluates both specs on the
same OOS bars and only promotes when *every* criterion holds.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config
from ..evaluation.metrics import Scorecard
from ..strategy.spec import StrategySpec
from . import backtester


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    champion_score: Scorecard
    challenger_score: Scorecard


def _gate(champ: Scorecard, chal: Scorecard) -> tuple[bool, str]:
    if chal.n_trades < config.PROMOTE_MIN_TRADES:
        return False, "insufficient_trades"
    if chal.sharpe < config.PROMOTE_MIN_ABS_SHARPE:
        return False, "absolute_sharpe_floor"
    if chal.sharpe - champ.sharpe < config.PROMOTE_MIN_SHARPE_DELTA:
        return False, "oos_sharpe_regression"
    # max_dd is negative; ratio of magnitudes vs champion floor of 1% to avoid div-by-tiny
    champ_dd = abs(champ.max_dd) if champ.max_dd < 0 else 0.01
    chal_dd = abs(chal.max_dd)
    if chal_dd > config.PROMOTE_MAX_DD_RATIO * champ_dd:
        return False, "oos_max_dd_regression"
    # Benchmark-relative: must make money AND beat buy-and-hold. Taking active
    # risk to underperform the index is a losing trade by definition.
    if config.PROMOTE_REQUIRE_POSITIVE_RETURN and chal.total_return <= 0:
        return False, "negative_absolute_return"
    if chal.excess_return < config.PROMOTE_MIN_EXCESS_RETURN:
        return False, "underperforms_benchmark"
    return True, "promoted"


def decide(
    champion: StrategySpec,
    challenger: StrategySpec,
    oos_bars: pd.DataFrame,
) -> Decision:
    champ_run = backtester.run(champion, oos_bars)
    chal_run = backtester.run(challenger, oos_bars)
    accepted, reason = _gate(champ_run.score, chal_run.score)
    return Decision(
        accepted=accepted,
        reason=reason,
        champion_score=champ_run.score,
        challenger_score=chal_run.score,
    )


def rank_challengers(
    champion: StrategySpec,
    challengers: list[StrategySpec],
    oos_bars: pd.DataFrame,
) -> list[tuple[StrategySpec, Decision]]:
    """Score each challenger on the OOS window, best first.

    Ordering: gate-passers ahead of failers, then by OOS Sharpe. The caller
    promotes the top entry only if its decision is accepted.
    """
    scored = [(c, decide(champion, c, oos_bars)) for c in challengers]
    scored.sort(
        key=lambda cd: (cd[1].accepted, cd[1].challenger_score.sharpe), reverse=True
    )
    return scored
