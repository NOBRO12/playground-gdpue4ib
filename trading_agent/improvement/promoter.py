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
    if chal.sharpe - champ.sharpe < config.PROMOTE_MIN_SHARPE_DELTA:
        return False, "oos_sharpe_regression"
    # max_dd is negative; ratio of magnitudes vs champion floor of 1% to avoid div-by-tiny
    champ_dd = abs(champ.max_dd) if champ.max_dd < 0 else 0.01
    chal_dd = abs(chal.max_dd)
    if chal_dd > config.PROMOTE_MAX_DD_RATIO * champ_dd:
        return False, "oos_max_dd_regression"
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
