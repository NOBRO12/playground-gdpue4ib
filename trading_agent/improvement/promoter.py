"""Champion vs challenger promotion gate.

The proposer never sees the OOS window. The gate evaluates both specs on the
same OOS bars and only promotes when *every* criterion holds.
"""
from __future__ import annotations

import math
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


def trials_penalty(n_trials: int) -> float:
    """Extra Sharpe-delta required when the challenger is the best of N candidates.

    The expected maximum of N noisy Sharpe estimates grows ~sqrt(2*ln N), so the
    winner of a wide search looks better than it is. Penalty is 0 at N<=1.
    """
    if n_trials <= 1:
        return 0.0
    return config.PROMOTE_TRIALS_PENALTY_COEF * math.sqrt(2.0 * math.log(n_trials))


def consistency(spec: StrategySpec, bars: pd.DataFrame) -> float | None:
    """Fraction of walk-forward folds in which the spec beats buy-and-hold.

    Returns None when there aren't enough bars to fold (no evidence either way →
    the caller skips the check rather than rejecting on a data shortage).
    """
    try:
        folds = backtester.walk_forward(
            spec, bars, n_folds=config.PROMOTE_CONSISTENCY_FOLDS
        )
    except ValueError:
        return None
    if not folds:
        return None
    positive = sum(1 for f in folds if f.score.excess_return > 0)
    return positive / len(folds)


def _gate(
    champ: Scorecard, chal: Scorecard, sharpe_delta_floor: float | None = None
) -> tuple[bool, str]:
    delta_floor = (
        config.PROMOTE_MIN_SHARPE_DELTA if sharpe_delta_floor is None else sharpe_delta_floor
    )
    if chal.n_trades < config.PROMOTE_MIN_TRADES:
        return False, "insufficient_trades"
    if chal.sharpe < config.PROMOTE_MIN_ABS_SHARPE:
        return False, "absolute_sharpe_floor"
    if chal.sharpe - champ.sharpe < delta_floor:
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
    *,
    consistency_bars: pd.DataFrame | None = None,
    n_trials: int = 1,
) -> Decision:
    champ_run = backtester.run(champion, oos_bars)
    chal_run = backtester.run(challenger, oos_bars)
    delta_floor = config.PROMOTE_MIN_SHARPE_DELTA + trials_penalty(n_trials)
    accepted, reason = _gate(champ_run.score, chal_run.score, sharpe_delta_floor=delta_floor)
    # Walk-forward consistency is an extra hurdle for specs that already cleared
    # the single-window gate. Skipped when there aren't enough bars to fold.
    if accepted and consistency_bars is not None:
        frac = consistency(challenger, consistency_bars)
        if frac is not None and frac < config.PROMOTE_MIN_POSITIVE_FOLD_FRAC:
            accepted, reason = False, "inconsistent_oos"
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
