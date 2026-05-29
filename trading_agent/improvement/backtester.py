"""Deterministic vectorized backtester.

Long-only, one position at a time per symbol, market-on-bar-close fills,
constant ``fee_bps`` and ``slippage_bps`` costs. Returns the equity curve
and trade ledger so :mod:`evaluation.metrics` can score the run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..evaluation import metrics, regime
from ..execution import exits, sizing
from ..strategy.base import Strategy
from ..strategy.registry import from_spec
from ..strategy.spec import StrategySpec

PERIODS_PER_YEAR = {"1h": 365 * 24, "4h": 365 * 6, "1d": 365}


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    score: metrics.Scorecard


def run(
    spec: StrategySpec,
    bars: pd.DataFrame,
    starting_equity: float = 100_000.0,
    fee_bps: float = config.BACKTEST_FEE_BPS,
    slippage_bps: float = config.BACKTEST_SLIPPAGE_BPS,
) -> BacktestResult:
    strat: Strategy = from_spec(spec)
    sig = strat.signals(bars)
    regime_at = regime.classify(bars).to_dict()

    cost = (fee_bps + slippage_bps) / 1e4
    pos_pct = spec.risk.position_pct
    tp_r = spec.risk.take_profit_r

    cash = starting_equity
    equity_pts: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []

    in_pos = False
    qty = 0.0
    entry_px = 0.0
    stop_px = float("nan")
    take_px = float("nan")
    entry_ts: pd.Timestamp | None = None
    entry_reason = ""
    risk_per_unit = 0.0

    for ts, row in bars.iterrows():
        price = float(row["close"])

        # Open positions: check stop or take-profit first (intrabar conservative).
        if in_pos:
            low = float(row["low"])
            high = float(row["high"])
            exit_reason, exit_px = exits.exit_for_levels(low, high, stop_px, take_px)
            if exit_px is None and bool(sig.at[ts, "exit"]):
                exit_px = price
                exit_reason = "signal_exit"

            if exit_px is not None:
                net_exit = exit_px * (1 - cost)
                pnl_usd = (net_exit - entry_px) * qty
                pnl_r = (net_exit - entry_px) / risk_per_unit if risk_per_unit > 0 else 0.0
                cash += qty * net_exit
                trades.append(
                    {
                        "entry_ts": entry_ts.isoformat() if entry_ts else "",
                        "entry_px": entry_px,
                        "exit_ts": ts.isoformat(),
                        "exit_px": exit_px,
                        "qty": qty,
                        "pnl_usd": pnl_usd,
                        "pnl_r": pnl_r,
                        "reason_entry": entry_reason,
                        "reason_exit": exit_reason,
                        "regime": str(regime_at.get(entry_ts, "")) if entry_ts is not None else "",
                    }
                )
                in_pos = False
                qty = 0.0
                stop_px = float("nan")
                take_px = float("nan")

        # Entry on the same bar as a fresh signal (after any exit has flattened).
        if (not in_pos) and bool(sig.at[ts, "entry"]):
            stop_candidate = float(sig.at[ts, "stop_px"])
            if np.isnan(stop_candidate) or stop_candidate >= price:
                continue
            risk_per_unit = price - stop_candidate
            qty = sizing.size_position(
                cash, price, risk_per_unit, pos_pct, spec.risk.risk_per_trade_pct
            )
            if qty <= 0:
                continue
            entry_px = price * (1 + cost)
            cash -= qty * entry_px
            stop_px = stop_candidate
            take_px = entry_px + tp_r * risk_per_unit
            entry_ts = ts
            entry_reason = str(sig.at[ts, "reason"])
            in_pos = True

        mark = qty * price if in_pos else 0.0
        equity_pts.append((ts, cash + mark))

    equity = pd.Series(dict(equity_pts), name="equity").sort_index()
    trades_df = pd.DataFrame(trades)
    # Buy-and-hold of the same instrument/window — the benchmark every strategy
    # must beat to justify the risk it takes.
    closes = bars["close"]
    benchmark_return = (
        float(closes.iloc[-1] / closes.iloc[0] - 1.0)
        if len(closes) >= 2 and closes.iloc[0] > 0
        else 0.0
    )
    score = metrics.score(
        equity, trades_df, PERIODS_PER_YEAR[spec.timeframe], benchmark_return
    )
    return BacktestResult(equity=equity, trades=trades_df, score=score)


def walk_forward(
    spec: StrategySpec,
    bars: pd.DataFrame,
    n_folds: int = 4,
    starting_equity: float = 100_000.0,
) -> list[BacktestResult]:
    """Rolling out-of-sample folds inside the IS window.

    Each fold trains on nothing (rules-only system) and evaluates on the test
    slice. Used as a robustness check before the held-out OOS pass.
    """
    if len(bars) < n_folds * 50:
        raise ValueError("not enough bars for the requested fold count")
    fold_size = len(bars) // n_folds
    results: list[BacktestResult] = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(bars)
        results.append(run(spec, bars.iloc[start:end], starting_equity=starting_equity))
    return results
