from __future__ import annotations

from trading_agent import config
from trading_agent.cli import _realized_edge
from trading_agent.evaluation import db, logger as evlogger


def _trade(conn, pnl: float, i: int) -> None:
    evlogger.log_trade(
        conn, version="v1", symbol="SPY", qty=1,
        entry_ts=f"2024-01-{i:02d}T00:00:00+00:00", entry_px=100.0,
    )
    # close_open_trade sets pnl_usd = (exit_px - entry_px) * qty, so pick exit_px
    # to realize the desired pnl on a 1-share, $100-entry trade.
    evlogger.close_open_trade(
        conn, symbol="SPY", version="v1",
        exit_ts=f"2024-01-{i:02d}T12:00:00+00:00", exit_px=100.0 + pnl,
        reason_exit="signal_exit",
    )


def test_realized_edge_computes_win_rate_and_payoff(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    s = config.load()
    db.init(s.db_path)
    with db.session(s.db_path) as conn:
        # 3 wins of +20, 2 losses of -10 -> p=0.6, payoff = 20/10 = 2.0
        for i, pnl in enumerate([20, 20, 20, -10, -10], start=1):
            _trade(conn, pnl, i)
        n, win_rate, payoff = _realized_edge(conn, "SPY", "v1")
    assert n == 5
    assert abs(win_rate - 0.6) < 1e-9
    assert abs(payoff - 2.0) < 1e-9


def test_realized_edge_none_without_both_wins_and_losses(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    s = config.load()
    db.init(s.db_path)
    with db.session(s.db_path) as conn:
        for i, pnl in enumerate([5, 5, 5], start=1):  # all wins, no losses
            _trade(conn, pnl, i)
        n, win_rate, payoff = _realized_edge(conn, "SPY", "v1")
    assert n == 3 and win_rate is None and payoff is None
