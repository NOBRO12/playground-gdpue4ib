from __future__ import annotations

from trading_agent.evaluation import db, logger


def test_close_open_trade_updates_row(tmp_path):
    db_path = tmp_path / "agent.db"
    db.init(db_path)
    with db.session(db_path) as conn:
        tid = logger.log_trade(
            conn,
            version="v1",
            symbol="BTC/USD",
            side="long",
            qty=0.5,
            entry_ts="2024-05-01T00:00:00+00:00",
            entry_px=30_000.0,
            reason_entry="donchian_breakout",
            regime="trend",
            meta={},
        )
        assert tid > 0
        closed = logger.close_open_trade(
            conn,
            symbol="BTC/USD",
            version="v1",
            exit_ts="2024-05-01T05:00:00+00:00",
            exit_px=31_000.0,
            reason_exit="signal_exit",
        )
    assert closed
    with db.session(db_path) as conn:
        row = conn.execute("SELECT * FROM trades WHERE id = ?", (tid,)).fetchone()
    assert row["exit_px"] == 31_000.0
    assert row["pnl_usd"] == (31_000.0 - 30_000.0) * 0.5
    assert row["reason_exit"] == "signal_exit"


def test_close_open_trade_no_match(tmp_path):
    db_path = tmp_path / "agent.db"
    db.init(db_path)
    with db.session(db_path) as conn:
        closed = logger.close_open_trade(
            conn,
            symbol="ETH/USD",
            version="v1",
            exit_ts="2024-05-01T05:00:00+00:00",
            exit_px=1.0,
            reason_exit="signal_exit",
        )
    assert closed is False
