from __future__ import annotations

from trading_agent import config
from trading_agent.evaluation import db, logger as evlogger


def test_slippage_bps_sign_convention():
    # Buying above intended is a cost (positive); below is a gain (negative).
    assert evlogger.slippage_bps("buy", 100.0, 100.5) == 50.0
    assert evlogger.slippage_bps("buy", 100.0, 99.5) == -50.0
    # Selling below intended is the cost on exits.
    assert evlogger.slippage_bps("sell", 100.0, 99.5) == 50.0
    assert evlogger.slippage_bps("sell", 100.0, 100.5) == -50.0
    assert evlogger.slippage_bps("buy", 0.0, 100.0) == 0.0  # guard


def test_log_execution_persists_and_derives_slippage(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    s = config.load()
    db.init(s.db_path)
    with db.session(s.db_path) as conn:
        evlogger.log_execution(
            conn,
            version="v1", symbol="SPY", side="buy", order_type="market",
            intended_px=600.0, fill_px=600.30, intended_qty=10, filled_qty=10,
            commission_usd=0.0, latency_ms=42.0, client_order_id="c1",
            order_id="o1", status="filled",
        )
    with db.session(s.db_path) as conn:
        row = conn.execute("SELECT * FROM executions").fetchone()
    assert row["symbol"] == "SPY"
    assert row["order_type"] == "market"
    assert abs(row["slippage_bps"] - 5.0) < 1e-9  # 0.30/600 = 5 bps, a cost
    assert row["filled_qty"] == 10
    assert row["latency_ms"] == 42.0
