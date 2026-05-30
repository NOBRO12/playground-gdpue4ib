from __future__ import annotations

from trading_agent import config
from trading_agent.cli import _latency_ms, _record_execution
from trading_agent.evaluation import db
from trading_agent.execution.mock_broker import MockBroker


def test_latency_ms_handles_pair_and_missing():
    a = "2024-01-02T14:30:00+00:00"
    b = "2024-01-02T14:30:00.250000+00:00"
    assert _latency_ms(a, b) == 250.0
    assert _latency_ms("", b) is None
    assert _latency_ms(a, "not-a-time") is None


def test_record_execution_persists_real_slippage_from_mock_fill(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_MOCK_BROKER_PATH", str(tmp_path / "mb.json"))
    s = config.load()
    db.init(s.db_path)

    broker = MockBroker(state_path=s.mock_broker_path)
    broker.mark("SPY", 600.0)
    fill = broker.submit_market("SPY", "buy", 2)  # mock adds 15bps -> fill 600.90

    with db.session(s.db_path) as conn:
        _record_execution(
            conn, version="v0", symbol="SPY", side="buy", order_type="market",
            intended_px=600.0, fill=fill, client_order_id="c1",
        )
        row = conn.execute("SELECT * FROM executions").fetchone()

    assert row["order_type"] == "market"
    assert abs(row["fill_px"] - 600.9) < 1e-6
    assert abs(row["slippage_bps"] - 15.0) < 1e-6   # 0.90/600 = 15 bps, the mock cost
    assert row["filled_qty"] == 2.0
    assert row["commission_usd"] == 0.0
    assert row["latency_ms"] == 0.0                 # mock fills instantly
    assert row["status"] == "filled"
