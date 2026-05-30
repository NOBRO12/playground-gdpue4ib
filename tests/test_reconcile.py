from __future__ import annotations

from trading_agent import config
from trading_agent.cli import _reconcile_start
from trading_agent.evaluation import db
from trading_agent.execution.broker import make_client_order_id
from trading_agent.execution.mock_broker import MockBroker
from trading_agent.execution.state import LiveState
from trading_agent.strategy.spec import StrategySpec


def _champion() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "type": "donchian",
            "symbol": "SPY",
            "timeframe": "1d",
            "params": {"entry_lookback": 20, "exit_lookback": 10},
            "filters": {},
            "risk": {"atr_mult_stop": 2.0, "take_profit_r": 3.0, "position_pct": 0.10},
            "version": "v0",
        }
    )


def _settings(tmp_path, monkeypatch) -> config.Settings:
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_LIVE_STATE_PATH", str(tmp_path / "live_state.json"))
    monkeypatch.setenv("AGENT_MOCK_BROKER_PATH", str(tmp_path / "mock_broker.json"))
    monkeypatch.setenv("AGENT_BROKER", "mock")
    s = config.load()
    db.init(s.db_path)
    return s


def test_client_order_id_is_deterministic_and_sanitized():
    a = make_client_order_id("BTC/USD", "buy", "2024-05-01T00:00:00+00:00")
    b = make_client_order_id("BTC/USD", "buy", "2024-05-01T00:00:00+00:00")
    assert a == b  # same inputs → same key (Alpaca rejects duplicates → no double-submit)
    assert "/" not in a and ":" not in a and " " not in a
    assert len(a) <= 128


def test_reconcile_clears_tracked_position_when_broker_flat(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    state = LiveState(open_entry_px=450.0, open_stop_px=440.0, open_take_px=480.0)
    state.save(s.live_state_path)
    broker = MockBroker(state_path=s.mock_broker_path)  # flat

    _reconcile_start(broker, _champion(), s)

    reloaded = LiveState.load(s.live_state_path)
    assert reloaded.open_entry_px == 0.0
    assert reloaded.open_stop_px == 0.0


def test_reconcile_leaves_state_when_broker_holds_untracked(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    LiveState().save(s.live_state_path)  # flat state
    broker = MockBroker(state_path=s.mock_broker_path)
    broker.mark("SPY", 450.0)
    broker.submit_market("SPY", "buy", 1)  # broker now holds, state doesn't know

    _reconcile_start(broker, _champion(), s)  # should warn, not crash

    reloaded = LiveState.load(s.live_state_path)
    assert reloaded.open_entry_px == 0.0  # unchanged; we don't fabricate an entry
