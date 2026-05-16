from __future__ import annotations

from trading_agent.execution.state import LiveState


def test_load_missing_returns_defaults(tmp_path):
    state = LiveState.load(tmp_path / "live_state.json")
    assert state == LiveState()


def test_round_trip(tmp_path):
    path = tmp_path / "live_state.json"
    state = LiveState(
        peak_equity_usd=110_000.0,
        day_start_equity_usd=105_000.0,
        day_start_date="2024-05-01",
        trades_today=3,
        kill_switch_tripped=True,
    )
    state.save(path)
    loaded = LiveState.load(path)
    assert loaded == state


def test_observe_rolls_day(tmp_path):
    state = LiveState(day_start_date="2024-05-01", day_start_equity_usd=100_000.0, trades_today=5)
    state.observe(99_000.0, "2024-05-02")
    assert state.day_start_date == "2024-05-02"
    assert state.day_start_equity_usd == 99_000.0
    assert state.trades_today == 0


def test_observe_updates_peak(tmp_path):
    state = LiveState(peak_equity_usd=100_000.0, day_start_date="2024-05-01")
    state.observe(105_000.0, "2024-05-01")
    assert state.peak_equity_usd == 105_000.0
    state.observe(102_000.0, "2024-05-01")  # ratcheting down does not lower peak
    assert state.peak_equity_usd == 105_000.0


def test_observe_no_op_within_same_day(tmp_path):
    state = LiveState(day_start_date="2024-05-01", day_start_equity_usd=100_000.0, trades_today=2)
    state.observe(98_000.0, "2024-05-01")
    assert state.day_start_equity_usd == 100_000.0
    assert state.trades_today == 2
