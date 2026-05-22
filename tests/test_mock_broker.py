from __future__ import annotations

import pytest

from trading_agent.execution.mock_broker import MockBroker


def _broker(tmp_path, starting_cash: float = 100_000.0) -> MockBroker:
    return MockBroker(
        state_path=tmp_path / "mock_broker.json",
        starting_cash=starting_cash,
        fee_bps=10.0,
        slippage_bps=5.0,
    )


def test_initial_state(tmp_path):
    b = _broker(tmp_path)
    assert b.cash() == 100_000.0
    assert b.positions() == {}
    b.mark("BTC/USD", 30_000.0)
    assert b.equity() == 100_000.0  # flat, equity == cash


def test_buy_then_sell_round_trip(tmp_path):
    b = _broker(tmp_path)
    b.mark("BTC/USD", 30_000.0)
    fill_buy = b.submit_market("BTC/USD", "buy", 0.5)
    # cost = 15 bps, fill px slightly above mark.
    assert fill_buy.avg_price > 30_000.0
    assert b.positions() == {"BTC/USD": 0.5}
    assert b.cash() < 100_000.0

    # Price up 10%; sell at the new mark.
    b.mark("BTC/USD", 33_000.0)
    eq_before_sell = b.equity()
    fill_sell = b.submit_market("BTC/USD", "sell", 0.5)
    assert fill_sell.avg_price < 33_000.0  # cost on the way out
    assert b.positions() == {}
    # Equity drops by exactly the transaction cost on the exit (qty * mark * total_bps).
    expected_drop = 0.5 * 33_000.0 * (10 + 5) / 1e4
    assert abs((eq_before_sell - b.equity()) - expected_drop) < 1e-6


def test_naked_short_rejected(tmp_path):
    b = _broker(tmp_path)
    b.mark("BTC/USD", 30_000.0)
    with pytest.raises(RuntimeError, match="naked short"):
        b.submit_market("BTC/USD", "sell", 0.1)


def test_oversize_sell_rejected(tmp_path):
    b = _broker(tmp_path)
    b.mark("BTC/USD", 30_000.0)
    b.submit_market("BTC/USD", "buy", 0.1)
    with pytest.raises(RuntimeError, match="naked short"):
        b.submit_market("BTC/USD", "sell", 0.2)


def test_insufficient_cash_rejected(tmp_path):
    b = _broker(tmp_path, starting_cash=1_000.0)
    b.mark("BTC/USD", 30_000.0)
    with pytest.raises(RuntimeError, match="insufficient cash"):
        b.submit_market("BTC/USD", "buy", 1.0)


def test_submit_without_mark_raises(tmp_path):
    b = _broker(tmp_path)
    with pytest.raises(RuntimeError, match="no mark"):
        b.submit_market("BTC/USD", "buy", 0.1)


def test_state_persists_across_instances(tmp_path):
    state = tmp_path / "mock_broker.json"
    b1 = MockBroker(state_path=state, starting_cash=100_000.0)
    b1.mark("BTC/USD", 30_000.0)
    b1.submit_market("BTC/USD", "buy", 0.25)
    cash_after = b1.cash()
    positions_after = b1.positions()

    b2 = MockBroker(state_path=state, starting_cash=999_999.0)  # ignored: file exists
    assert b2.cash() == cash_after
    assert b2.positions() == positions_after


def test_equity_reflects_marks(tmp_path):
    b = _broker(tmp_path)
    b.mark("BTC/USD", 30_000.0)
    b.submit_market("BTC/USD", "buy", 1.0)
    eq_at_entry = b.equity()
    b.mark("BTC/USD", 33_000.0)  # +10% move
    eq_after_pump = b.equity()
    assert eq_after_pump > eq_at_entry


def test_weighted_avg_entry_on_second_buy(tmp_path):
    b = _broker(tmp_path)
    b.mark("BTC/USD", 30_000.0)
    b.submit_market("BTC/USD", "buy", 1.0)
    b.mark("BTC/USD", 40_000.0)
    b.submit_market("BTC/USD", "buy", 1.0)
    assert b.positions()["BTC/USD"] == pytest.approx(2.0)
    # Avg entry is between the two marks (plus fee).
    payload = (tmp_path / "mock_broker.json").read_text()
    assert "avg_entry_px" in payload
