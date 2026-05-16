from __future__ import annotations

from trading_agent.guardrails.risk import Decision, Order, PortfolioState, allow


def _state(**overrides) -> PortfolioState:
    base = dict(
        equity_usd=100_000.0,
        peak_equity_usd=100_000.0,
        day_start_equity_usd=100_000.0,
        trades_today=0,
        current_qty=0.0,
        kill_switch_tripped=False,
    )
    base.update(overrides)
    return PortfolioState(**base)


def test_allow_buy_within_limits():
    d = allow(Order("BTC/USD", "buy", 0.5, 15_000.0), _state())
    assert d == Decision(True, "")


def test_kill_switch_blocks():
    d = allow(Order("BTC/USD", "buy", 0.1, 5_000.0), _state(kill_switch_tripped=True))
    assert not d.allowed and d.reason == "kill_switch_tripped"


def test_hard_dd_blocks():
    d = allow(
        Order("BTC/USD", "buy", 0.1, 5_000.0),
        _state(equity_usd=89_000.0, peak_equity_usd=100_000.0),
    )
    assert not d.allowed and d.reason == "hard_dd_kill"


def test_daily_loss_cap():
    d = allow(
        Order("BTC/USD", "buy", 0.1, 5_000.0),
        _state(equity_usd=96_000.0, day_start_equity_usd=100_000.0),
    )
    assert not d.allowed and d.reason == "daily_loss_cap"


def test_max_trades_per_day():
    d = allow(Order("BTC/USD", "buy", 0.1, 5_000.0), _state(trades_today=8))
    assert not d.allowed and d.reason == "max_trades_per_day"


def test_max_position_pct():
    d = allow(Order("BTC/USD", "buy", 1.0, 30_000.0), _state())
    assert not d.allowed and d.reason == "max_position_pct"


def test_exit_full_position_allowed():
    d = allow(Order("BTC/USD", "sell", 0.1, 5_000.0), _state(current_qty=0.1))
    assert d.allowed


def test_naked_short_blocked_when_flat():
    d = allow(Order("BTC/USD", "sell", 0.1, 5_000.0), _state(current_qty=0.0))
    assert not d.allowed and d.reason == "would_open_short"


def test_sell_more_than_held_blocked():
    d = allow(Order("BTC/USD", "sell", 0.2, 10_000.0), _state(current_qty=0.1))
    assert not d.allowed and d.reason == "would_open_short"


def test_exits_not_blocked_by_daily_loss_cap():
    d = allow(
        Order("BTC/USD", "sell", 0.1, 5_000.0),
        _state(
            current_qty=0.1,
            equity_usd=96_000.0,
            day_start_equity_usd=100_000.0,
        ),
    )
    assert d.allowed
