from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_alpaca(monkeypatch):
    """Install a fake alpaca SDK + paper keys so AlpacaStockBroker runs offline.

    Returns the fake trading-client module; submitted orders land in
    ``mod.submitted`` so tests can assert on qty and time_in_force.
    """
    monkeypatch.setenv("ALPACA_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")
    monkeypatch.setenv("AGENT_LIVE_MODE", "false")

    pkg = types.ModuleType("alpaca")
    trading = types.ModuleType("alpaca.trading")
    client_mod = types.ModuleType("alpaca.trading.client")
    enums_mod = types.ModuleType("alpaca.trading.enums")
    requests_mod = types.ModuleType("alpaca.trading.requests")

    enums_mod.OrderSide = types.SimpleNamespace(BUY="buy", SELL="sell")
    enums_mod.TimeInForce = types.SimpleNamespace(DAY="day", GTC="gtc")

    class _MarketOrderRequest:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    requests_mod.MarketOrderRequest = _MarketOrderRequest

    client_mod.submitted = []

    class _TradingClient:
        def __init__(self, key, secret, paper=True):
            self.paper = paper

        def submit_order(self, req):
            client_mod.submitted.append(req)
            return types.SimpleNamespace(
                qty=req.qty, filled_avg_price="601.00", id="ord_1"
            )

        def get_clock(self):
            return types.SimpleNamespace(is_open=True)

        def get_account(self):
            return types.SimpleNamespace(
                daytrade_count=2, equity="10000.0", cash="9000.0"
            )

    client_mod.TradingClient = _TradingClient

    monkeypatch.setitem(sys.modules, "alpaca", pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", enums_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", requests_mod)
    return client_mod


def _broker():
    from trading_agent.execution.broker import AlpacaStockBroker

    return AlpacaStockBroker()


def test_floors_to_whole_shares_and_uses_day_tif(fake_alpaca):
    fill = _broker().submit_market("SPY", "buy", 2.9)
    assert fill.qty == 2.0  # floored from 2.9
    req = fake_alpaca.submitted[-1]
    assert req.qty == 2  # whole shares submitted
    assert req.time_in_force == "day"  # not GTC
    assert req.side == "buy"


def test_subshare_order_raises(fake_alpaca):
    with pytest.raises(ValueError, match="floors to 0 shares"):
        _broker().submit_market("SPY", "buy", 0.5)


def test_is_market_open_reads_clock(fake_alpaca):
    assert _broker().is_market_open() is True


def test_daytrade_count_reads_account(fake_alpaca):
    assert _broker().daytrade_count() == 2
