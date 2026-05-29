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
    enums_mod.OrderClass = types.SimpleNamespace(BRACKET="bracket")
    enums_mod.QueryOrderStatus = types.SimpleNamespace(OPEN="open")

    class _Req:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    requests_mod.MarketOrderRequest = _Req
    requests_mod.StopLossRequest = _Req
    requests_mod.TakeProfitRequest = _Req
    requests_mod.GetOrdersRequest = _Req

    client_mod.submitted = []
    client_mod.cancelled = []
    # Symbol -> qty the fake account already holds (for reconciliation tests).
    client_mod.held_positions = {}
    # Open orders the fake exchange reports back for cancel_open_orders.
    client_mod.open_orders = []

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

        def get_all_positions(self):
            return [
                types.SimpleNamespace(symbol=sym, qty=str(qty))
                for sym, qty in client_mod.held_positions.items()
            ]

        def get_orders(self, filter=None):
            client_mod.last_orders_filter = filter
            return list(client_mod.open_orders)

        def cancel_order_by_id(self, order_id):
            client_mod.cancelled.append(order_id)

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


def test_bracket_builds_correct_request_shape(fake_alpaca):
    """The resting-bracket entry must carry GTC + BRACKET class + both legs.

    This pins the exact request the broker hands to Alpaca so the shape can't
    silently drift; the server-side TIF acceptance is what `paper --once`
    confirms live (see docs/paper_verification.md)."""
    fill = _broker().submit_bracket("SPY", "buy", 3.7, stop_px=590.123, take_px=620.987)
    assert fill.qty == 3.0  # SimpleNamespace echoes whole-share qty
    req = fake_alpaca.submitted[-1]
    assert req.qty == 3  # floored to whole shares
    assert req.side == "buy"
    assert req.time_in_force == "gtc"  # legs persist across days
    assert req.order_class == "bracket"
    # Legs rounded to the penny that the exchange accepts.
    assert req.stop_loss.stop_price == 590.12
    assert req.take_profit.limit_price == 620.99


def test_bracket_subshare_raises(fake_alpaca):
    with pytest.raises(ValueError, match="floors to 0 shares"):
        _broker().submit_bracket("SPY", "buy", 0.4, stop_px=1.0, take_px=2.0)


def test_cancel_open_orders_filters_by_symbol_and_cancels(fake_alpaca):
    import types

    fake_alpaca.open_orders = [
        types.SimpleNamespace(id="o1"),
        types.SimpleNamespace(id="o2"),
    ]
    _broker().cancel_open_orders("SPY")
    # Queried only OPEN orders for the one symbol...
    assert fake_alpaca.last_orders_filter.status == "open"
    assert fake_alpaca.last_orders_filter.symbols == ["SPY"]
    # ...and cancelled every one returned.
    assert fake_alpaca.cancelled == ["o1", "o2"]


def test_positions_parses_held_qty_for_reconciliation(fake_alpaca):
    """Startup reconciliation seeds held_qty from the real broker; confirm the
    qty string Alpaca returns is parsed to a float keyed by symbol."""
    fake_alpaca.held_positions = {"SPY": 5, "AAPL": 2}
    assert _broker().positions() == {"SPY": 5.0, "AAPL": 2.0}
