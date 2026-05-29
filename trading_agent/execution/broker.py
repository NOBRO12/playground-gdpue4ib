"""Thin wrappers over Alpaca's trading API (crypto and US equities).

Paper mode by default. Set ``AGENT_LIVE_MODE=true`` to route orders to the
real-money live endpoint. All guardrails in ``trading_agent.guardrails.risk``
apply identically in both modes.

``AlpacaCryptoBroker`` trades 24/7 with fractional quantities and GTC orders.
``AlpacaStockBroker`` trades whole shares with DAY orders, respects market
hours, and exposes Alpaca's rolling day-trade count so the PDT guardrail can
fire.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .. import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    qty: float
    avg_price: float
    order_id: str


def make_client_order_id(symbol: str, side: str, ts_iso: str) -> str:
    """Deterministic idempotency key: replaying the same bar's order is a no-op
    because Alpaca rejects a duplicate client_order_id. Sanitized + length-capped."""
    raw = f"{symbol}-{side}-{ts_iso}"
    return "".join(c for c in raw if c.isalnum() or c in "-_.")[:128]


class _AlpacaBrokerBase:
    """Shared client setup + account/position reads for both asset classes."""

    # Whether the venue accepts resting stop/take-profit (bracket) orders that
    # fire between ticks. Stocks: yes. Crypto/Mock: no (the live loop simulates
    # the stop at tick cadence instead, matching the backtester).
    supports_resting_orders = False

    def __init__(self) -> None:
        from alpaca.trading.client import TradingClient

        s = config.load()
        if not s.alpaca_key or not s.alpaca_secret:
            raise RuntimeError(
                "ALPACA_KEY/ALPACA_SECRET not set; broker requires keys at runtime"
            )
        if s.live_mode:
            log.warning(
                "LIVE MODE ACTIVE — orders execute against the real Alpaca endpoint "
                "using real funds. Set AGENT_LIVE_MODE=false to revert to paper."
            )
        self._client = TradingClient(s.alpaca_key, s.alpaca_secret, paper=not s.live_mode)

    def equity(self) -> float:
        return float(self._client.get_account().equity)

    def cash(self) -> float:
        return float(self._client.get_account().cash)

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self._client.get_all_positions()}

    def mark(self, symbol: str, px: float) -> None:
        """No-op: Alpaca prices its own positions. Present for interface parity."""

    def is_market_open(self) -> bool:  # overridden for stocks
        return True

    def daytrade_count(self) -> int:  # overridden for stocks
        return 0

    def cancel_open_orders(self, symbol: str) -> None:
        """Cancel resting orders for a symbol. No-op when unsupported."""


class AlpacaCryptoBroker(_AlpacaBrokerBase):
    """Alpaca crypto broker. 24/7, fractional qty, GTC market orders."""

    def submit_market(
        self, symbol: str, side: str, qty: float, client_order_id: str | None = None
    ) -> Fill:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )
        order = self._client.submit_order(req)
        return Fill(
            symbol=symbol,
            side=side,
            qty=float(order.qty),
            avg_price=float(order.filled_avg_price or 0.0),
            order_id=str(order.id),
        )


class AlpacaStockBroker(_AlpacaBrokerBase):
    """Alpaca US-equities broker. Whole shares, DAY orders, market-hours aware.

    Equity market orders must use ``TimeInForce.DAY`` (GTC market orders are
    rejected) and whole-share quantities unless fractional trading is enabled.
    We floor to whole shares to stay safe across account types.
    """

    supports_resting_orders = True

    def submit_bracket(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_px: float,
        take_px: float,
        client_order_id: str | None = None,
    ) -> Fill:
        """Market entry with resting stop-loss + take-profit (OCO) legs.

        The stop and take-profit rest on the exchange and fire intraday between
        ticks — essential for a once-per-day scheduler, and what makes live
        match the backtester (which assumes intrabar stop/take fills).
        GTC so the legs persist across days for a swing strategy. (Bracket TIF
        is validated live via `paper --once`.)
        """
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        whole = int(qty)
        if whole <= 0:
            raise ValueError(
                f"stock order qty {qty} floors to 0 shares; position too small to trade"
            )
        req = MarketOrderRequest(
            symbol=symbol,
            qty=whole,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_px, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_px, 2)),
            client_order_id=client_order_id,
        )
        order = self._client.submit_order(req)
        return Fill(
            symbol=symbol,
            side=side,
            qty=float(order.qty),
            avg_price=float(order.filled_avg_price or 0.0),
            order_id=str(order.id),
        )

    def cancel_open_orders(self, symbol: str) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        for o in self._client.get_orders(filter=req):
            self._client.cancel_order_by_id(o.id)

    def submit_market(
        self, symbol: str, side: str, qty: float, client_order_id: str | None = None
    ) -> Fill:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        whole = int(qty)  # floor to whole shares
        if whole <= 0:
            raise ValueError(
                f"stock order qty {qty} floors to 0 shares; position too small to trade"
            )
        req = MarketOrderRequest(
            symbol=symbol,
            qty=whole,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self._client.submit_order(req)
        return Fill(
            symbol=symbol,
            side=side,
            qty=float(order.qty),
            avg_price=float(order.filled_avg_price or 0.0),
            order_id=str(order.id),
        )

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def daytrade_count(self) -> int:
        """Alpaca's rolling 5-business-day day-trade count for the account."""
        return int(getattr(self._client.get_account(), "daytrade_count", 0) or 0)
