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
    # Execution-quality detail (defaults preserve back-compat for callers/tests
    # that build a Fill positionally). Populated by the Alpaca brokers after the
    # order reaches a terminal state and by the MockBroker synchronously.
    submitted_qty: float = 0.0
    filled_qty: float = 0.0
    commission: float = 0.0
    submitted_at: str = ""
    filled_at: str = ""
    status: str = ""


def _iso(value: object) -> str:
    """Best-effort ISO string for an Alpaca timestamp (datetime or str or None)."""
    if value is None:
        return ""
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


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

    def last_exit_fill(self, symbol: str) -> Fill | None:
        """Most recent filled SELL for ``symbol`` (an exchange-managed resting
        stop/take leg). None unless the venue rests orders (overridden)."""
        return None


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
        filled_qty = float(getattr(order, "filled_qty", 0) or order.qty)
        return Fill(
            symbol=symbol,
            side=side,
            qty=float(order.qty),
            avg_price=float(order.filled_avg_price or 0.0),
            order_id=str(order.id),
            submitted_qty=float(order.qty),
            filled_qty=filled_qty,
            status=str(getattr(order, "status", "")),
        )


class AlpacaStockBroker(_AlpacaBrokerBase):
    """Alpaca US-equities broker. Whole shares, DAY orders, market-hours aware.

    Equity market orders must use ``TimeInForce.DAY`` (GTC market orders are
    rejected) and whole-share quantities unless fractional trading is enabled.
    We floor to whole shares to stay safe across account types.
    """

    supports_resting_orders = True
    _poll_interval_s = 0.5  # overridden to 0 in tests to avoid real sleeps

    def _fill_from_order(self, order, symbol: str, side: str) -> Fill:
        """Build a Fill from a (terminal) Alpaca order, capturing fill price,
        partial quantity, commission, latency, and status."""
        submitted_qty = float(getattr(order, "qty", 0) or 0)
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        avg_price = float(getattr(order, "filled_avg_price", None) or 0.0)
        commission = float(getattr(order, "commission", 0) or 0)
        return Fill(
            symbol=symbol,
            side=side,
            qty=filled_qty or submitted_qty,
            avg_price=avg_price,
            order_id=str(getattr(order, "id", "")),
            submitted_qty=submitted_qty,
            filled_qty=filled_qty,
            commission=commission,
            submitted_at=_iso(getattr(order, "submitted_at", None)),
            filled_at=_iso(getattr(order, "filled_at", None)),
            status=str(getattr(order, "status", "")),
        )

    def _await_fill(self, order_id: str):
        """Poll an order until it reaches a terminal state or the timeout, so the
        returned order carries the real fill (a just-submitted order is empty)."""
        import time

        terminal = {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}
        deadline = time.monotonic() + config.ORDER_FILL_TIMEOUT_S
        order = self._client.get_order_by_id(order_id)
        while str(getattr(order, "status", "")).lower() not in terminal:
            if time.monotonic() >= deadline:
                break
            if self._poll_interval_s:
                time.sleep(self._poll_interval_s)
            order = self._client.get_order_by_id(order_id)
        return order

    def last_exit_fill(self, symbol: str) -> Fill | None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=20)
        sells = [
            o for o in self._client.get_orders(filter=req)
            if str(getattr(o, "side", "")).lower().endswith("sell")
            and str(getattr(o, "status", "")).lower() == "filled"
        ]
        if not sells:
            return None
        sells.sort(key=lambda o: _iso(getattr(o, "filled_at", None)), reverse=True)
        return self._fill_from_order(sells[0], symbol, "sell")

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
        # Wait for the parent entry to fill so we capture the real price/qty; the
        # stop/take legs rest on the exchange and are handled on later ticks.
        order = self._await_fill(order.id)
        return self._fill_from_order(order, symbol, side)

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
        order = self._await_fill(order.id)
        return self._fill_from_order(order, symbol, side)

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def daytrade_count(self) -> int:
        """Alpaca's rolling 5-business-day day-trade count for the account."""
        return int(getattr(self._client.get_account(), "daytrade_count", 0) or 0)
