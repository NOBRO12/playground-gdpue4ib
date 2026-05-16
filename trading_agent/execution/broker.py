"""Thin wrapper over Alpaca's paper crypto trading API."""
from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    qty: float
    avg_price: float
    order_id: str


class AlpacaCryptoBroker:
    """Paper trading only. `paper=True` is hard-coded."""

    def __init__(self) -> None:
        from alpaca.trading.client import TradingClient

        s = config.load()
        if not s.alpaca_key or not s.alpaca_secret:
            raise RuntimeError(
                "ALPACA_KEY/ALPACA_SECRET not set; broker requires keys at runtime"
            )
        self._client = TradingClient(s.alpaca_key, s.alpaca_secret, paper=True)

    def submit_market(self, symbol: str, side: str, qty: float) -> Fill:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        order = self._client.submit_order(req)
        return Fill(
            symbol=symbol,
            side=side,
            qty=float(order.qty),
            avg_price=float(order.filled_avg_price or 0.0),
            order_id=str(order.id),
        )

    def equity(self) -> float:
        return float(self._client.get_account().equity)

    def cash(self) -> float:
        return float(self._client.get_account().cash)

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self._client.get_all_positions()}
