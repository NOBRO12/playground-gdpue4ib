"""Thin wrapper over Alpaca's crypto trading API.

Paper mode by default. Set ``AGENT_LIVE_MODE=true`` to route orders to the
real-money live endpoint. All guardrails in ``trading_agent.guardrails.risk``
apply identically in both modes.
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


class AlpacaCryptoBroker:
    """Alpaca crypto broker. Paper by default; `AGENT_LIVE_MODE=true` selects live."""

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

    def mark(self, symbol: str, px: float) -> None:
        """No-op: Alpaca prices its own positions. Present for interface parity with MockBroker."""
