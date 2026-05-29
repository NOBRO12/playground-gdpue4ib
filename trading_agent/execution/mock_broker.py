"""In-memory broker for offline development.

Mirrors the public surface of :class:`AlpacaCryptoBroker` (``equity``, ``cash``,
``positions``, ``submit_market``) plus a ``mark`` method used to set the
reference price before each tick. State persists to JSON so equity drift,
positions, and the kill-switch path can be observed across runs.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .broker import Fill


@dataclass
class MockBroker:
    state_path: Path
    starting_cash: float = 100_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    _cash: float = field(init=False, default=0.0)
    _positions: dict[str, dict[str, float]] = field(init=False, default_factory=dict)
    _marks: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text())
            self._cash = float(payload["cash"])
            self._positions = {
                k: {"qty": float(v["qty"]), "avg_entry_px": float(v["avg_entry_px"])}
                for k, v in payload.get("positions", {}).items()
            }
        else:
            self._cash = float(self.starting_cash)
            self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"cash": self._cash, "positions": self._positions},
                indent=2,
                sort_keys=True,
            )
        )
        tmp.replace(self.state_path)

    def mark(self, symbol: str, px: float) -> None:
        """Set the reference price used for equity() and submit_market() fills."""
        self._marks[symbol] = float(px)

    def equity(self) -> float:
        exposure = 0.0
        for sym, pos in self._positions.items():
            if pos["qty"] <= 0:
                continue
            mark_px = self._marks.get(sym, pos["avg_entry_px"])
            exposure += mark_px * pos["qty"]
        return self._cash + exposure

    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, float]:
        return {s: p["qty"] for s, p in self._positions.items() if p["qty"] > 0}

    # No exchange to rest orders on; the live loop simulates stop/take at tick
    # cadence (matching the backtester) when this is False.
    supports_resting_orders = False

    def is_market_open(self) -> bool:
        """Always open offline — interface parity with the Alpaca brokers."""
        return True

    def daytrade_count(self) -> int:
        """No PDT tracking offline — interface parity with the Alpaca brokers."""
        return 0

    def cancel_open_orders(self, symbol: str) -> None:
        """No resting orders offline — no-op for interface parity."""

    def submit_market(self, symbol: str, side: str, qty: float) -> Fill:
        if symbol not in self._marks:
            raise RuntimeError(
                f"mock broker has no mark for {symbol}; call mark() before submit_market()"
            )
        px = self._marks[symbol]
        cost = (self.fee_bps + self.slippage_bps) / 1e4

        if side == "buy":
            fill_px = px * (1 + cost)
            notional = qty * fill_px
            if notional > self._cash:
                raise RuntimeError(
                    f"insufficient cash for buy: have {self._cash:.2f}, need {notional:.2f}"
                )
            existing = self._positions.get(symbol)
            if existing and existing["qty"] > 0:
                new_qty = existing["qty"] + qty
                self._positions[symbol] = {
                    "qty": new_qty,
                    "avg_entry_px": (
                        existing["qty"] * existing["avg_entry_px"] + qty * fill_px
                    )
                    / new_qty,
                }
            else:
                self._positions[symbol] = {"qty": qty, "avg_entry_px": fill_px}
            self._cash -= notional

        elif side == "sell":
            held = self._positions.get(symbol, {}).get("qty", 0.0)
            if qty > held + 1e-12:
                raise RuntimeError(
                    f"mock broker rejects naked short on {symbol}: held={held}, qty={qty}"
                )
            fill_px = px * (1 - cost)
            self._positions[symbol]["qty"] = held - qty
            self._cash += qty * fill_px
            if self._positions[symbol]["qty"] <= 1e-12:
                self._positions[symbol]["qty"] = 0.0

        else:
            raise ValueError(f"unknown side: {side!r}")

        self._save()
        return Fill(
            symbol=symbol, side=side, qty=qty, avg_price=fill_px, order_id=str(uuid.uuid4())
        )
