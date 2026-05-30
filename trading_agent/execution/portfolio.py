"""In-memory portfolio state used by the live loop and backtester."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Position:
    symbol: str
    qty: float
    entry_px: float
    stop_px: float | None
    entry_ts: str


@dataclass
class Portfolio:
    cash_usd: float
    positions: dict[str, Position] = field(default_factory=dict)
    peak_equity_usd: float = 0.0
    day_start_equity_usd: float = 0.0
    day_start_date: str = ""
    trades_today: int = 0
    kill_switch: bool = False

    def equity(self, mark_prices: dict[str, float]) -> float:
        exposure = sum(p.qty * mark_prices.get(p.symbol, p.entry_px) for p in self.positions.values())
        return self.cash_usd + exposure

    def roll_day_if_needed(self, mark_prices: dict[str, float]) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.day_start_date:
            self.day_start_date = today
            self.day_start_equity_usd = self.equity(mark_prices)
            self.trades_today = 0

    def update_peak(self, mark_prices: dict[str, float]) -> None:
        eq = self.equity(mark_prices)
        if eq > self.peak_equity_usd:
            self.peak_equity_usd = eq
