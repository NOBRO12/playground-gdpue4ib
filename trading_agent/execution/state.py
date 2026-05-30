"""Persisted state for the live `paper` loop.

The backtester is stateless (it computes everything from bars). The live
loop, however, needs durable peak/day-start/trades-today/kill-switch
across ticks for the guardrails to fire. This module owns that file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LiveState:
    peak_equity_usd: float = 0.0
    day_start_equity_usd: float = 0.0
    day_start_date: str = ""
    trades_today: int = 0
    kill_switch_tripped: bool = False
    last_tick_utc: str = ""  # heartbeat: ISO timestamp of the most recent tick
    # Open-position exit levels (0.0 = flat / not set). Persisted so resting
    # stop/take orders can be reconciled and the live loop's exit decision
    # matches the backtester.
    open_entry_px: float = 0.0
    open_stop_px: float = 0.0
    open_take_px: float = 0.0

    def clear_open(self) -> None:
        self.open_entry_px = 0.0
        self.open_stop_px = 0.0
        self.open_take_px = 0.0

    @classmethod
    def load(cls, path: str | Path) -> "LiveState":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        tmp.replace(path)

    def observe(self, current_equity: float, today_utc: str) -> None:
        """Update peak and roll the UTC day if needed."""
        if today_utc != self.day_start_date:
            self.day_start_date = today_utc
            self.day_start_equity_usd = current_equity
            self.trades_today = 0
        if current_equity > self.peak_equity_usd:
            self.peak_equity_usd = current_equity
