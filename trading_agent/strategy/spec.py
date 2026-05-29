"""Pydantic schema for strategy specs. The LLM's only output surface."""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# A US-equity ticker (1–5 uppercase letters, e.g. SPY, AAPL) or a crypto pair
# (e.g. BTC/USD). Constrained by pattern rather than a free string so the LLM
# proposer can't emit arbitrary/illiquid symbols, but isn't locked to crypto.
SYMBOL_PATTERN = r"^([A-Z]{1,5}|[A-Z]{2,5}/USD)$"
Timeframe = Literal["1h", "4h", "1d"]
StrategyType = Literal["donchian", "ema_cross", "mean_reversion"]


class DonchianParams(BaseModel):
    entry_lookback: int = Field(ge=5, le=200)
    exit_lookback: int = Field(ge=3, le=100)


class EmaCrossParams(BaseModel):
    fast: int = Field(ge=3, le=100)
    slow: int = Field(ge=5, le=300)

    @model_validator(mode="after")
    def _fast_lt_slow(self) -> "EmaCrossParams":
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")
        return self


class MeanReversionParams(BaseModel):
    """Bollinger-style mean reversion: buy stretched-below-mean dips, exit on
    reversion back toward the mean. Orthogonal to the trend-following templates
    (donchian/ema_cross) so the proposer has range-bound edge to explore."""

    lookback: int = Field(ge=5, le=200)  # SMA / std-dev window
    entry_z: float = Field(ge=0.5, le=4.0)  # std-devs below mean to enter long
    exit_z: float = Field(ge=-2.0, le=2.0)  # z-level to exit (0 = revert to mean)

    @model_validator(mode="after")
    def _exit_above_entry(self) -> "MeanReversionParams":
        # Exit must sit above the entry trigger (-entry_z), else we'd exit lower
        # than we entered and never capture the reversion.
        if self.exit_z <= -self.entry_z:
            raise ValueError("exit_z must be greater than -entry_z")
        return self


class Filters(BaseModel):
    adx_min: Optional[float] = Field(default=None, ge=0, le=100)
    vol_max: Optional[float] = Field(default=None, ge=0, le=5.0)


class RiskSpec(BaseModel):
    atr_mult_stop: float = Field(ge=0.25, le=10.0)
    take_profit_r: float = Field(ge=0.25, le=10.0)
    position_pct: float = Field(ge=0.01, le=0.25)
    # Optional volatility-targeted sizing: risk this fraction of equity per trade
    # over the entry-to-stop distance. None = fixed position_pct sizing.
    # position_pct always remains the hard notional cap.
    risk_per_trade_pct: Optional[float] = Field(default=None, ge=0.001, le=0.05)


class StrategySpec(BaseModel):
    type: StrategyType
    symbol: str = Field(pattern=SYMBOL_PATTERN)
    timeframe: Timeframe
    params: dict[str, Any]
    filters: Filters = Field(default_factory=Filters)
    risk: RiskSpec
    version: str
    parent: Optional[str] = None

    @field_validator("params")
    @classmethod
    def _params_nonempty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("params cannot be empty")
        return v

    @model_validator(mode="after")
    def _typed_params(self) -> "StrategySpec":
        if self.type == "donchian":
            DonchianParams.model_validate(self.params)
        elif self.type == "ema_cross":
            EmaCrossParams.model_validate(self.params)
        elif self.type == "mean_reversion":
            MeanReversionParams.model_validate(self.params)
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)


# JSON schema used as the input_schema of the Claude tool. Anthropic's tool
# input_schema is JSON Schema, so pydantic's output is reusable.
def claude_tool_input_schema() -> dict[str, Any]:
    return StrategySpec.model_json_schema()
