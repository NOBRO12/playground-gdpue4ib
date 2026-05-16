"""Pydantic schema for strategy specs. The LLM's only output surface."""
from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Symbol = Literal["BTC/USD", "ETH/USD"]
Timeframe = Literal["1h", "4h", "1d"]
StrategyType = Literal["donchian", "ema_cross"]


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


class Filters(BaseModel):
    adx_min: Optional[float] = Field(default=None, ge=0, le=100)
    vol_max: Optional[float] = Field(default=None, ge=0, le=5.0)


class RiskSpec(BaseModel):
    atr_mult_stop: float = Field(ge=0.25, le=10.0)
    take_profit_r: float = Field(ge=0.25, le=10.0)
    position_pct: float = Field(ge=0.01, le=0.25)


class StrategySpec(BaseModel):
    type: StrategyType
    symbol: Symbol
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
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)


# JSON schema used as the input_schema of the Claude tool. Anthropic's tool
# input_schema is JSON Schema, so pydantic's output is reusable.
def claude_tool_input_schema() -> dict[str, Any]:
    return StrategySpec.model_json_schema()
