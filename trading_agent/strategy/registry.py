"""Load strategies by spec; pick the right implementation by type."""
from __future__ import annotations

import json
from pathlib import Path

from .base import Strategy
from .donchian import DonchianStrategy
from .ema_cross import EmaCrossStrategy
from .spec import StrategySpec

_REGISTRY: dict[str, type[Strategy]] = {
    "donchian": DonchianStrategy,
    "ema_cross": EmaCrossStrategy,
}


def from_spec(spec: StrategySpec) -> Strategy:
    cls = _REGISTRY[spec.type]
    return cls(spec)


def load_spec(path: str | Path) -> StrategySpec:
    raw = json.loads(Path(path).read_text())
    return StrategySpec.model_validate(raw)


def save_spec(spec: StrategySpec, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec.model_dump(), indent=2, sort_keys=True))
