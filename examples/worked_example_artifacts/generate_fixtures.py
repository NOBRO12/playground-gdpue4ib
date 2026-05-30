"""Generate deterministic SPY daily-bar fixtures for the worked example.

Produces two CSVs (daily bars, matching the SPY/1d champion):
    - SPY_1d_IS.csv : ~3 years of daily bars (proposer & in-sample stats)
    - SPY_1d_OOS.csv: ~2 disjoint years of daily bars (promoter only)

Re-running with the same seeds reproduces the exact bars in this repo. The bars
are synthetic — no Alpaca keys or network needed — so CI can reproduce the
champion-vs-challenger rejection path offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def synth(start: str, days: int, seed: int, drift: float, vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="1D", tz="UTC")
    returns = rng.normal(drift, vol, size=days)
    close = 450 * np.exp(np.cumsum(returns))  # SPY-scale starting price
    high = close * (1 + rng.uniform(0, 0.006, size=days))
    low = close * (1 - rng.uniform(0, 0.006, size=days))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(5e7, 1.2e8, size=days)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    is_dir = repo / "data" / "is"
    oos_dir = repo / "data" / "oos"
    is_dir.mkdir(parents=True, exist_ok=True)
    oos_dir.mkdir(parents=True, exist_ok=True)

    # ~3 years IS, mildly trending with realistic daily vol.
    is_bars = synth("2021-01-01", days=750, seed=42, drift=0.0005, vol=0.011)
    is_bars.to_csv(is_dir / "SPY_1d_IS.csv", index=False)

    # ~2 years OOS, similar params but different seed/regime.
    oos_bars = synth("2024-01-01", days=500, seed=99, drift=0.0003, vol=0.013)
    oos_bars.to_csv(oos_dir / "SPY_1d_OOS.csv", index=False)

    print(f"wrote {is_dir/'SPY_1d_IS.csv'} ({len(is_bars)} bars)")
    print(f"wrote {oos_dir/'SPY_1d_OOS.csv'} ({len(oos_bars)} bars)")


if __name__ == "__main__":
    main()
