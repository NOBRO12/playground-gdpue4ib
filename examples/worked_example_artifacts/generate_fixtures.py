"""Generate deterministic BTC/USD bar fixtures for the worked example.

Produces two CSVs:
    - is_btc_1h.csv : 90 days of 1h bars (proposer & in-sample stats)
    - oos_btc_1h.csv: 60 disjoint days of 1h bars (promoter only)

Re-running with the same seeds reproduces the exact bars in this repo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def synth(start: str, hours: int, seed: int, drift: float, vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    returns = rng.normal(drift, vol, size=hours)
    close = 30_000 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.004, size=hours))
    low = close * (1 - rng.uniform(0, 0.004, size=hours))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(50, 200, size=hours)
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

    # 90 days IS, mildly trending with one chop cluster.
    is_bars = synth("2024-01-01", hours=90 * 24, seed=42, drift=0.0006, vol=0.014)
    is_bars.to_csv(is_dir / "BTC-USD_1h_2024Q1.csv", index=False)

    # 60 days OOS, similar params but different seed.
    oos_bars = synth("2024-04-01", hours=60 * 24, seed=99, drift=0.0004, vol=0.016)
    oos_bars.to_csv(oos_dir / "BTC-USD_1h_2024Q2.csv", index=False)

    print(f"wrote {is_dir/'BTC-USD_1h_2024Q1.csv'} ({len(is_bars)} bars)")
    print(f"wrote {oos_dir/'BTC-USD_1h_2024Q2.csv'} ({len(oos_bars)} bars)")


if __name__ == "__main__":
    main()
