"""Bar data fetcher.

Reads cached parquet/CSV from disk for backtests; pulls fresh bars from Alpaca
for the live loop. The proposer only ever sees IS bars; the OOS directory is
read by the promoter alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    return df[cols].sort_index()


def load_csv(path: str | Path) -> pd.DataFrame:
    return _normalize(pd.read_csv(path))


def load_window(directory: str | Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """Concatenate every CSV named ``{symbol}_{timeframe}*.csv`` in dir."""
    directory = Path(directory)
    pattern = f"{symbol.replace('/', '-')}_{timeframe}*.csv"
    parts = [pd.read_csv(p) for p in sorted(directory.glob(pattern))]
    if not parts:
        raise FileNotFoundError(f"no bars found in {directory} for {symbol} {timeframe}")
    return _normalize(pd.concat(parts, ignore_index=True))


def fetch_live(symbol: str, timeframe: str, lookback_hours: int = 200) -> pd.DataFrame:
    """Pull recent crypto bars from Alpaca. Keys must be in env."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf_map = {
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
    }
    client = CryptoHistoricalDataClient()  # crypto market data is unauthenticated
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(hours=lookback_hours)
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol], timeframe=tf_map[timeframe], start=start, end=end
    )
    bars = client.get_crypto_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    return _normalize(bars.reset_index())
