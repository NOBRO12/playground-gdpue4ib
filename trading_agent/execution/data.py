"""Bar data fetcher.

Reads cached parquet/CSV from disk for backtests; pulls fresh bars from Alpaca
for the live loop and for populating IS/OOS windows. The proposer only ever
sees IS bars; the OOS directory is read by the promoter alone.
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


def csv_path(directory: str | Path, symbol: str, timeframe: str, start: str, end: str) -> Path:
    """Filename convention that ``load_window`` will find."""
    safe_symbol = symbol.replace("/", "-")
    return Path(directory) / f"{safe_symbol}_{timeframe}_{start}_{end}.csv"


def write_csv(bars: pd.DataFrame, path: str | Path, overwrite: bool = False) -> Path:
    """Persist bars to CSV with the canonical 'timestamp' index column.

    Refuses to overwrite by default — the OOS window in particular must not be
    silently clobbered.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass overwrite=True to force")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars.copy()
    out.index.name = "timestamp"
    out.reset_index().to_csv(path, index=False)
    return path


def _tf_map():
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    return {
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
    }


def _fetch_crypto(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest

    client = CryptoHistoricalDataClient()  # crypto market data is unauthenticated
    req = CryptoBarsRequest(
        symbol_or_symbols=[symbol], timeframe=_tf_map()[timeframe], start=start, end=end
    )
    return client.get_crypto_bars(req).df


def _fetch_stock(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    from .. import config

    s = config.load()
    if not s.alpaca_key or not s.alpaca_secret:
        raise RuntimeError(
            "ALPACA_KEY/ALPACA_SECRET required for stock market data "
            "(unlike crypto, the equities feed is authenticated)"
        )
    client = StockHistoricalDataClient(s.alpaca_key, s.alpaca_secret)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=_tf_map()[timeframe],
        start=start,
        end=end,
        feed=s.stock_feed,  # "iex" (free) or "sip" (paid)
    )
    return client.get_stock_bars(req).df


def fetch_history(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Pull historical bars from Alpaca between start and end (UTC).

    Routes to the stock or crypto data API based on ``AGENT_ASSET_CLASS``.
    Both return the same MultiIndex (symbol, timestamp) OHLCV frame, normalized
    to a flat UTC-indexed OHLCV frame here.
    """
    from .. import config

    if config.load().asset_class == "crypto":
        bars = _fetch_crypto(symbol, timeframe, start, end)
    else:
        bars = _fetch_stock(symbol, timeframe, start, end)
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    return _normalize(bars.reset_index())


def fetch_live(symbol: str, timeframe: str, lookback_hours: int = 200) -> pd.DataFrame:
    """Recent bars ending now — thin wrapper around fetch_history."""
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(hours=lookback_hours)
    return fetch_history(symbol, timeframe, start, end)


def bar_age_seconds(bars: pd.DataFrame, now: datetime | None = None) -> float:
    """Seconds between the most recent bar's timestamp and ``now`` (UTC).

    Used by the live loop's staleness circuit breaker: a market-data outage
    leaves the last bar far in the past, and trading on it is dangerous.
    """
    now = now or datetime.now(timezone.utc)
    last = pd.Timestamp(bars.index[-1]).to_pydatetime()
    return (now - last).total_seconds()
