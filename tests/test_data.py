from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.execution import data


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-05-01", periods=24, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 30_000 + rng.normal(0, 100, size=24).cumsum()
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.uniform(50, 200, size=24),
        },
        index=idx,
    )


def test_csv_path_matches_load_window_glob(tmp_path):
    path = data.csv_path(tmp_path, "BTC/USD", "1h", "2024-05-01", "2024-05-02")
    bars = _bars()
    data.write_csv(bars, path)
    # load_window must find the file by symbol+timeframe glob.
    loaded = data.load_window(tmp_path, "BTC/USD", "1h")
    assert len(loaded) == len(bars)


def test_write_csv_refuses_overwrite(tmp_path):
    path = tmp_path / "BTC-USD_1h_a_b.csv"
    data.write_csv(_bars(), path)
    with pytest.raises(FileExistsError):
        data.write_csv(_bars(), path)


def test_write_csv_force_overwrites(tmp_path):
    path = tmp_path / "BTC-USD_1h_a_b.csv"
    data.write_csv(_bars(), path)
    data.write_csv(_bars(), path, overwrite=True)  # should not raise


def test_csv_round_trip_preserves_ohlcv(tmp_path):
    path = data.csv_path(tmp_path, "ETH/USD", "4h", "2024-05-01", "2024-05-02")
    bars = _bars()
    data.write_csv(bars, path)
    loaded = data.load_csv(path)
    for col in ["open", "high", "low", "close", "volume"]:
        np.testing.assert_allclose(loaded[col].values, bars[col].values, rtol=1e-9)
