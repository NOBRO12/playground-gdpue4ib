from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from trading_agent import diagnostics


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Inject a fake `anthropic` module so check_anthropic is offline."""
    mod = types.ModuleType("anthropic")

    class _Client:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.messages = MagicMock()
            usage = types.SimpleNamespace(output_tokens=5)
            self.messages.create.return_value = types.SimpleNamespace(usage=usage)

    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


@pytest.fixture
def fake_anthropic_auth_error(monkeypatch):
    mod = types.ModuleType("anthropic")

    class _AuthError(Exception):
        pass

    class _Client:
        def __init__(self, api_key: str):
            self.messages = MagicMock()
            self.messages.create.side_effect = _AuthError("invalid x-api-key")

    mod.Anthropic = _Client
    mod.AuthenticationError = _AuthError
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return mod


def test_anthropic_skipped_when_no_key():
    r = diagnostics.check_anthropic(None)
    assert r["status"] == "SKIPPED"
    assert "ANTHROPIC_API_KEY" in r["detail"]


def test_anthropic_ok_with_valid_key(fake_anthropic):
    r = diagnostics.check_anthropic("sk-ant-api03-real")
    assert r["status"] == "OK"
    assert "reachable" in r["detail"]


def test_anthropic_fail_on_auth_error(fake_anthropic_auth_error):
    r = diagnostics.check_anthropic("sk-ant-api03-bogus")
    assert r["status"] == "FAIL"
    assert "invalid x-api-key" in r["detail"]


def test_alpaca_trading_skipped_when_no_key():
    assert diagnostics.check_alpaca_trading(None, None)["status"] == "SKIPPED"
    assert diagnostics.check_alpaca_trading("k", None)["status"] == "SKIPPED"
    assert diagnostics.check_alpaca_trading(None, "s")["status"] == "SKIPPED"


def test_alpaca_trading_ok(monkeypatch):
    pkg = types.ModuleType("alpaca")
    trading_pkg = types.ModuleType("alpaca.trading")
    client_mod = types.ModuleType("alpaca.trading.client")

    class _TradingClient:
        def __init__(self, key, secret, paper=True):
            assert paper is True

        def get_account(self):
            return types.SimpleNamespace(cash="100000.0", equity="100050.0")

    client_mod.TradingClient = _TradingClient
    monkeypatch.setitem(sys.modules, "alpaca", pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client_mod)

    r = diagnostics.check_alpaca_trading("k", "s")
    assert r["status"] == "OK"
    assert "$100,000.00" in r["detail"]
    assert "$100,050.00" in r["detail"]


def test_alpaca_trading_fail_surfaces_error(monkeypatch):
    pkg = types.ModuleType("alpaca")
    trading_pkg = types.ModuleType("alpaca.trading")
    client_mod = types.ModuleType("alpaca.trading.client")

    class _TradingClient:
        def __init__(self, *a, **kw):
            raise RuntimeError("bad key")

    client_mod.TradingClient = _TradingClient
    monkeypatch.setitem(sys.modules, "alpaca", pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading_pkg)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client_mod)

    r = diagnostics.check_alpaca_trading("k", "s")
    assert r["status"] == "FAIL"
    assert "RuntimeError" in r["detail"]
    assert "bad key" in r["detail"]


def test_alpaca_data_fail_surfaces_error(monkeypatch):
    from trading_agent.execution import data as data_mod

    def _boom(*a, **kw):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(data_mod, "fetch_history", _boom)
    r = diagnostics.check_alpaca_data()
    assert r["status"] == "FAIL"
    assert "ConnectionError" in r["detail"]


def test_alpaca_data_ok(monkeypatch):
    import pandas as pd

    from trading_agent.execution import data as data_mod

    def _ok(symbol, timeframe, start, end):
        idx = pd.date_range(start, end, freq="1h", tz="UTC")[:24]
        return pd.DataFrame(
            {
                "open": [30000.0] * len(idx),
                "high": [30100.0] * len(idx),
                "low": [29900.0] * len(idx),
                "close": [30050.0] * len(idx),
                "volume": [100.0] * len(idx),
            },
            index=idx,
        )

    monkeypatch.setattr(data_mod, "fetch_history", _ok)
    r = diagnostics.check_alpaca_data()
    assert r["status"] == "OK"
    assert "BTC/USD" in r["detail"]


def test_overall_exit_code():
    assert diagnostics.overall_exit_code(
        [{"status": "OK"}, {"status": "SKIPPED"}, {"status": "OK"}]
    ) == 0
    assert diagnostics.overall_exit_code(
        [{"status": "OK"}, {"status": "FAIL"}]
    ) == 1
    assert diagnostics.overall_exit_code(
        [{"status": "SKIPPED"}, {"status": "SKIPPED"}]
    ) == 0  # nothing failed
