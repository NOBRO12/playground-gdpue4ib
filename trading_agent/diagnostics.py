"""Connectivity + credential checks for the integrations the agent depends on.

Each ``check_*`` returns a small dict ``{name, status, detail}`` so the CLI
formatter can render a table and tests can assert on the result without
touching the network.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _result(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def check_anthropic(api_key: str | None, model: str = "claude-haiku-4-5-20251001") -> dict[str, str]:
    if not api_key:
        return _result("anthropic", "SKIPPED", "ANTHROPIC_API_KEY not set")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        usage = getattr(msg, "usage", None)
        out_tok = getattr(usage, "output_tokens", "?") if usage else "?"
        return _result("anthropic", "OK", f"model {model} reachable (output_tokens={out_tok})")
    except Exception as exc:  # noqa: BLE001 — surface any class to the user
        return _result("anthropic", "FAIL", f"{type(exc).__name__}: {exc}")


def check_alpaca_trading(
    key: str | None, secret: str | None, paper: bool = True
) -> dict[str, str]:
    if not key or not secret:
        return _result("alpaca-trading", "SKIPPED", "ALPACA_KEY/ALPACA_SECRET not set")
    try:
        from alpaca.trading.client import TradingClient

        client = TradingClient(key, secret, paper=paper)
        acct = client.get_account()
        mode = "paper" if paper else "LIVE"
        return _result(
            "alpaca-trading",
            "OK",
            f"{mode} account cash=${float(acct.cash):,.2f} equity=${float(acct.equity):,.2f}",
        )
    except Exception as exc:  # noqa: BLE001
        return _result("alpaca-trading", "FAIL", f"{type(exc).__name__}: {exc}")


def check_alpaca_data(symbol: str = "BTC/USD", timeframe: str = "1h") -> dict[str, str]:
    """Crypto market-data reachability (unauthenticated endpoint)."""
    try:
        from .execution.data import fetch_history

        end = datetime.now(timezone.utc) - timedelta(hours=2)
        start = end - timedelta(hours=24)
        bars = fetch_history(symbol, timeframe, start, end)
        return _result("alpaca-data", "OK", f"got {len(bars)} {symbol} {timeframe} bars")
    except Exception as exc:  # noqa: BLE001
        return _result("alpaca-data", "FAIL", f"{type(exc).__name__}: {exc}")


def check_alpaca_stock_data(
    key: str | None, secret: str | None, feed: str = "iex", symbol: str = "SPY"
) -> dict[str, str]:
    """Authenticated stock market-data reachability (validates keys + feed access)."""
    if not key or not secret:
        return _result("alpaca-stock-data", "SKIPPED", "ALPACA_KEY/ALPACA_SECRET not set")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        end = datetime.now(timezone.utc) - timedelta(days=1)
        start = end - timedelta(days=7)
        client = StockHistoricalDataClient(key, secret)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
            end=end,
            feed=feed,
        )
        bars = client.get_stock_bars(req).df
        return _result(
            "alpaca-stock-data", "OK", f"got {len(bars)} {symbol} 1d bars (feed={feed})"
        )
    except Exception as exc:  # noqa: BLE001
        return _result("alpaca-stock-data", "FAIL", f"{type(exc).__name__}: {exc}")


def run_all(
    anthropic_key: str | None,
    alpaca_key: str | None,
    alpaca_secret: str | None,
    anthropic_model: str | None = None,
    alpaca_paper: bool = True,
    asset_class: str = "crypto",
    stock_feed: str = "iex",
) -> list[dict[str, str]]:
    anthropic_kwargs = {"model": anthropic_model} if anthropic_model else {}
    data_check = (
        check_alpaca_stock_data(alpaca_key, alpaca_secret, feed=stock_feed)
        if asset_class == "stock"
        else check_alpaca_data()
    )
    return [
        check_anthropic(anthropic_key, **anthropic_kwargs),
        check_alpaca_trading(alpaca_key, alpaca_secret, paper=alpaca_paper),
        data_check,
    ]


def overall_exit_code(results: list[dict[str, Any]]) -> int:
    """0 if no FAIL; 1 if any check failed. SKIPPED does not fail."""
    return 1 if any(r["status"] == "FAIL" for r in results) else 0
