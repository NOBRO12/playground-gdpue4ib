from __future__ import annotations

import argparse
import json

from trading_agent import config
from trading_agent.cli import _reconcile
from trading_agent.evaluation import db, logger as evlogger
from trading_agent.evaluation.reconcile import build_report


def _row(symbol, slippage_bps, fill_px=600.0, filled_qty=10, intended_qty=10,
         commission_usd=0.0, latency_ms=20.0, side="buy"):
    return {
        "symbol": symbol, "side": side, "slippage_bps": slippage_bps,
        "commission_usd": commission_usd, "fill_px": fill_px,
        "filled_qty": filled_qty, "intended_qty": intended_qty, "latency_ms": latency_ms,
    }


def test_report_passes_when_cost_under_stress():
    rows = [_row("SPY", 8.0), _row("SPY", 12.0), _row("SPY", 10.0)]  # mean 10 < 30
    rep = build_report(rows, modeled_cost_bps=15.0, stress_cost_bps=30.0)
    assert rep["overall"]["live_verification"] == "PASS"
    assert abs(rep["overall"]["realized_cost_bps"]["mean"] - 10.0) < 1e-9
    assert rep["overall"]["n_executions"] == 3
    assert rep["by_symbol"]["SPY"]["live_verification"] == "PASS"


def test_report_fails_when_cost_exceeds_stress():
    rows = [_row("SPY", 40.0), _row("SPY", 35.0)]  # mean 37.5 > 30
    rep = build_report(rows, modeled_cost_bps=15.0, stress_cost_bps=30.0)
    assert rep["overall"]["live_verification"] == "FAIL"


def test_report_includes_commission_in_realized_cost():
    # $0.60 commission on $6000 notional = 1 bp; + 10 bps slippage = 11 bps realized.
    rows = [_row("SPY", 10.0, fill_px=600.0, filled_qty=10, commission_usd=0.60)]
    rep = build_report(rows, modeled_cost_bps=15.0, stress_cost_bps=30.0)
    assert abs(rep["overall"]["realized_cost_bps"]["mean"] - 11.0) < 1e-9


def test_report_fill_rate_reflects_partials():
    rows = [_row("SPY", 5.0, filled_qty=7, intended_qty=10)]
    rep = build_report(rows, modeled_cost_bps=15.0, stress_cost_bps=30.0)
    assert abs(rep["overall"]["fill_rate"] - 0.7) < 1e-9


def test_report_groups_by_symbol():
    rows = [_row("SPY", 5.0), _row("QQQ", 45.0)]
    rep = build_report(rows, modeled_cost_bps=15.0, stress_cost_bps=30.0)
    assert rep["by_symbol"]["SPY"]["live_verification"] == "PASS"
    assert rep["by_symbol"]["QQQ"]["live_verification"] == "FAIL"


def test_report_no_data():
    rep = build_report([], modeled_cost_bps=15.0, stress_cost_bps=30.0)
    assert rep["overall"]["live_verification"] == "NO_DATA"
    assert rep["overall"]["n_executions"] == 0


def test_reconcile_cli_prints_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    s = config.load()
    db.init(s.db_path)
    with db.session(s.db_path) as conn:
        evlogger.log_execution(
            conn, version="v0", symbol="SPY", side="buy", order_type="market",
            intended_px=600.0, fill_px=600.6, intended_qty=10, filled_qty=10,
            commission_usd=0.0, latency_ms=15.0, order_id="o1", status="filled",
        )
    _reconcile(argparse.Namespace())
    rep = json.loads(capsys.readouterr().out)
    assert rep["stress_cost_bps"] == 30.0          # 2x the 15bps modeled
    assert rep["overall"]["live_verification"] == "PASS"  # 10bps slippage < 30
