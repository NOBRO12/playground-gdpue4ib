from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from trading_agent import config
from trading_agent.cli import _status
from trading_agent.evaluation import db, logger as evlogger
from trading_agent.execution.state import LiveState


def test_status_reports_performance_and_heartbeat(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_LIVE_STATE_PATH", str(tmp_path / "ls.json"))
    s = config.load()
    db.init(s.db_path)
    with db.session(s.db_path) as conn:
        evlogger.log_equity(conn, "2024-01-01T00:00:00+00:00", 100_000, 100_000, 0, 100_000)
        evlogger.log_equity(conn, "2024-06-01T00:00:00+00:00", 110_000, 50_000, 60_000, 110_000)
        evlogger.log_trade(
            conn, version="v0", symbol="SPY", qty=1,
            entry_ts="2024-01-02T00:00:00+00:00", entry_px=100.0,
        )
        evlogger.close_open_trade(
            conn, symbol="SPY", version="v0",
            exit_ts="2024-01-03T00:00:00+00:00", exit_px=110.0, reason_exit="signal_exit",
        )
    LiveState(last_tick_utc=datetime.now(timezone.utc).isoformat()).save(s.live_state_path)

    _status(argparse.Namespace())
    snap = json.loads(capsys.readouterr().out)

    assert snap["performance"]["since_inception_return"] == 0.1  # 110k / 100k - 1
    assert snap["performance"]["closed_trades"] == 1
    assert snap["performance"]["win_rate_to_date"] == 1.0
    assert "minutes_since_last_tick" in snap["live_state"]
