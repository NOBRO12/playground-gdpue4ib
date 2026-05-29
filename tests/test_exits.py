from __future__ import annotations

import math

from trading_agent.execution.exits import exit_for_levels, take_from_stop


def test_stop_triggers_when_low_breaches():
    reason, px = exit_for_levels(low=95.0, high=101.0, stop_px=96.0, take_px=110.0)
    assert reason == "stop" and px == 96.0


def test_take_triggers_when_high_breaches():
    reason, px = exit_for_levels(low=99.0, high=111.0, stop_px=96.0, take_px=110.0)
    assert reason == "take_profit" and px == 110.0


def test_stop_checked_before_take_when_bar_spans_both():
    # Intrabar-conservative: a bar that hits both exits at the stop.
    reason, px = exit_for_levels(low=95.0, high=111.0, stop_px=96.0, take_px=110.0)
    assert reason == "stop" and px == 96.0


def test_no_exit_when_neither_breached():
    reason, px = exit_for_levels(low=99.0, high=101.0, stop_px=96.0, take_px=110.0)
    assert reason is None and px is None


def test_nan_levels_never_trigger():
    reason, px = exit_for_levels(low=1.0, high=1e9, stop_px=math.nan, take_px=math.nan)
    assert reason is None and px is None


def test_take_from_stop_is_r_multiple_of_risk():
    # entry 100, stop 96 → risk 4/unit; 3R take = 100 + 12 = 112.
    assert take_from_stop(entry_px=100.0, stop_px=96.0, take_profit_r=3.0) == 112.0
