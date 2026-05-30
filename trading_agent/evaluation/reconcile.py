"""Backtest-vs-live execution reconciliation.

Pure functions over rows from the ``executions`` table. The headline question is
whether realized live trading cost (slippage + commission) came in at or under the
cost the gate's stress test assumed (``PROMOTE_COST_STRESS_MULT`` x the modeled
cost). If realized cost exceeds the stress assumption, the strategy's backtested
edge is an illusion — it fails live verification.

Keyed by symbol so a multi-instrument universe (Phase B) reports per-symbol with
no rewrite.
"""
from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Sequence


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = int(math.ceil(q / 100.0 * len(s))) - 1
    return s[max(0, min(len(s) - 1, k))]


def _dist(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": float(_percentile(values, 90)),
        "max": float(max(values)),
    }


def _commission_bps(row: Any) -> float:
    notional = float(row["fill_px"]) * float(row["filled_qty"])
    if notional <= 0:
        return 0.0
    return float(row["commission_usd"]) / notional * 1e4


def _aggregate(rows: list[Any], stress_cost_bps: float) -> dict[str, Any]:
    if not rows:
        return {
            "n_executions": 0,
            "fill_rate": None,
            "slippage_bps": _dist([]),
            "commission_bps": _dist([]),
            "realized_cost_bps": _dist([]),
            "latency_ms": _dist([]),
            "live_verification": "NO_DATA",
        }
    slips = [float(r["slippage_bps"]) for r in rows]
    comm = [_commission_bps(r) for r in rows]
    realized = [s + c for s, c in zip(slips, comm)]
    lat = [float(r["latency_ms"]) for r in rows if r["latency_ms"] is not None]
    tot_intended = sum(float(r["intended_qty"]) for r in rows)
    tot_filled = sum(float(r["filled_qty"]) for r in rows)
    realized_dist = _dist(realized)
    # Acceptance: MEAN realized per-side cost must be at/under the stress assumption.
    verdict = "PASS" if realized_dist["mean"] <= stress_cost_bps else "FAIL"
    return {
        "n_executions": len(rows),
        "fill_rate": (tot_filled / tot_intended) if tot_intended > 0 else None,
        "slippage_bps": _dist(slips),
        "commission_bps": _dist(comm),
        "realized_cost_bps": realized_dist,
        "latency_ms": _dist(lat),
        "live_verification": verdict,
    }


def build_report(
    rows: Sequence[Any], *, modeled_cost_bps: float, stress_cost_bps: float
) -> dict[str, Any]:
    """Aggregate execution rows into an overall + per-symbol reconciliation report."""
    rows = list(rows)
    symbols = sorted({r["symbol"] for r in rows})
    return {
        "modeled_cost_bps": modeled_cost_bps,
        "stress_cost_bps": stress_cost_bps,
        "overall": _aggregate(rows, stress_cost_bps),
        "by_symbol": {
            sym: _aggregate([r for r in rows if r["symbol"] == sym], stress_cost_bps)
            for sym in symbols
        },
    }
