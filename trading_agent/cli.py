"""Single CLI entry point.

Subcommands:
    init-db            Create tables and seed champion version row.
    check-keys         Verify Anthropic + Alpaca credentials and network reachability.
    fetch-bars         Pull historical crypto bars from Alpaca and write CSV.
    backtest           Run a deterministic backtest of a spec against bars on disk.
    improve            Run the full proposer -> backtest -> promoter loop.
    paper              Live paper loop (requires Alpaca + Anthropic keys).
    status             Print champion + recent trades + recent decisions + live state.
    reset-kill-switch  Clear the kill_switch_tripped flag in live state.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .evaluation import db, logger as evlogger
from .evaluation import regime as regime_mod
from .execution import data, exits, sizing
from .execution.broker import make_client_order_id
from .execution.state import LiveState
from .improvement import backtester, promoter, versioning
from .notify import notify
from .strategy.registry import load_spec
from .strategy.spec import StrategySpec

log = logging.getLogger(__name__)


def _init_db(args: argparse.Namespace) -> int:
    s = config.load()
    db.init(s.db_path)
    spec = load_spec(s.strategies_dir / "champion.json")
    with db.session(s.db_path) as conn:
        versioning.record_version(conn, spec, status="champion", notes="seed")
    print(f"initialized {s.db_path}; champion={spec.version}")
    return 0


def _check_keys(args: argparse.Namespace) -> int:
    from . import diagnostics

    s = config.load()
    model = s.proposer_model if args.model == "proposer" else args.model
    results = diagnostics.run_all(
        s.anthropic_key,
        s.alpaca_key,
        s.alpaca_secret,
        anthropic_model=model,
        alpaca_paper=not s.live_mode,
        asset_class=s.asset_class,
        stock_feed=s.stock_feed,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        width = max(len(r["name"]) for r in results)
        for r in results:
            print(f"{r['name']:<{width}}  {r['status']:<7}  {r['detail']}")

    return diagnostics.overall_exit_code(results)


def _fetch_bars(args: argparse.Namespace) -> int:
    s = config.load()
    target_dir = s.is_dir if args.to == "is" else s.oos_dir
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    path = data.csv_path(target_dir, args.symbol, args.timeframe, args.start, args.end)
    if path.exists() and not args.force:
        print(f"refusing to overwrite {path}; pass --force to clobber", file=sys.stderr)
        return 2
    bars = data.fetch_history(args.symbol, args.timeframe, start, end)
    data.write_csv(bars, path, overwrite=args.force)
    print(f"wrote {len(bars)} bars to {path}")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    s = config.load()
    spec = load_spec(args.spec)
    bars = data.load_csv(args.bars)
    if args.window:
        start, end = args.window.split(":")
        bars = bars.loc[start:end]
    result = backtester.run(spec, bars)
    print(json.dumps(result.score.__dict__, indent=2))
    with db.session(s.db_path) as conn:
        conn.execute(
            """
            INSERT INTO backtests
            (version, window_start, window_end, oos_flag, sharpe, sortino, max_dd, n_trades, equity_final, params_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.version,
                bars.index[0].isoformat(),
                bars.index[-1].isoformat(),
                int(args.oos),
                result.score.sharpe,
                result.score.sortino,
                result.score.max_dd,
                result.score.n_trades,
                result.score.equity_final,
                spec.to_json(),
            ),
        )
    return 0


def _improve(args: argparse.Namespace) -> int:
    s = config.load()
    champion = load_spec(s.strategies_dir / "champion.json")

    n_trials = 1  # number of candidates considered; feeds the best-of-N penalty
    if args.fixture_proposal:
        challenger_raw = json.loads(Path(args.fixture_proposal).read_text())
        challenger_raw.setdefault("parent", champion.version)
        challenger_raw.setdefault(
            "version", "v-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        challenger = StrategySpec.model_validate(challenger_raw)
        prompt_hash = "fixture"
        response_json = json.dumps(challenger_raw, sort_keys=True)
    else:
        from .improvement import proposer, summary as summary_mod

        is_bars = data.load_window(s.is_dir, champion.symbol, champion.timeframe)
        champ_run = backtester.run(champion, is_bars)
        summary = summary_mod.build_trade_summary(
            champ_run.score, champ_run.trades, champ_run.equity
        )
        # Recent guardrail blocks (live) give the proposer real failure context.
        if s.db_path.exists():
            with db.session(s.db_path) as conn:
                rows = conn.execute(
                    "SELECT kind, COUNT(*) AS c FROM guardrail_events "
                    "GROUP BY kind ORDER BY c DESC LIMIT 10"
                ).fetchall()
            summary["recent_guardrail_blocks"] = {r["kind"]: r["c"] for r in rows}
        n = max(1, getattr(args, "n_challengers", 1))
        n_trials = n
        if n == 1:
            result = proposer.propose(champion, summary)
        else:
            # Generate N, score on OOS, keep the best gate-passer (else best by Sharpe).
            results = proposer.propose_many(champion, summary, n)
            rank_oos = data.load_window(
                Path(args.oos_dir) if args.oos_dir else s.oos_dir,
                champion.symbol,
                champion.timeframe,
            )
            ranked = promoter.rank_challengers(champion, [r.spec for r in results], rank_oos)
            best_spec, best_decision = ranked[0]
            result = next(r for r in results if r.spec.version == best_spec.version)
            log.info(
                "best-of-%d: chose %s (accepted=%s, sharpe=%.3f)",
                n, best_spec.version, best_decision.accepted,
                best_decision.challenger_score.sharpe,
            )
        challenger = result.spec
        prompt_hash = result.prompt_hash
        response_json = result.response_json

    oos_bars = data.load_window(
        Path(args.oos_dir) if args.oos_dir else s.oos_dir,
        challenger.symbol,
        challenger.timeframe,
    )
    # IS bars feed the walk-forward consistency check. A data shortage shouldn't
    # block promotion, so fall back to None (consistency is then skipped).
    try:
        consistency_bars = data.load_window(s.is_dir, challenger.symbol, challenger.timeframe)
    except Exception:
        consistency_bars = None
    decision = promoter.decide(
        champion, challenger, oos_bars,
        consistency_bars=consistency_bars, n_trials=n_trials,
    )

    print(
        json.dumps(
            {
                "accepted": decision.accepted,
                "reason": decision.reason,
                "champion": decision.champion_score.__dict__,
                "challenger": decision.challenger_score.__dict__,
                "challenger_at_2x_cost": (
                    decision.challenger_stress_score.__dict__
                    if decision.challenger_stress_score is not None
                    else None
                ),
            },
            indent=2,
        )
    )

    if args.dry_run:
        expected_accept = args.expect == "accept"
        if decision.accepted == expected_accept:
            return 0
        log.warning("expected %s but got %s", args.expect, "accept" if decision.accepted else "reject")
        return 2

    with db.session(s.db_path) as conn:
        conn.execute(
            """
            INSERT INTO proposals (ts, parent_version, prompt_hash, response_json, decision, reject_reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                champion.version,
                prompt_hash,
                response_json,
                "accepted" if decision.accepted else "rejected",
                None if decision.accepted else decision.reason,
            ),
        )
        versioning.record_version(conn, challenger, status="challenger")
        if decision.accepted:
            versioning.promote(conn, challenger, s.strategies_dir)
        else:
            versioning.reject(conn, challenger, s.strategies_dir, decision.reason)
    if decision.accepted:
        notify(
            "promotion",
            {"version": challenger.version, "parent": champion.version},
            s,
        )
    return 0


def _realized_edge(
    conn, symbol: str, version: str
) -> tuple[int, float | None, float | None]:
    """Realized (n_closed, win_rate, payoff_ratio) for a strategy version from
    its closed live trades' P&L. payoff_ratio = avg win / avg loss. Returns
    (n, None, None) when there aren't both wins and losses to estimate from."""
    rows = conn.execute(
        "SELECT pnl_usd FROM trades WHERE symbol = ? AND version = ? "
        "AND pnl_usd IS NOT NULL",
        (symbol, version),
    ).fetchall()
    pnls = [float(r["pnl_usd"]) for r in rows]
    return sizing.edge_from_pnls(pnls)


def _latency_ms(submitted_at: str, filled_at: str) -> float | None:
    if not submitted_at or not filled_at:
        return None
    try:
        a = datetime.fromisoformat(submitted_at)
        b = datetime.fromisoformat(filled_at)
    except ValueError:
        return None
    return (b - a).total_seconds() * 1000.0


def _record_execution(
    conn, *, version, symbol, side, order_type, intended_px, fill, client_order_id=None
) -> None:
    """Persist one order's realized execution quality for reconciliation."""
    evlogger.log_execution(
        conn,
        version=version,
        symbol=symbol,
        side=side,
        order_type=order_type,
        intended_px=float(intended_px),
        fill_px=float(fill.avg_price),
        intended_qty=float(fill.submitted_qty or fill.qty),
        filled_qty=float(fill.filled_qty or fill.qty),
        commission_usd=float(fill.commission),
        latency_ms=_latency_ms(fill.submitted_at, fill.filled_at),
        client_order_id=client_order_id,
        order_id=fill.order_id,
        status=fill.status,
    )


def _status(args: argparse.Namespace) -> int:
    s = config.load()
    champion = load_spec(s.strategies_dir / "champion.json")
    state = LiveState.load(s.live_state_path)
    snapshot = {
        "champion": {
            "version": champion.version,
            "type": champion.type,
            "symbol": champion.symbol,
            "timeframe": champion.timeframe,
            "params": champion.params,
            "filters": champion.filters.model_dump(),
            "risk": champion.risk.model_dump(),
        },
        "live_state": {
            "peak_equity_usd": state.peak_equity_usd,
            "day_start_equity_usd": state.day_start_equity_usd,
            "day_start_date": state.day_start_date,
            "trades_today": state.trades_today,
            "kill_switch_tripped": state.kill_switch_tripped,
            "last_tick_utc": state.last_tick_utc,
        },
        "performance": {},
        "recent_trades": [],
        "recent_backtests": [],
        "recent_proposals": [],
        "recent_guardrail_events": [],
    }
    # Heartbeat: how long since the scheduler last ticked.
    if state.last_tick_utc:
        try:
            last = datetime.fromisoformat(state.last_tick_utc)
            age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
            snapshot["live_state"]["minutes_since_last_tick"] = round(age_min, 1)
        except ValueError:
            pass
    if s.db_path.exists():
        with db.session(s.db_path) as conn:
            for row in conn.execute(
                "SELECT id, version, symbol, side, qty, entry_ts, entry_px, exit_ts, exit_px, "
                "pnl_usd, reason_entry, reason_exit, regime "
                "FROM trades ORDER BY id DESC LIMIT 5"
            ):
                snapshot["recent_trades"].append(dict(row))
            for row in conn.execute(
                "SELECT id, version, window_start, window_end, oos_flag, sharpe, max_dd, n_trades, equity_final "
                "FROM backtests ORDER BY id DESC LIMIT 5"
            ):
                snapshot["recent_backtests"].append(dict(row))
            for row in conn.execute(
                "SELECT id, ts, parent_version, decision, reject_reason "
                "FROM proposals ORDER BY id DESC LIMIT 5"
            ):
                snapshot["recent_proposals"].append(dict(row))
            for row in conn.execute(
                "SELECT id, ts, kind, detail_json "
                "FROM guardrail_events ORDER BY id DESC LIMIT 5"
            ):
                snapshot["recent_guardrail_events"].append(dict(row))

            perf: dict = {}
            first = conn.execute(
                "SELECT equity_usd FROM equity_curve ORDER BY ts ASC LIMIT 1"
            ).fetchone()
            last = conn.execute(
                "SELECT equity_usd, exposure_usd FROM equity_curve ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if first and last and first["equity_usd"]:
                perf["since_inception_return"] = round(
                    last["equity_usd"] / first["equity_usd"] - 1.0, 4
                )
                perf["current_equity_usd"] = last["equity_usd"]
                perf["current_exposure_usd"] = last["exposure_usd"]
            wr = conn.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS w "
                "FROM trades WHERE exit_ts IS NOT NULL"
            ).fetchone()
            if wr and wr["n"]:
                perf["closed_trades"] = wr["n"]
                perf["win_rate_to_date"] = round((wr["w"] or 0) / wr["n"], 3)
            snapshot["performance"] = perf
    print(json.dumps(snapshot, indent=2, default=str))
    return 0


def _reset_kill_switch(args: argparse.Namespace) -> int:
    s = config.load()
    state = LiveState.load(s.live_state_path)
    if not state.kill_switch_tripped:
        print(f"kill switch already clear at {s.live_state_path}")
        return 0
    state.kill_switch_tripped = False
    state.save(s.live_state_path)
    print(f"kill switch cleared at {s.live_state_path}")
    return 0


def _make_broker(s: config.Settings):
    if s.broker == "mock":
        from .execution.mock_broker import MockBroker

        log.info("using MockBroker (state at %s)", s.mock_broker_path)
        return MockBroker(state_path=s.mock_broker_path)
    if s.broker == "alpaca":
        if s.asset_class == "crypto":
            from .execution.broker import AlpacaCryptoBroker

            return AlpacaCryptoBroker()
        from .execution.broker import AlpacaStockBroker

        log.info("using AlpacaStockBroker (feed=%s)", s.stock_feed)
        return AlpacaStockBroker()
    raise ValueError(f"unknown AGENT_BROKER={s.broker!r}; expected 'alpaca' or 'mock'")


def _reconcile_start(broker, champion, s: config.Settings) -> None:
    """Sync LiveState with the broker's real position at startup.

    A crash/restart must not leave the loop acting on stale state. If we tracked
    an open position but the broker is flat (a resting stop/take fired, or it was
    closed out of band), record the close and clear. If the broker holds a
    position we don't know about, warn rather than guess.
    """
    state = LiveState.load(s.live_state_path)
    held = broker.positions().get(champion.symbol, 0.0)
    if held <= 0 and state.open_entry_px > 0:
        with db.session(s.db_path) as conn:
            evlogger.close_open_trade(
                conn,
                symbol=champion.symbol,
                version=champion.version,
                exit_ts=datetime.now(timezone.utc).isoformat(),
                exit_px=state.open_entry_px,  # best estimate; real fill unknown at restart
                reason_exit="reconciled_flat",
            )
        log.warning("reconcile: broker flat but state tracked a position; cleared")
        state.clear_open()
        state.save(s.live_state_path)
    elif held > 0 and state.open_entry_px == 0:
        log.warning(
            "reconcile: broker holds %s %s but live state has no open position; "
            "exit management left to resting orders — verify manually",
            held,
            champion.symbol,
        )


def _paper(args: argparse.Namespace) -> int:
    from .guardrails import risk
    from .strategy.registry import from_spec

    s = config.load()
    if s.live_mode and s.broker == "alpaca":
        import time

        print(
            "\n*** LIVE TRADING MODE ACTIVE ***\n"
            "Orders will execute against the real Alpaca endpoint using real money.\n"
            "All guardrails are enforced. Press Ctrl-C within 5 seconds to abort.\n",
            flush=True,
        )
        time.sleep(5)
    champion = load_spec(s.strategies_dir / "champion.json")
    broker = _make_broker(s)
    _reconcile_start(broker, champion, s)

    def tick() -> None:
        state = LiveState.load(s.live_state_path)
        # Heartbeat first, so even skipped ticks (closed market, stale data,
        # kill switch) prove the scheduler is alive.
        state.last_tick_utc = datetime.now(timezone.utc).isoformat()
        state.save(s.live_state_path)
        if state.kill_switch_tripped:
            log.warning("kill switch tripped; tick skipped")
            return

        if not broker.is_market_open():
            log.info("market closed; tick skipped")
            return

        bars = data.fetch_live(champion.symbol, champion.timeframe)
        sig = from_spec(champion).signals(bars)
        last_ts = bars.index[-1]
        last_close = float(bars["close"].iloc[-1])

        # Staleness circuit breaker: don't trade on a stale last bar (data outage).
        age = data.bar_age_seconds(bars)
        max_age = config.MAX_BAR_AGE_SECONDS.get(champion.timeframe, 0)
        if max_age and age > max_age:
            with db.session(s.db_path) as conn:
                evlogger.log_guardrail(
                    conn,
                    "stale_data",
                    {"age_seconds": age, "timeframe": champion.timeframe, "last_bar": last_ts.isoformat()},
                )
            log.warning("stale data: last bar %.0fs old (max %ds); tick skipped", age, max_age)
            return

        broker.mark(champion.symbol, last_close)

        eq = broker.equity()
        cash = broker.cash()
        positions = broker.positions()
        held_qty = positions.get(champion.symbol, 0.0)
        day_trades = broker.daytrade_count()
        today = datetime.now(timezone.utc).date().isoformat()
        state.observe(eq, today)

        # Hard drawdown trip is sticky once set.
        if state.peak_equity_usd > 0:
            dd = eq / state.peak_equity_usd - 1.0
            if dd <= config.HARD_DD_KILL_PCT:
                state.kill_switch_tripped = True
                state.save(s.live_state_path)
                with db.session(s.db_path) as conn:
                    evlogger.log_guardrail(
                        conn,
                        "hard_dd_kill",
                        {"dd": dd, "equity": eq, "peak": state.peak_equity_usd},
                    )
                log.warning("hard DD kill tripped at %.2f%%", dd * 100)
                notify("kill_switch", {"dd_pct": round(dd * 100, 2), "equity": eq}, s)
                return

        exposure = held_qty * last_close
        with db.session(s.db_path) as conn:
            evlogger.log_equity(
                conn, last_ts.isoformat(), eq, cash, exposure, state.peak_equity_usd
            )

        regime_label = str(regime_mod.classify(bars).iloc[-1])

        # A resting stop/take (exchange-managed) fired between ticks: we tracked
        # an open position but the broker now shows flat. Record the exit.
        if broker.supports_resting_orders and state.open_entry_px > 0 and held_qty == 0:
            # Recover the exact exchange-managed leg fill so the exit price and its
            # slippage are real, not estimated from the bar close.
            leg = broker.last_exit_fill(champion.symbol)
            exit_px = leg.avg_price if leg and leg.avg_price > 0 else last_close
            # Intended = the resting level the fill is closest to (stop vs take).
            intended_px = exit_px
            if leg and state.open_stop_px and state.open_take_px:
                intended_px = (
                    state.open_stop_px
                    if abs(exit_px - state.open_stop_px) <= abs(exit_px - state.open_take_px)
                    else state.open_take_px
                )
            with db.session(s.db_path) as conn:
                evlogger.close_open_trade(
                    conn,
                    symbol=champion.symbol,
                    version=champion.version,
                    exit_ts=last_ts.isoformat(),
                    exit_px=exit_px,
                    reason_exit="resting_stop_or_take",
                )
                if leg:
                    _record_execution(
                        conn, version=champion.version, symbol=champion.symbol,
                        side="sell", order_type="resting_leg",
                        intended_px=intended_px, fill=leg,
                    )
            log.info("resting stop/take fired for %s; position flat", champion.symbol)
            notify("exit", {"symbol": champion.symbol, "reason": "resting_stop_or_take"}, s)
            state.clear_open()
            state.save(s.live_state_path)
            return

        # Venues without resting orders (crypto/mock): simulate the stop/take at
        # tick cadence using the SAME decision as the backtester.
        if (not broker.supports_resting_orders) and held_qty > 0 and state.open_entry_px > 0:
            low = float(bars["low"].iloc[-1])
            high = float(bars["high"].iloc[-1])
            reason_exit, exit_px = exits.exit_for_levels(
                low, high, state.open_stop_px or float("nan"), state.open_take_px or float("nan")
            )
            if exit_px is not None:
                coid = make_client_order_id(champion.symbol, f"sell-{reason_exit}", last_ts.isoformat())
                fill = broker.submit_market(
                    champion.symbol, "sell", held_qty, client_order_id=coid
                )
                with db.session(s.db_path) as conn:
                    evlogger.close_open_trade(
                        conn,
                        symbol=champion.symbol,
                        version=champion.version,
                        exit_ts=last_ts.isoformat(),
                        exit_px=exit_px,
                        reason_exit=reason_exit,
                    )
                    _record_execution(
                        conn, version=champion.version, symbol=champion.symbol,
                        side="sell", order_type="market",
                        intended_px=exit_px, fill=fill, client_order_id=coid,
                    )
                log.info("%s exit %s qty=%s px=%s", reason_exit, fill.symbol, fill.qty, exit_px)
                notify("exit", {"symbol": champion.symbol, "reason": reason_exit, "px": exit_px}, s)
                state.clear_open()
                state.save(s.live_state_path)
                return

        if bool(sig.at[last_ts, "entry"]) and held_qty == 0:
            # Match the backtester: need a valid stop before sizing/entering.
            stop_px = float(sig.at[last_ts, "stop_px"])
            if math.isnan(stop_px) or stop_px >= last_close:
                log.info("skip entry: no valid stop (stop_px=%s price=%s)", stop_px, last_close)
                state.save(s.live_state_path)
                return
            risk_per_unit = last_close - stop_px
            # Fractional-Kelly scales by the realized edge, but only once enough
            # closed trades exist to trust it; otherwise edge stays None and
            # size_position falls back to fixed / risk-targeted sizing.
            k_win = k_payoff = None
            if champion.risk.kelly_fraction:
                with db.session(s.db_path) as conn:
                    n_closed, k_win, k_payoff = _realized_edge(
                        conn, champion.symbol, champion.version
                    )
                if n_closed < config.KELLY_MIN_TRADES:
                    k_win = k_payoff = None
            qty = round(
                sizing.size_position(
                    eq,
                    last_close,
                    risk_per_unit,
                    champion.risk.position_pct,
                    champion.risk.risk_per_trade_pct,
                    kelly_fraction=champion.risk.kelly_fraction,
                    win_rate=k_win,
                    payoff_ratio=k_payoff,
                ),
                6,
            )
            notional = qty * last_close
            order = risk.Order(champion.symbol, "buy", qty, notional)
            rstate = risk.PortfolioState(
                equity_usd=eq,
                peak_equity_usd=state.peak_equity_usd,
                day_start_equity_usd=state.day_start_equity_usd,
                trades_today=state.trades_today,
                current_qty=held_qty,
                kill_switch_tripped=state.kill_switch_tripped,
                day_trades_in_window=day_trades,
            )
            decision = risk.allow(order, rstate)
            if not decision.allowed:
                with db.session(s.db_path) as conn:
                    evlogger.log_guardrail(
                        conn, decision.reason, {"order": order.__dict__, "equity": eq}
                    )
                log.warning("guardrail blocked entry: %s", decision.reason)
                notify("guardrail_block", {"side": "buy", "reason": decision.reason}, s)
                state.save(s.live_state_path)
                return
            coid = make_client_order_id(champion.symbol, "buy", last_ts.isoformat())
            if broker.supports_resting_orders:
                take_px = exits.take_from_stop(last_close, stop_px, champion.risk.take_profit_r)
                fill = broker.submit_bracket(
                    champion.symbol, "buy", qty, stop_px, take_px, client_order_id=coid
                )
            else:
                fill = broker.submit_market(
                    champion.symbol, "buy", qty, client_order_id=coid
                )
                take_px = exits.take_from_stop(
                    fill.avg_price, stop_px, champion.risk.take_profit_r
                )
            state.open_entry_px = fill.avg_price
            state.open_stop_px = stop_px
            state.open_take_px = take_px
            state.trades_today += 1
            with db.session(s.db_path) as conn:
                evlogger.log_trade(
                    conn,
                    version=champion.version,
                    symbol=champion.symbol,
                    side="long",
                    qty=fill.qty,
                    entry_ts=last_ts.isoformat(),
                    entry_px=fill.avg_price,
                    reason_entry=str(sig.at[last_ts, "reason"]),
                    regime=regime_label,
                    meta={"order_id": fill.order_id},
                )
                _record_execution(
                    conn, version=champion.version, symbol=champion.symbol,
                    side="buy",
                    order_type="bracket" if broker.supports_resting_orders else "market",
                    intended_px=last_close, fill=fill, client_order_id=coid,
                )
            log.info("filled %s qty=%s px=%s", fill.symbol, fill.qty, fill.avg_price)
            notify(
                "entry_fill",
                {"symbol": fill.symbol, "qty": fill.qty, "px": fill.avg_price},
                s,
            )

        elif bool(sig.at[last_ts, "exit"]) and held_qty > 0:
            order = risk.Order(champion.symbol, "sell", held_qty, held_qty * last_close)
            rstate = risk.PortfolioState(
                equity_usd=eq,
                peak_equity_usd=state.peak_equity_usd,
                day_start_equity_usd=state.day_start_equity_usd,
                trades_today=state.trades_today,
                current_qty=held_qty,
                kill_switch_tripped=state.kill_switch_tripped,
                day_trades_in_window=day_trades,
            )
            decision = risk.allow(order, rstate)
            if not decision.allowed:
                with db.session(s.db_path) as conn:
                    evlogger.log_guardrail(
                        conn, decision.reason, {"order": order.__dict__, "equity": eq}
                    )
                log.warning("guardrail blocked exit: %s", decision.reason)
                notify("guardrail_block", {"side": "sell", "reason": decision.reason}, s)
                state.save(s.live_state_path)
                return
            broker.cancel_open_orders(champion.symbol)  # clear resting bracket legs first
            coid = make_client_order_id(champion.symbol, "sell-signal", last_ts.isoformat())
            fill = broker.submit_market(
                champion.symbol, "sell", held_qty, client_order_id=coid
            )
            with db.session(s.db_path) as conn:
                evlogger.close_open_trade(
                    conn,
                    symbol=champion.symbol,
                    version=champion.version,
                    exit_ts=last_ts.isoformat(),
                    exit_px=fill.avg_price,
                    reason_exit="signal_exit",
                )
                _record_execution(
                    conn, version=champion.version, symbol=champion.symbol,
                    side="sell", order_type="market",
                    intended_px=last_close, fill=fill, client_order_id=coid,
                )
            state.clear_open()
            log.info("exited %s qty=%s px=%s", fill.symbol, fill.qty, fill.avg_price)
            notify(
                "exit",
                {"symbol": fill.symbol, "reason": "signal_exit", "px": fill.avg_price},
                s,
            )

        state.save(s.live_state_path)

    if args.once:
        tick()
        return 0

    from . import scheduler

    def improve_job() -> None:
        _improve(
            argparse.Namespace(
                dry_run=False, fixture_proposal=None, oos_dir=None, expect="reject"
            )
        )

    # Daily-bar strategies only need one tick per trading day; intraday bars tick hourly.
    cadence = "daily_open" if champion.timeframe == "1d" else "hourly"
    scheduler.run_forever(tick, improve_job, cadence=cadence)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trading_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db").set_defaults(func=_init_db)

    ck = sub.add_parser("check-keys")
    ck.add_argument("--json", action="store_true", help="machine-readable output")
    ck.add_argument(
        "--model",
        default=None,
        help="Anthropic model to verify (default: a cheap Haiku call). "
        "Pass 'proposer' to verify your configured AGENT_PROPOSER_MODEL, "
        "or an explicit id like claude-sonnet-4-6.",
    )
    ck.set_defaults(func=_check_keys)

    fb = sub.add_parser("fetch-bars")
    fb.add_argument("--symbol", default="SPY")
    fb.add_argument("--timeframe", default="1d", choices=["1h", "4h", "1d"])
    fb.add_argument("--start", required=True, help="ISO date, e.g. 2024-01-01")
    fb.add_argument("--end", required=True, help="ISO date, e.g. 2024-09-30")
    fb.add_argument("--to", required=True, choices=["is", "oos"])
    fb.add_argument("--force", action="store_true", help="overwrite existing file")
    fb.set_defaults(func=_fetch_bars)

    bt = sub.add_parser("backtest")
    bt.add_argument("--spec", required=True)
    bt.add_argument("--bars", required=True, help="CSV of OHLCV bars")
    bt.add_argument("--window", help="ISO start:end slice", default=None)
    bt.add_argument("--oos", action="store_true", help="mark this run as OOS in the DB")
    bt.set_defaults(func=_backtest)

    imp = sub.add_parser("improve")
    imp.add_argument("--dry-run", action="store_true")
    imp.add_argument(
        "--fixture-proposal",
        help="path to a JSON challenger spec; skip Claude API and use this instead",
    )
    imp.add_argument("--oos-dir", help="override OOS bars dir (used by worked example)")
    imp.add_argument(
        "--n-challengers",
        type=int,
        default=1,
        help="generate N challengers and promote the best OOS gate-passer (default 1)",
    )
    imp.add_argument(
        "--expect",
        choices=["accept", "reject"],
        default="reject",
        help="dry-run exit code: 0 if decision matches, 2 if it doesn't",
    )
    imp.set_defaults(func=_improve)

    pap = sub.add_parser("paper")
    pap.add_argument("--once", action="store_true")
    pap.set_defaults(func=_paper)

    sub.add_parser("status").set_defaults(func=_status)
    sub.add_parser("reset-kill-switch").set_defaults(func=_reset_kill_switch)

    args = parser.parse_args(argv)
    s = config.load()
    config.configure_logging(s.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
