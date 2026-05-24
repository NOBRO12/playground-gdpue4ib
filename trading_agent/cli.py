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
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .evaluation import db, logger as evlogger
from .evaluation import regime as regime_mod
from .execution import data
from .execution.state import LiveState
from .improvement import backtester, promoter, versioning
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
        from .improvement import proposer

        is_bars = data.load_window(s.is_dir, champion.symbol, champion.timeframe)
        champ_run = backtester.run(champion, is_bars)
        summary = {
            "n_trades": champ_run.score.n_trades,
            "sharpe": round(champ_run.score.sharpe, 3),
            "max_dd": round(champ_run.score.max_dd, 3),
            "win_rate": round(champ_run.score.win_rate, 3),
            "expectancy_r": round(champ_run.score.expectancy_r, 3),
            "by_regime": (
                champ_run.trades.groupby("regime")["pnl_r"].agg(["count", "mean"]).to_dict()
                if len(champ_run.trades)
                else {}
            ),
        }
        result = proposer.propose(champion, summary)
        challenger = result.spec
        prompt_hash = result.prompt_hash
        response_json = result.response_json

    oos_bars = data.load_window(
        Path(args.oos_dir) if args.oos_dir else s.oos_dir,
        challenger.symbol,
        challenger.timeframe,
    )
    decision = promoter.decide(champion, challenger, oos_bars)

    print(
        json.dumps(
            {
                "accepted": decision.accepted,
                "reason": decision.reason,
                "champion": decision.champion_score.__dict__,
                "challenger": decision.challenger_score.__dict__,
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
    return 0


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
        },
        "recent_trades": [],
        "recent_backtests": [],
        "recent_proposals": [],
        "recent_guardrail_events": [],
    }
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
        from .execution.broker import AlpacaCryptoBroker

        return AlpacaCryptoBroker()
    raise ValueError(f"unknown AGENT_BROKER={s.broker!r}; expected 'alpaca' or 'mock'")


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

    def tick() -> None:
        state = LiveState.load(s.live_state_path)
        if state.kill_switch_tripped:
            log.warning("kill switch tripped; tick skipped")
            return

        bars = data.fetch_live(champion.symbol, champion.timeframe)
        sig = from_spec(champion).signals(bars)
        last_ts = bars.index[-1]
        last_close = float(bars["close"].iloc[-1])
        broker.mark(champion.symbol, last_close)

        eq = broker.equity()
        cash = broker.cash()
        positions = broker.positions()
        held_qty = positions.get(champion.symbol, 0.0)
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
                return

        exposure = held_qty * last_close
        with db.session(s.db_path) as conn:
            evlogger.log_equity(
                conn, last_ts.isoformat(), eq, cash, exposure, state.peak_equity_usd
            )

        regime_label = str(regime_mod.classify(bars).iloc[-1])

        if bool(sig.at[last_ts, "entry"]) and held_qty == 0:
            notional = eq * champion.risk.position_pct
            qty = round(notional / last_close, 6)
            order = risk.Order(champion.symbol, "buy", qty, notional)
            rstate = risk.PortfolioState(
                equity_usd=eq,
                peak_equity_usd=state.peak_equity_usd,
                day_start_equity_usd=state.day_start_equity_usd,
                trades_today=state.trades_today,
                current_qty=held_qty,
                kill_switch_tripped=state.kill_switch_tripped,
            )
            decision = risk.allow(order, rstate)
            if not decision.allowed:
                with db.session(s.db_path) as conn:
                    evlogger.log_guardrail(
                        conn, decision.reason, {"order": order.__dict__, "equity": eq}
                    )
                log.warning("guardrail blocked entry: %s", decision.reason)
                state.save(s.live_state_path)
                return
            fill = broker.submit_market(champion.symbol, "buy", qty)
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
            log.info("filled %s qty=%s px=%s", fill.symbol, fill.qty, fill.avg_price)

        elif bool(sig.at[last_ts, "exit"]) and held_qty > 0:
            order = risk.Order(champion.symbol, "sell", held_qty, held_qty * last_close)
            rstate = risk.PortfolioState(
                equity_usd=eq,
                peak_equity_usd=state.peak_equity_usd,
                day_start_equity_usd=state.day_start_equity_usd,
                trades_today=state.trades_today,
                current_qty=held_qty,
                kill_switch_tripped=state.kill_switch_tripped,
            )
            decision = risk.allow(order, rstate)
            if not decision.allowed:
                with db.session(s.db_path) as conn:
                    evlogger.log_guardrail(
                        conn, decision.reason, {"order": order.__dict__, "equity": eq}
                    )
                log.warning("guardrail blocked exit: %s", decision.reason)
                state.save(s.live_state_path)
                return
            fill = broker.submit_market(champion.symbol, "sell", held_qty)
            with db.session(s.db_path) as conn:
                evlogger.close_open_trade(
                    conn,
                    symbol=champion.symbol,
                    version=champion.version,
                    exit_ts=last_ts.isoformat(),
                    exit_px=fill.avg_price,
                    reason_exit="signal_exit",
                )
            log.info("exited %s qty=%s px=%s", fill.symbol, fill.qty, fill.avg_price)

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

    scheduler.run_forever(tick, improve_job)
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
    fb.add_argument("--symbol", default="BTC/USD")
    fb.add_argument("--timeframe", default="1h", choices=["1h", "4h", "1d"])
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
