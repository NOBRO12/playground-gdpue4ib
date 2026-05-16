"""Single CLI entry point.

Subcommands:
    init-db       Create tables and seed champion version row.
    backtest      Run a deterministic backtest of a spec against bars on disk.
    improve       Run the full proposer -> backtest -> promoter loop.
    paper         Live paper loop (requires Alpaca + Anthropic keys).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .evaluation import db
from .execution import data
from .improvement import backtester, promoter
from .improvement import versioning
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
        return 0 if decision.accepted else 0

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


def _paper(args: argparse.Namespace) -> int:
    from .execution.broker import AlpacaCryptoBroker
    from .execution.portfolio import Portfolio
    from .guardrails import risk

    s = config.load()
    champion = load_spec(s.strategies_dir / "champion.json")
    broker = AlpacaCryptoBroker()

    def tick() -> None:
        bars = data.fetch_live(champion.symbol, champion.timeframe)
        from .strategy.registry import from_spec

        sig = from_spec(champion).signals(bars)
        last_ts = bars.index[-1]
        last_close = float(bars["close"].iloc[-1])
        eq = broker.equity()
        positions = broker.positions()
        portfolio = Portfolio(
            cash_usd=eq - sum(positions.values()) * last_close,
            peak_equity_usd=eq,
            day_start_equity_usd=eq,
            trades_today=0,
        )
        portfolio.roll_day_if_needed({champion.symbol: last_close})

        if bool(sig.at[last_ts, "entry"]) and champion.symbol not in positions:
            notional = eq * champion.risk.position_pct
            qty = round(notional / last_close, 6)
            order = risk.Order(champion.symbol, "buy", qty, notional)
            state = risk.PortfolioState(
                equity_usd=eq,
                peak_equity_usd=portfolio.peak_equity_usd,
                day_start_equity_usd=portfolio.day_start_equity_usd,
                trades_today=portfolio.trades_today,
                open_short_qty=0.0,
                kill_switch_tripped=portfolio.kill_switch,
            )
            decision = risk.allow(order, state)
            if not decision.allowed:
                log.warning("guardrail blocked entry: %s", decision.reason)
                return
            fill = broker.submit_market(champion.symbol, "buy", qty)
            log.info("filled %s qty=%s px=%s", fill.symbol, fill.qty, fill.avg_price)
        elif bool(sig.at[last_ts, "exit"]) and champion.symbol in positions:
            qty = positions[champion.symbol]
            fill = broker.submit_market(champion.symbol, "sell", qty)
            log.info("exited %s qty=%s px=%s", fill.symbol, fill.qty, fill.avg_price)

    if args.once:
        tick()
        return 0

    from . import scheduler

    def improve_job() -> None:
        _improve(argparse.Namespace(dry_run=False, fixture_proposal=None, oos_dir=None))

    scheduler.run_forever(tick, improve_job)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trading_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db").set_defaults(func=_init_db)

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
    imp.set_defaults(func=_improve)

    pap = sub.add_parser("paper")
    pap.add_argument("--once", action="store_true")
    pap.set_defaults(func=_paper)

    args = parser.parse_args(argv)
    s = config.load()
    config.configure_logging(s.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
