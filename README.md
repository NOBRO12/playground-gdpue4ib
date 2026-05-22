# trading_agent — autonomous AI crypto paper-trading agent

A four-layer paper-trading agent for **BTC/USD** (and optionally ETH/USD)
on Alpaca's crypto paper endpoint. A rule-based strategy runs the book; an
LLM (Claude) proposes parameter and filter changes; a deterministic
backtester evaluates challengers on out-of-sample bars the proposer never
sees; a promotion gate decides whether the new spec replaces the
champion. Everything is versioned, audited, and gated by hard risk
limits.

This is **not** a profit machine. It's a disciplined experiment framework
designed to prevent the most obvious failure mode of self-improving
trading systems: the LLM fitting its own training signal.

## Layers

| Layer            | Module                          | Responsibility                              |
|------------------|---------------------------------|---------------------------------------------|
| Execution        | `trading_agent.execution`       | Alpaca paper broker, bar data, portfolio    |
| Strategy         | `trading_agent.strategy`        | Donchian / EMA crossover (spec-driven)      |
| Evaluation       | `trading_agent.evaluation`      | SQLite, metrics, regime, trade logger       |
| Self-improvement | `trading_agent.improvement`     | Proposer, walk-forward backtester, promoter |
| Guardrails       | `trading_agent.guardrails`      | Pre-trade risk gate (non-negotiable limits) |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in ALPACA_KEY / ALPACA_SECRET / ANTHROPIC_API_KEY

python examples/worked_example_artifacts/generate_fixtures.py
python -m trading_agent init-db
pytest tests/ -q

# Reproduce the worked example end-to-end (no API keys needed):
python -m trading_agent improve --dry-run \
    --fixture-proposal examples/worked_example_artifacts/proposal.json
```

See `examples/worked_example.md` for a full trace.

## Operator runbook

End-to-end deployment + monitoring guide: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
Short version: green tests are not the same as functional — you also need
Alpaca paper credentials, an Anthropic key, real BTC/USD bars in `data/is/`
and `data/oos/`, and a 24/7 host with persistent disk for `agent.db` and
`live_state.json`.

**Don't have an Alpaca account yet?** Set `AGENT_BROKER=mock` to use the
in-process simulator. `fetch-bars`, `backtest`, `improve`, and `paper --once`
all work offline — only real fills require Alpaca. See `docs/RUNBOOK.md` §0.

## CLI

```
python -m trading_agent init-db
python -m trading_agent fetch-bars --symbol BTC/USD --timeframe 1h \
    --start 2024-01-01 --end 2024-09-30 --to is
python -m trading_agent backtest --spec <spec.json> --bars <bars.csv> [--window START:END] [--oos]
python -m trading_agent improve [--dry-run] [--fixture-proposal <path>] [--oos-dir <dir>] [--expect accept|reject]
python -m trading_agent paper [--once]
python -m trading_agent status
python -m trading_agent reset-kill-switch
```

`improve --dry-run --expect=<accept|reject>` exits 0 when the decision matches,
2 when it doesn't. CI uses this to assert the worked example's rejection path.

## Live state

The live loop persists peak equity, day-start equity, day boundary, trades-today,
and the kill-switch flag to `data/live_state.json` (override with
`AGENT_LIVE_STATE_PATH`). Without this, the guardrails would be reconstructed
from "current equity" each tick and never fire. The kill switch is sticky once
tripped; clear it manually with `python -m trading_agent reset-kill-switch`.

## Strategy spec — the LLM's only output surface

The LLM never emits Python. It calls a Claude tool whose `input_schema` is
generated from `trading_agent.strategy.spec.StrategySpec`. Anything outside
the schema fails validation before reaching the backtester:

```python
class StrategySpec(BaseModel):
    type: Literal["donchian", "ema_cross"]
    symbol: Literal["BTC/USD", "ETH/USD"]
    timeframe: Literal["1h", "4h", "1d"]
    params: dict             # type-specific, validated per type
    filters: Filters         # adx_min, vol_max
    risk: RiskSpec           # atr_mult_stop, take_profit_r, position_pct (<= 0.25)
    version: str
    parent: Optional[str]
```

## Guardrails (`trading_agent.guardrails.risk`)

Hard-coded, **not** LLM-tunable, all enforced via a single
`risk.allow(order, state) -> Decision` call before every order:

- `HARD_DD_KILL = -10%` from peak equity → flatten and pause until reset
- `MAX_DAILY_LOSS = -3%` of UTC-day starting equity → no new entries
- `MAX_TRADES_PER_DAY = 8`
- `MAX_POSITION_PCT = 25%` of equity per symbol
- `LEVERAGE = 1.0`, `NO_SHORTS = True`

Blocks write a row to `guardrail_events` with full context.

## Promotion gate (`trading_agent.improvement.promoter`)

Both specs are scored on the same OOS bars. Challenger is promoted only if
**all** criteria hold:

- `oos_n_trades(challenger) >= 30`
- `oos_sharpe(challenger) >= 0.20` (absolute floor — prevents promotion when
  the champion has degenerated and any half-decent challenger clears the delta)
- `oos_sharpe(challenger) - oos_sharpe(champion) >= 0.10`
- `|oos_max_dd(challenger)| <= 1.2 * |oos_max_dd(champion)|`

On accept: atomic swap of `strategies/champion.json`, old champion moves to
`strategies/archive/`, status flipped in `strategy_versions`.
On reject: challenger archived with reason, champion unchanged.

## Database schema (`data/agent.db`)

`trades`, `equity_curve`, `strategy_versions`, `backtests`, `proposals`,
`guardrail_events`. See `trading_agent/evaluation/db.py` for full DDL.
Every proposal — accepted or not — is persisted with its prompt hash and
raw response JSON, so any decision is replayable.

## What this design can't do

- **It does not forecast regime shifts.** It reacts after one has begun.
- **Backtests are not live.** Slippage, partial fills, exchange outages,
  funding (if you move to perps) — not fully modeled. Constant fee+slip
  in bps is an approximation, not a guarantee.
- **The LLM will overfit narratives** to recent losses. The promoter is
  the only thing standing between that bias and the live book. Do not
  weaken the gate. The worked example shows exactly this failure mode and
  the gate catching it.
- **Multiple-testing bias accumulates** across proposals. Cap proposals at
  1/week and rotate the OOS window quarterly so it isn't reused to bless
  proposer-seen ideas twice.
- **No tail-event modeling.** Flash crashes, exchange halts, stablecoin
  depeg — out of scope here.
- **Paper ≠ live.** Run paper for *at least a few months of forward data*
  before even thinking about real capital. There is no `paper=False` path
  in this repo by design.

## Layout

```
trading_agent/
  config.py                 # env + guardrail constants
  cli.py / __main__.py      # subcommands
  scheduler.py              # APScheduler jobs
  execution/                # broker, data, portfolio, state
  strategy/                 # spec (LLM contract), donchian, ema_cross, registry
  evaluation/               # db, logger, metrics, regime
  improvement/              # proposer, backtester, promoter, versioning
  guardrails/               # risk
strategies/                 # champion.json + challengers/ + archive/
data/                       # is/, oos/, agent.db
examples/                   # worked_example.md + artifacts
tests/                      # backtester, guardrails, promoter, spec schema
```
