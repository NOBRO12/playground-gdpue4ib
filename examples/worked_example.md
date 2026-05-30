# Worked Example — Trade Logs → LLM Proposal → Backtest → Decision

This example walks the self-improvement loop end-to-end against committed
fixture data. No API keys are required. Re-running it produces the same
numbers, because all randomness is seeded.

## 0. Setup

```bash
pip install -r requirements.txt
python examples/worked_example_artifacts/generate_fixtures.py
python -m trading_agent init-db
```

This produces:

- `data/is/SPY_1d_IS.csv` — 750 daily bars (~3 years), seed=42. Used by the
  proposer for trade-log summarization.
- `data/oos/SPY_1d_OOS.csv` — 500 daily bars (~2 disjoint years), seed=99.
  The proposer **never** sees these; only the promoter does.
- `data/agent.db` with the schema in place and the seed champion row
  (`v0-baseline`) recorded.

The committed champion is `strategies/champion.json`:

```json
{ "type": "donchian", "symbol": "SPY", "timeframe": "1d",
  "params": { "entry_lookback": 20, "exit_lookback": 10 },
  "filters": {}, "risk": { "atr_mult_stop": 2.0, "take_profit_r": 3.0,
  "position_pct": 0.10 }, "version": "v0-baseline", "parent": null }
```

A 20-day / 10-day Donchian channel on daily SPY bars — positions are held
across days, so it is a swing strategy that essentially never day-trades
(keeping it clear of the Pattern Day Trader rule).

## 1. Trade log from the in-sample window

```bash
python -m trading_agent backtest --spec strategies/champion.json \
    --bars data/is/SPY_1d_IS.csv
```

Recorded scorecard on the IS window (deterministic):

| metric        | value     |
|---------------|-----------|
| sharpe        | -0.1140   |
| sortino       | -0.0937   |
| max_dd        | -0.0153   |
| n_trades      |    18     |
| win_rate      |  0.2222   |
| expectancy_r  | -0.0662   |
| equity_final  |  99,738   |

The IS window is a losing one for the baseline. The trade ledger is persisted
in `trades` (and the run is logged in `backtests`). A summary of the trades —
n_trades, sharpe, win_rate, by-regime mean R — is what the proposer is allowed
to see. Raw OOS bars are not touched at this step.

## 2. LLM proposal

In a real run, `trading_agent.improvement.proposer.propose()` calls Claude
with a forced `submit_challenger_spec` tool, and the JSON output is
validated against `StrategySpec`. For this example we ship the recorded
proposal as a fixture so the loop is reproducible without API keys:

`examples/worked_example_artifacts/proposal.json`

```json
{
  "type": "donchian", "symbol": "SPY", "timeframe": "1d",
  "params": { "entry_lookback": 20, "exit_lookback": 10 },
  "filters": { "adx_min": 20, "vol_max": null },
  "risk": { "atr_mult_stop": 1.5, "take_profit_r": 3.0, "position_pct": 0.10 },
  "version": "v-challenger-001", "parent": "v0-baseline"
}
```

Plain English: tighten the stop (2.0 → 1.5 ATR) and only enter in trending
regimes (ADX ≥ 20). Both are defensible *narratives* from the IS losses.

## 3. Backtest vs promote/reject

```bash
python -m trading_agent improve --dry-run \
    --fixture-proposal examples/worked_example_artifacts/proposal.json
```

Output (verbatim):

```json
{
  "accepted": false,
  "reason": "insufficient_trades",
  "champion":   { "sharpe": 0.3953, "max_dd": -0.0152, "n_trades": 15 },
  "challenger": { "sharpe": 0.9295, "max_dd": -0.0071, "n_trades": 11 }
}
```

The promoter scored **both** specs on the held-out OOS window. The challenger's
OOS Sharpe (0.93) actually *looks* better than the champion's (0.40) — but it
got there on only **11 trades**, far below the gate's 30-trade minimum
(`PROMOTE_MIN_TRADES`). The gate rejects with `insufficient_trades`: a handful
of lucky daily swings is not statistical evidence that the challenger is
better, no matter how good its Sharpe looks. The gate refuses to promote on
thin evidence.

(With intraday bars or a longer horizon you instead see the classic
`oos_sharpe_regression` path — the challenger's edge evaporating once it leaves
the data it was implicitly tuned to. Same discipline, different trigger.)

## 4. What the audit trail shows

If you drop `--dry-run`:

- `proposals` row inserted with `decision = "rejected"`,
  `reject_reason = "insufficient_trades"`, the prompt hash, and the
  full response JSON.
- `strategy_versions` row for `v-challenger-001` is `status = "rejected"`
  with a note recording the reason. The champion row is **unchanged**.
- The rejected spec is archived to
  `strategies/archive/v-challenger-001_rejected.json`.
- `strategies/champion.json` is byte-identical to before the run.

## Why this matters

The proposer was given the right context (regime breakdown, recent losses)
and made a plausible suggestion. The gate still refused it — here because the
out-of-sample evidence was too thin to trust. This is the entire reason the
design exists: the LLM is *expected* to occasionally fit noise or look good on
a few lucky trades, and the only thing standing between that and the live book
is the champion-vs-challenger gate. Weaken the gate — lower the trade floor,
drop the Sharpe-delta requirement — and the system becomes a self-overfitting
machine.
