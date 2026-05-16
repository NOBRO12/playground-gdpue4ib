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

- `data/is/BTC-USD_1h_2024Q1.csv` — 2,160 hourly bars (90 days), seed=42.
  Used by the proposer for trade-log summarization.
- `data/oos/BTC-USD_1h_2024Q2.csv` — 1,440 hourly bars (60 disjoint days),
  seed=99. The proposer **never** sees these; only the promoter does.
- `data/agent.db` with the schema in place and the seed champion row
  (`v0-baseline`) recorded.

The committed champion is `strategies/champion.json`:

```json
{ "type": "donchian", "symbol": "BTC/USD", "timeframe": "1h",
  "params": { "entry_lookback": 20, "exit_lookback": 10 },
  "filters": {}, "risk": { "atr_mult_stop": 2.0, "take_profit_r": 3.0,
  "position_pct": 0.10 }, "version": "v0-baseline", "parent": null }
```

## 1. Trade log from the in-sample window

```bash
python -m trading_agent backtest --spec strategies/champion.json \
    --bars data/is/BTC-USD_1h_2024Q1.csv
```

Recorded scorecard on the IS window (deterministic):

| metric        | value     |
|---------------|-----------|
| sharpe        |  0.2271   |
| sortino       |  0.2020   |
| max_dd        | -0.0281   |
| n_trades      |    65     |
| win_rate      |  0.3231   |
| expectancy_r  |  0.0165   |
| equity_final  | 100,327   |

The trade ledger is persisted in `trades` (and the run is logged in
`backtests`). A summary of the trades — n_trades, sharpe, win_rate,
by-regime mean R — is what the proposer is allowed to see. Raw OOS bars
are not touched at this step.

## 2. LLM proposal

In a real run, `trading_agent.improvement.proposer.propose()` calls Claude
with a forced `submit_challenger_spec` tool, and the JSON output is
validated against `StrategySpec`. For this example we ship the recorded
proposal as a fixture so the loop is reproducible without API keys:

`examples/worked_example_artifacts/proposal.json`

```json
{
  "type": "donchian", "symbol": "BTC/USD", "timeframe": "1h",
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
  "reason": "oos_sharpe_regression",
  "champion":   { "sharpe": 0.6548, "max_dd": -0.0202, "n_trades": 49 },
  "challenger": { "sharpe": 0.2251, "max_dd": -0.0254, "n_trades": 41 }
}
```

The promoter scored **both** specs on the held-out OOS window. The
challenger's Sharpe collapsed (0.65 → 0.23) once it left the data it was
implicitly tuned to. The gate rejects with reason `oos_sharpe_regression`
(Sharpe delta of -0.43 is far below the required +0.10).

## 4. What the audit trail shows

If you drop `--dry-run`:

- `proposals` row inserted with `decision = "rejected"`,
  `reject_reason = "oos_sharpe_regression"`, the prompt hash, and the
  full response JSON.
- `strategy_versions` row for `v-challenger-001` is `status = "rejected"`
  with a note recording the reason. The champion row is **unchanged**.
- The rejected spec is archived to
  `strategies/archive/v-challenger-001_rejected.json`.
- `strategies/champion.json` is byte-identical to before the run.

## Why this matters

The proposer was given the right context (regime breakdown, recent losses)
and made a plausible suggestion. It still failed OOS. This is the entire
reason the design exists: the LLM is *expected* to occasionally fit noise,
and the only thing standing between that and the live book is the
champion-vs-challenger gate. Weaken the gate and the system becomes a
self-overfitting machine.
