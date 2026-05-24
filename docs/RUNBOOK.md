# Operator Runbook

Practical steps to take this repo from "green tests" to "running continuously
against Alpaca paper crypto." ~80% of the work below is not code — it is
credentials, hosting, real bars, and monitoring.

## 0. Offline development (no Alpaca account)

Everything except real fills works without an Alpaca account. The crypto
historical endpoint is unauthenticated, and a `MockBroker` stands in for
the trading endpoint so you can exercise the full live tick path —
including DB writes, equity drift, and the kill-switch trip — entirely
offline.

```bash
# Use the simulator instead of Alpaca.
export AGENT_BROKER=mock
export ANTHROPIC_API_KEY=sk-ant-...     # only needed for real proposer runs

python -m trading_agent init-db

# Verify your Anthropic key works before doing anything else.
python -m trading_agent check-keys

# Fetch real BTC/USD history (Alpaca crypto data is unauthenticated).
python -m trading_agent fetch-bars --symbol BTC/USD --timeframe 1h \
    --start 2023-01-01 --end 2024-09-30 --to is
python -m trading_agent fetch-bars --symbol BTC/USD --timeframe 1h \
    --start 2024-10-01 --end 2025-01-01 --to oos

# Backtest against real bars.
python -m trading_agent backtest --spec strategies/champion.json \
    --bars data/is/BTC-USD_1h_2023-01-01_2024-09-30.csv

# Run a REAL Claude proposal -> backtest -> promoter decision.
python -m trading_agent improve --dry-run

# Simulate one live tick against the mock broker.
python -m trading_agent paper --once
python -m trading_agent status

# Deliberately trip the kill switch to verify your monitoring path.
python -c "
import json
from pathlib import Path
p = Path('data/live_state.json')
s = json.loads(p.read_text())
s['peak_equity_usd'] = s.get('peak_equity_usd', 100_000.0) * 1.20
p.write_text(json.dumps(s, indent=2))
"
python -m trading_agent paper --once    # next tick will trip hard_dd_kill
python -m trading_agent status          # guardrail_events should show it
python -m trading_agent reset-kill-switch

# Reset the simulator any time:
rm -f data/mock_broker.json data/live_state.json
```

When Alpaca becomes available, set `AGENT_BROKER=alpaca` (or remove the
override — it defaults to alpaca) and add `ALPACA_KEY` / `ALPACA_SECRET`.
Nothing else changes.

## 1. One-time prerequisites

| Item | Where | Notes |
|---|---|---|
| Alpaca paper account | alpaca.markets | Free. Enable Crypto under Account → Profile. |
| Alpaca paper API key + secret | dashboard | Two strings. Keep distinct from any live keys. |
| Anthropic API key | console.anthropic.com | Pay-as-you-go. Cost depends on `AGENT_PROPOSER_MODEL`. |
| A 24/7 host | your choice | Crypto is 24/7. Anything that powers off breaks the loop. See §3. |
| Python 3.11 on the host | host | `python --version` ≥ 3.11. |

## 2. Per-deployment setup

```bash
git clone <your-fork> && cd playground-gdpue4ib
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill ALPACA_KEY, ALPACA_SECRET, ANTHROPIC_API_KEY.
# AGENT_PROPOSER_MODEL defaults to claude-sonnet-4-6.

python -m trading_agent init-db
python -m trading_agent check-keys   # verify Anthropic + Alpaca before fetching bars

# Fetch REAL bars. Do not run the loop against the synthetic fixtures.
python -m trading_agent fetch-bars --symbol BTC/USD --timeframe 1h \
    --start 2023-01-01 --end 2024-09-30 --to is
python -m trading_agent fetch-bars --symbol BTC/USD --timeframe 1h \
    --start 2024-10-01 --end 2025-01-01 --to oos

# Sanity-check the champion on real IS bars.
python -m trading_agent backtest \
    --spec strategies/champion.json \
    --bars data/is/BTC-USD_1h_2023-01-01_2024-09-30.csv

# Smoke-test the live loop without scheduling.
python -m trading_agent paper --once
python -m trading_agent status   # should show 1 equity_curve row
```

If `paper --once` errors on Alpaca auth or returns no bars, fix that before
moving on — the scheduler will silently swallow the same error every hour.

## 3. Hosting

Pick one.

1. **systemd unit on a small VPS** ($5/mo Hetzner/DO). Total control,
   easy disk backups. Recommended for individuals.
2. **Docker container on a managed host** (fly.io, Railway, render).
   Easier deploys. `data/agent.db` and `data/live_state.json` MUST live on a
   persistent volume — otherwise the audit trail and guardrails vanish on
   every restart.
3. **A laptop you never close.** Works until you close it.

Minimal systemd unit:

```ini
[Unit]
Description=trading_agent paper loop
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/trading_agent
EnvironmentFile=/opt/trading_agent/.env
ExecStart=/opt/trading_agent/.venv/bin/python -m trading_agent paper
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=multi-user.target
```

## 4. Operational checklist

| When | Task | How |
|---|---|---|
| Daily | Glance at trades + equity | `python -m trading_agent status` |
| Daily | Confirm scheduler alive | `systemctl status trading_agent` or container logs |
| Weekly | Review proposals + decisions | `status` shows last 5; SQL for more |
| Weekly | Backup `agent.db` + `live_state.json` | `cp` to another host or object storage |
| Quarterly | Rotate OOS window | Move current OOS CSVs to `data/archive/`, fetch a new disjoint window |
| As needed | Clear kill switch (after investigation) | `python -m trading_agent reset-kill-switch` |

The quarterly OOS rotation matters: once you've observed N rejected
challengers against the same OOS bars, that window is no longer truly
out-of-sample to *you* — you're implicitly selecting strategies that beat
those specific bars. Rotate at least quarterly. Not automated; calendar
reminder.

## 5. Monitoring

Minimum bar:

- **Kill-switch trip → page you immediately.** Cron job that greps the
  latest `guardrail_events` row and alerts on `kind = "hard_dd_kill"`.
- **Equity-curve gap detector.** If no `equity_curve` row in the last 2
  hours, the scheduler is dead. Alert.
- **Proposal review queue.** Even when proposals are rejected, the
  `proposals` table is the only record. Skim weekly — catches degenerate
  LLM behavior early.

Nice-to-have:

- Daily snapshot of `agent.db` to object storage.
- Flask endpoint that plots the equity curve.
- Manual flatten-all script for emergencies (today you'd use the Alpaca
  web UI).

## 6. What this is still not

Live trading is gated behind `AGENT_LIVE_MODE` (off by default) for a
reason. Before considering live capital:

1. Re-read the README's "What this design can't do" section.
2. Run paper for at least 3 months of forward data.
3. Compare paper P&L against a buy-and-hold benchmark over the same window.
4. Going live is a separate, much larger project (compliance, taxes,
   slippage, exchange selection). Not in scope here.

## 7. Switching to live trading

**Prerequisites before you flip this switch:**

1. Run paper for at least 3 months of *forward* data (not backtested).
2. Compare paper P&L to a BTC buy-and-hold benchmark over the same window.
3. Obtain a separate set of *live* Alpaca API keys. Paper keys are rejected
   by the live endpoint. Verify the account has crypto trading enabled and
   is identity-verified.
4. Reduce `position_pct` in your champion spec to a level you can afford to
   lose entirely.

**To enable:**

```bash
# In .env (or export in your shell):
AGENT_LIVE_MODE=true
ALPACA_KEY=<your-live-key>        # NOT your paper key
ALPACA_SECRET=<your-live-secret>

python -m trading_agent check-keys    # confirm the live account is reachable
python -m trading_agent paper --once   # dry-fire one tick; watch the 5-second warning
```

`paper --once` (and the full scheduler) prints a 5-second abort banner
whenever live mode is active. `AlpacaCryptoBroker` also logs a `WARNING`
on every construction so it lands in your log sink.

**The guardrails are unchanged.** `HARD_DD_KILL` (-10%), `MAX_DAILY_LOSS`
(-3%), `MAX_TRADES_PER_DAY` (8), and `MAX_POSITION_PCT` (25%) all apply
exactly as in paper mode — they are enforced before every order regardless
of endpoint.

**To revert at any time:** set `AGENT_LIVE_MODE=false` (or remove it). The
agent switches back to paper on the next restart.
