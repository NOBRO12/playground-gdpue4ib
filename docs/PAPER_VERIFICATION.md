# Paper-keys verification checklist

A few behaviors can only be confirmed against Alpaca's real (paper) endpoint —
the offline test suite mocks the trading client, so it proves the *request
shape* we send but not that Alpaca *accepts* it. Run this once against paper
keys before trusting the live loop with real funds.

Everything here uses **paper** keys and `AGENT_LIVE_MODE=false`. No real money
moves. Never paste keys into a prompt or commit them — put them in `.env`.

## Setup

```bash
cp .env.example .env
# Fill ALPACA_KEY / ALPACA_SECRET with PAPER keys, ANTHROPIC_API_KEY.
# Keep AGENT_ASSET_CLASS=stock and AGENT_LIVE_MODE=false.

python -m trading_agent init-db
python -m trading_agent check-keys          # must pass before continuing
```

## 1. Bracket-order TIF / validation accepted by Alpaca

The offline test (`tests/test_stock_broker.py::test_bracket_builds_correct_request_shape`)
pins that we send `OrderClass.BRACKET` + `TimeInForce.GTC` + both legs. What it
*can't* check is that Alpaca's server accepts that combination for an equity
market entry. Confirm during regular market hours:

```bash
python -m trading_agent paper --once
```

- [ ] Tick runs without an Alpaca 422 / "invalid time_in_force" /
      "bracket not allowed" rejection.
- [ ] In the Alpaca paper dashboard → **Orders**, the entry shows as a
      **bracket** parent with a **stop-loss** and a **take-profit** child.
- [ ] Both child legs are **resting / open** (not rejected), so they can fire
      intraday between daily ticks.

If Alpaca rejects the bracket TIF, the fix is in
[broker.py](../trading_agent/execution/broker.py) `AlpacaStockBroker.submit_bracket`
(adjust `time_in_force`) — the offline test will then need its expected value
updated to match.

## 2. Resting legs actually fire between ticks

- [ ] Leave the position open across a session where price crosses either leg
      (or move the levels close to market to force it).
- [ ] Confirm Alpaca fills the stop or take-profit **on its own**, with no tick
      from this agent in between.
- [ ] On the next `paper --once`, the loop detects the broker-side exit and
      `status` shows the trade closed with the right `reason_exit`.

## 3. Startup reconciliation against a real position

Offline this is covered by `tests/test_reconcile.py` against the MockBroker and
`tests/test_stock_broker.py::test_positions_parses_held_qty_for_reconciliation`.
Confirm it reconciles against a *real* Alpaca position:

```bash
# With a position already open from step 1, delete local state to simulate a
# crash/restart that lost the live_state file:
rm -f data/live_state.json
python -m trading_agent paper --once
python -m trading_agent status
```

- [ ] The loop seeds held qty from `broker.positions()` rather than assuming
      flat — `status` reflects the real open position, not a fresh start.
- [ ] It does **not** submit a duplicate entry for the bar it already traded
      (deterministic `client_order_id` is rejected by Alpaca as a duplicate).

## 4. Webhook notifications (if enabled)

```bash
# In .env:
AGENT_NOTIFY=webhook
AGENT_NOTIFY_WEBHOOK_URL=<your Slack/Discord incoming-webhook URL>
python -m trading_agent paper --once
```

- [ ] A message lands in the channel on a fill / guardrail block / kill-switch
      trip. Delivery failures are swallowed (verified offline in
      `tests/test_notify.py`) so they can never break the trading loop.

---

When all boxes are checked, the live-only gaps noted in the PR are closed. The
guardrails and the OOS promotion gate are unchanged by any of this.
