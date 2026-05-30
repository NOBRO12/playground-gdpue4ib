"""Best-effort operator notifications.

Configured via ``AGENT_NOTIFY`` (``none`` | ``webhook``) and
``AGENT_NOTIFY_WEBHOOK_URL`` (Slack/Discord-compatible). A notification failure
must never break the trading loop, so every send is wrapped and only logged.
Uses stdlib urllib — no extra dependency.
"""
from __future__ import annotations

import json
import logging
import urllib.request

from . import config

log = logging.getLogger(__name__)


def notify(event: str, detail: dict, settings: config.Settings | None = None) -> None:
    s = settings or config.load()
    if s.notify != "webhook" or not s.notify_webhook_url:
        return
    text = f"[trading-agent] {event}: {json.dumps(detail, default=str)}"
    # "text" (Slack) and "content" (Discord) so either platform renders it.
    body = json.dumps({"text": text, "content": text, "event": event, "detail": detail}, default=str).encode()
    try:
        req = urllib.request.Request(
            s.notify_webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)  # noqa: S310 — operator-configured URL
    except Exception as exc:  # noqa: BLE001 — never let notify break trading
        log.warning("notify failed (%s): %s", event, exc)
