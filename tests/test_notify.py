from __future__ import annotations

import json

from trading_agent import config, notify as notify_mod


def _settings(monkeypatch, **env) -> config.Settings:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return config.load()


def test_notify_none_is_noop(monkeypatch):
    s = _settings(monkeypatch, AGENT_NOTIFY="none")

    def _boom(*a, **k):
        raise AssertionError("urlopen should not be called when notify is off")

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", _boom)
    notify_mod.notify("entry_fill", {"symbol": "SPY"}, s)  # must not raise


def test_notify_webhook_posts_payload(monkeypatch):
    s = _settings(
        monkeypatch,
        AGENT_NOTIFY="webhook",
        AGENT_NOTIFY_WEBHOOK_URL="https://hooks.example.com/abc",
    )
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _Resp()

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", _fake_urlopen)
    notify_mod.notify("promotion", {"version": "v-9"}, s)
    assert captured["url"] == "https://hooks.example.com/abc"
    assert captured["body"]["event"] == "promotion"
    assert "v-9" in captured["body"]["text"]  # Slack-style field present


def test_notify_swallows_errors(monkeypatch):
    s = _settings(
        monkeypatch,
        AGENT_NOTIFY="webhook",
        AGENT_NOTIFY_WEBHOOK_URL="https://hooks.example.com/abc",
    )

    def _raise(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(notify_mod.urllib.request, "urlopen", _raise)
    notify_mod.notify("kill_switch", {"dd_pct": -10.0}, s)  # must not raise
