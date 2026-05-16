"""Claude API call to generate a challenger spec.

Hard constraints:
    - The model is forced to call a tool whose ``input_schema`` mirrors
      :class:`StrategySpec`. Anything outside the schema is rejected by the SDK.
    - The trade summary handed to Claude is restricted to the IS window; OOS
      bars never appear in this module's call graph.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .. import config
from ..strategy.spec import StrategySpec, claude_tool_input_schema

SYSTEM = """You are a quantitative researcher improving a crypto trading strategy.
You are given the current champion spec, summary stats by regime, and recent
trade outcomes from an in-sample window. Propose ONE challenger spec that may
improve risk-adjusted returns. Stay strictly within the schema. Only change
parameters/filters/risk; keep type and symbol unchanged unless clearly warranted.
Bias toward small, defensible changes."""


@dataclass(frozen=True)
class ProposalResult:
    spec: StrategySpec
    prompt_hash: str
    response_json: str


def _prompt_body(champion: StrategySpec, trade_summary: dict[str, Any]) -> str:
    return (
        "CHAMPION:\n"
        + json.dumps(champion.model_dump(), indent=2, sort_keys=True)
        + "\n\nIS_TRADE_SUMMARY:\n"
        + json.dumps(trade_summary, indent=2, sort_keys=True)
        + "\n\nReturn a single tool call with the challenger spec."
    )


def _hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def propose(champion: StrategySpec, trade_summary: dict[str, Any]) -> ProposalResult:
    import anthropic

    s = config.load()
    if not s.anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set; proposer requires it at runtime")
    client = anthropic.Anthropic(api_key=s.anthropic_key)
    tool = {
        "name": "submit_challenger_spec",
        "description": "Submit a single challenger StrategySpec proposal.",
        "input_schema": claude_tool_input_schema(),
    }
    body = _prompt_body(champion, trade_summary)
    msg = client.messages.create(
        model=s.proposer_model,
        max_tokens=2048,
        system=SYSTEM,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_challenger_spec"},
        messages=[{"role": "user", "content": body}],
    )
    tool_block = next((b for b in msg.content if getattr(b, "type", None) == "tool_use"), None)
    if tool_block is None:
        raise RuntimeError("Claude did not return a tool_use block")
    raw = dict(tool_block.input)
    raw["parent"] = champion.version
    raw["version"] = "v-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    spec = StrategySpec.model_validate(raw)
    response_json = json.dumps(raw, sort_keys=True)
    return ProposalResult(
        spec=spec,
        prompt_hash=_hash(SYSTEM + body),
        response_json=response_json,
    )


def persist(conn: sqlite3.Connection, parent_version: str, result: ProposalResult) -> int:
    cur = conn.execute(
        """
        INSERT INTO proposals (ts, parent_version, prompt_hash, response_json, decision, reject_reason)
        VALUES (?, ?, ?, ?, NULL, NULL)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            parent_version,
            result.prompt_hash,
            result.response_json,
        ),
    )
    return int(cur.lastrowid)
