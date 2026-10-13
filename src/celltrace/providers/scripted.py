"""Offline scripted adapter for orchestration tests and the fixture demo.

It replays decisions from a policy object through the same protocol as the
live adapter. Its responses always carry ``execution_mode=scripted_fixture``
and must never be reported as live model behavior. Token counts are
character estimates and the monetary cost is zero because no provider runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from celltrace.canonical import canonical_json
from celltrace.contracts.common import ExecutionMode
from celltrace.providers.base import (
    FailureKind,
    ProviderFailure,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    TokenEstimate,
    ToolCall,
)

CHARACTERS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class ScriptedTurn:
    text: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


class ScriptedPolicy(Protocol):
    def next_turn(self, request: ProviderRequest) -> ScriptedTurn: ...


def assistant_turn_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every tool result payload in a transcript, oldest first, as parsed JSON."""
    import json

    results = []
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") == "tool_result":
                content = block.get("content")
                text = content if isinstance(content, str) else content[0]["text"]
                results.append(json.loads(text))
    return results


class ScriptedModelAdapter:
    execution_mode = ExecutionMode.scripted_fixture

    def __init__(
        self,
        policies: dict[str, ScriptedPolicy],
        failures: dict[int, FailureKind] | None = None,
    ):
        self.model_id: str | None = "scripted-fixture"
        self.policies = policies
        self.failures = failures or {}
        self.attempts = 0

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "scripted",
            "execution_mode": self.execution_mode.value,
            "policies": {role: type(policy).__name__ for role, policy in self.policies.items()},
            "scheduled_failures": {str(key): value.value for key, value in self.failures.items()},
        }

    def estimate_input_tokens(self, request: ProviderRequest) -> TokenEstimate:
        characters = len(canonical_json([request.system_prompt, request.messages, request.tools]))
        return TokenEstimate(
            -(-characters // CHARACTERS_PER_TOKEN_ESTIMATE), "scripted_character_estimate"
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.attempts += 1
        scheduled = self.failures.get(self.attempts)
        if scheduled is not None:
            raise ProviderFailure(scheduled, f"scheduled {scheduled.value} failure", 529)
        turn = self.policies[request.role].next_turn(request)
        content: list[dict[str, Any]] = [{"type": "text", "text": turn.text}]
        tool_calls = []
        if turn.tool_name is not None:
            tool_use_id = f"toolu_{request.role}_{assistant_turn_count(request.messages) + 1:03d}"
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": turn.tool_name,
                    "input": turn.arguments,
                }
            )
            tool_calls.append(ToolCall(tool_use_id, turn.tool_name, turn.arguments))
        output_characters = len(canonical_json(content))
        return ProviderResponse(
            assistant_content=content,
            tool_calls=tool_calls,
            text=turn.text,
            stop_reason="tool_use" if tool_calls else "end_turn",
            usage=ProviderUsage(
                usage_known=True,
                input_tokens=self.estimate_input_tokens(request).input_tokens,
                output_tokens=-(-output_characters // CHARACTERS_PER_TOKEN_ESTIMATE),
            ),
            reported_model="scripted-fixture",
            request_id=None,
            execution_mode=ExecutionMode.scripted_fixture,
            effective_settings={"note": "scripted fixture, no model inference"},
        )
