"""Scripted adapter labels and failures, tool schemas and rate card arithmetic."""

from __future__ import annotations

import pytest

from celltrace.agents.roles import INVESTIGATOR, REVIEWER, inline_schema
from celltrace.contracts.common import ExecutionMode
from celltrace.contracts.findings import InvestigationFindings
from celltrace.providers.base import FailureKind, ProviderFailure, ProviderRequest
from celltrace.providers.rate_card import RateCard
from celltrace.providers.scripted import ScriptedModelAdapter, ScriptedTurn


def request() -> ProviderRequest:
    return ProviderRequest(
        role="investigator",
        system_prompt="system",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=INVESTIGATOR.tool_definitions(),
        maximum_output_tokens=256,
        timeout_seconds=5,
        effort="low",
    )


def test_scripted_adapter_is_labeled_and_schedules_failures() -> None:
    class Policy:
        def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
            return ScriptedTurn("check", "get_session_context", {})

    adapter = ScriptedModelAdapter({"investigator": Policy()}, failures={2: FailureKind.transient})
    response = adapter.complete(request())
    assert response.execution_mode == ExecutionMode.scripted_fixture
    assert response.tool_calls[0].tool_use_id == "toolu_investigator_001"
    with pytest.raises(ProviderFailure):
        adapter.complete(request())


def test_tool_schemas_are_plain_json_schema() -> None:
    for spec in (INVESTIGATOR, REVIEWER):
        text = str(spec.tool_definitions())
        assert "$ref" not in text and "discriminator" not in text and "oneOf" not in text
    schema = inline_schema(InvestigationFindings)
    assert schema["additionalProperties"] is False


def test_rate_card_cost() -> None:
    card = RateCard(
        model_id="m",
        currency="USD",
        unit="per_million_tokens",
        input=4,
        output=20,
        cache_write_5m=5,
        cache_write_1h=8,
        cache_read=0.2,
        tool_use_system_prompt_tokens_auto=286,
        source_url="u",
        checked_date="2026-10-08",
        cache_treatment="t",
        extra_charges="e",
    )
    from celltrace.providers.base import ProviderUsage

    usage = ProviderUsage(usage_known=True, input_tokens=1_000_000, output_tokens=100_000)
    assert card.cost(usage) == pytest.approx(4 + 2)
