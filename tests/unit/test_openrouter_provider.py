"""Native protocol, immutable continuation, failure accounting and routing checks."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from celltrace.agents.roles import INVESTIGATOR
from celltrace.providers.base import FailureKind, ProviderFailure, ProviderRequest
from celltrace.providers.openrouter_adapter import OpenRouterAdapter, provider_usage
from celltrace.providers.profiles import PRIMARY_MODEL_ID
from celltrace.providers.rate_card import RateCard


def provider_request(**changes: Any) -> ProviderRequest:
    request = ProviderRequest(
        role="investigator",
        system_prompt="Investigate measurements",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Check context"}]}],
        tools=INVESTIGATOR.tool_definitions(),
        maximum_output_tokens=2048,
        timeout_seconds=45,
        effort="high",
    )
    return replace(request, **changes)


def completion_payload(arguments: str = "{}", **changes: Any) -> dict[str, Any]:
    payload = {
        "id": "generation_1",
        "model": PRIMARY_MODEL_ID,
        "provider": "DeepSeek",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "data": "opaque_signature_123", "index": 0}
                    ],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_session_context",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 200},
            "completion_tokens_details": {"reasoning_tokens": 80},
            "cost": 0.0004,
        },
    }
    payload.update(changes)
    return payload


def mocked_adapter(payload: dict[str, Any], status: int = 200) -> OpenRouterAdapter:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=payload),
        )
    )
    return OpenRouterAdapter(PRIMARY_MODEL_ID, client=client)


def test_native_request_has_exact_profile_no_fallback_and_only_client_tools() -> None:
    body = mocked_adapter(completion_payload()).request_body(provider_request())
    assert body["model"] == PRIMARY_MODEL_ID
    assert body["provider"] == {
        "only": ["deepseek"],
        "order": ["deepseek"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert body["reasoning"] == {"effort": "high", "exclude": False}
    assert body["max_tokens"] == 2048 and body["stream"] is False
    assert "response_format" not in body and "plugins" not in body
    assert all(tool["type"] == "function" for tool in body["tools"])
    assert body["tools"][-1]["function"]["name"] == "submit_findings"
    with pytest.raises(ValueError, match="effort high"):
        mocked_adapter(completion_payload()).request_body(provider_request(effort="low"))


def test_tool_result_continues_with_identical_opaque_provider_fields() -> None:
    payload = completion_payload()
    adapter = mocked_adapter(payload)
    response = adapter.complete(provider_request())
    messages = [
        *provider_request().messages,
        {"role": "assistant", "content": response.assistant_content},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "{}"}],
        },
    ]
    body = adapter.request_body(provider_request(messages=messages))
    assert body["messages"][-2] == payload["choices"][0]["message"]
    assert body["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "{}"}
    reviewer_body = adapter.request_body(provider_request(role="reviewer"))
    assert "opaque_signature_123" not in json.dumps(reviewer_body)


def test_mixed_tool_result_and_budget_text_preserve_wire_order_and_opaque_reasoning() -> None:
    adapter = mocked_adapter(completion_payload())
    response = adapter.complete(provider_request())
    blocks = [
        {"type": "tool_result", "tool_use_id": "call_1", "content": '{"evidence":true}'},
        {"type": "text", "text": "Current ledger budget: 12 requests left"},
    ]
    request = provider_request(
        messages=[
            *provider_request().messages,
            {"role": "assistant", "content": response.assistant_content},
            {"role": "user", "content": blocks},
        ]
    )
    body = adapter.request_body(request)
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "tool", "user"]
    assert body["messages"][2] == completion_payload()["choices"][0]["message"]
    assert body["messages"][3]["tool_call_id"] == "call_1"
    assert body["messages"][4]["content"] == blocks[1]["text"]
    assert request.messages[-1]["content"] == blocks


@pytest.mark.parametrize("fault", ["interrupted", "missing", "wrong_id", "duplicate"])
def test_invalid_tool_continuation_is_refused_before_http(fault: str) -> None:
    adapter = mocked_adapter(completion_payload())
    response = adapter.complete(provider_request())
    reply = {"type": "tool_result", "tool_use_id": "call_1", "content": "{}"}
    blocks = [reply]
    if fault == "interrupted":
        blocks.insert(0, {"type": "text", "text": "incorrectly before the required reply"})
    elif fault == "missing":
        blocks = []
    elif fault == "wrong_id":
        reply["tool_use_id"] = "different_call"
    else:
        blocks.append(reply)
    adapter.client = httpx.Client(
        transport=httpx.MockTransport(lambda _: pytest.fail("invalid request reached HTTP"))
    )
    with pytest.raises(ValueError, match="native tool"):
        adapter.complete(
            provider_request(
                messages=[
                    {"role": "assistant", "content": response.assistant_content},
                    {"role": "user", "content": blocks},
                ]
            )
        )
    assert response.effective_settings["effective_effort"] is None


@pytest.mark.parametrize("arguments", ['{"broken":', "[]", '{"number": NaN}', "null"])
def test_malformed_arguments_remain_a_typed_validation_failure(arguments: str) -> None:
    response = mocked_adapter(completion_payload(arguments)).complete(provider_request())
    assert "invalid_provider_arguments" in response.tool_calls[0].arguments
    assert (
        response.assistant_content[0]["message"]["tool_calls"][0]["function"]["arguments"]
        == arguments
    )


@pytest.mark.parametrize(
    "status, kind",
    [
        (400, FailureKind.permanent),
        (401, FailureKind.permanent),
        (429, FailureKind.transient),
        (500, FailureKind.uncertain),
        (503, FailureKind.uncertain),
    ],
)
def test_errors_classify_unknown_upstream_execution_conservatively(
    status: int, kind: FailureKind
) -> None:
    with pytest.raises(ProviderFailure) as raised:
        mocked_adapter({"error": {"message": "secret text must not enter logs"}}, status).complete(
            provider_request()
        )
    assert raised.value.kind == kind
    assert "secret text" not in str(raised.value)
    assert str(raised.value).endswith(": unclassified")


@pytest.mark.parametrize(
    "message, category",
    [
        ("No endpoints found matching your data policy (secret text)", "data_policy"),
        ("No endpoints found that can handle the requested parameters.", "parameters"),
        ("No endpoints found for deepseek/deepseek-v4.1-flash.", "no_endpoint_found"),
    ],
)
def test_routing_rejections_keep_a_fixed_category_without_raw_text(
    message: str, category: str
) -> None:
    with pytest.raises(ProviderFailure) as raised:
        mocked_adapter({"error": {"code": 404, "message": message}}, 404).complete(
            provider_request()
        )
    assert raised.value.kind == FailureKind.permanent
    assert category in str(raised.value)
    assert "secret text" not in str(raised.value) and "deepseek" not in str(raised.value)


def test_timeout_is_uncertain_and_client_does_not_retry() -> None:
    attempts = []

    def timeout(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ReadTimeout("timeout", request=request)

    adapter = OpenRouterAdapter(
        PRIMARY_MODEL_ID, httpx.Client(transport=httpx.MockTransport(timeout))
    )
    with pytest.raises(ProviderFailure) as raised:
        adapter.complete(provider_request())
    assert raised.value.kind == FailureKind.uncertain
    assert len(attempts) == 1


def test_cache_and_reasoning_are_subsets_and_billed_cost_is_not_added_twice() -> None:
    usage = provider_usage(completion_payload()["usage"])
    assert usage.input_tokens == 800 and usage.cache_read_input_tokens == 200
    assert usage.output_tokens == 100 and usage.reasoning_tokens == 80
    rate_card = RateCard(
        model_id=PRIMARY_MODEL_ID,
        currency="USD",
        unit="per_million_tokens",
        input=0.3,
        output=1.2,
        cache_write_5m=0,
        cache_write_1h=0,
        cache_read=0.006,
        tool_use_system_prompt_tokens_auto=0,
        source_url="https://openrouter.ai",
        checked_date="2026-10-08",
        cache_treatment="subset",
        extra_charges="none",
    )
    assert rate_card.cost(usage) == 0.0004
    assert provider_usage(None).usage_known is False
    assert provider_usage({"prompt_tokens": 2, "completion_tokens": -1}).usage_known is False
    assert (
        provider_usage(
            {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 3},
            }
        ).usage_known
        is False
    )


def test_model_and_provider_mismatch_are_visible_after_usage_is_preserved() -> None:
    response = mocked_adapter(completion_payload(provider="Another Provider")).complete(
        provider_request()
    )
    assert response.usage.usage_known
    assert response.effective_settings["settings_errors"]


@pytest.mark.parametrize("reasoning_content", [None, "opaque protocol content"])
def test_reasoning_visibility_does_not_certify_effective_effort(
    reasoning_content: str | None,
) -> None:
    payload = completion_payload()
    original = payload["choices"][0]["message"]
    original.pop("reasoning_details")
    original["reasoning_content"] = reasoning_content
    payload["usage"]["completion_tokens_details"]["reasoning_tokens"] = 0
    adapter = mocked_adapter(payload)
    response = adapter.complete(provider_request())
    assert response.effective_settings["settings_errors"] == []
    assert response.effective_settings["reasoning_observed"] == bool(reasoning_content)
    assert response.effective_settings["effective_effort"] is None
    assert response.usage.usage_known
    continued = adapter.request_body(
        provider_request(
            messages=[
                {"role": "assistant", "content": response.assistant_content},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "{}"}],
                },
            ]
        )
    )
    assert continued["messages"][-2] == original
