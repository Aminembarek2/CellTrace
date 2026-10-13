"""Synchronous OpenRouter chat completions with native tools and no client retries.

The workflow keeps its existing content blocks. Conversion to the provider
protocol happens here. The original assistant message is retained as opaque
role local data so reasoning details and signatures survive continuation.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

import httpx

from celltrace.canonical import canonical_json
from celltrace.contracts.common import ExecutionMode
from celltrace.providers.base import (
    BYTE_ESTIMATE_METHOD,
    FailureKind,
    ProviderFailure,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    TokenEstimate,
    ToolCall,
)
from celltrace.providers.profiles import model_profile

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
ADAPTER_VERSION = "openrouter_v3"

# Provider error text can quote request content, so it is never persisted.
# Only one of these fixed categories enters the failure message.
ERROR_CATEGORIES = (
    ("data policy", "no_endpoint_matching_data_policy"),
    ("requested parameters", "no_endpoint_supporting_requested_parameters"),
    ("allowed providers", "no_allowed_provider"),
    ("guardrail", "account_guardrail"),
    ("no endpoints found", "no_endpoint_found"),
    ("not a valid model", "model_not_found"),
    ("credits", "insufficient_credits"),
)


def error_category(response: httpx.Response) -> str:
    try:
        error = response.json()["error"]
        message = str(error["message"]).lower()
    except (ValueError, TypeError, KeyError):
        return "unreadable_error_body"
    for phrase, category in ERROR_CATEGORIES:
        if phrase in message:
            return category
    return "unclassified"


def chat_messages(request: ProviderRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": request.system_prompt}]
    for message in request.messages:
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
            continue
        originals = [block["message"] for block in content if block.get("type") == "openrouter"]
        if originals:
            if message["role"] != "assistant" or len(originals) != 1:
                raise ValueError("opaque provider data must belong to one assistant message")
            messages.append(copy.deepcopy(originals[0]))
            continue
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_result":
                if text_parts:
                    messages.append({"role": message["role"], "content": "\n".join(text_parts)})
                    text_parts = []
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    }
                )
            else:
                raise ValueError("cannot continue another provider's assistant protocol")
        if text_parts:
            messages.append({"role": message["role"], "content": "\n".join(text_parts)})
    _validate_tool_replies(messages)
    return messages


def _validate_tool_replies(messages: list[dict[str, Any]]) -> None:
    """Refuse unmatched or interrupted native replies before reservation or HTTP."""
    pending: set[str] = set()
    for message in messages:
        if pending:
            if message["role"] != "tool" or message.get("tool_call_id") not in pending:
                raise ValueError("native tool calls require matching contiguous tool replies")
            pending.remove(message["tool_call_id"])
        elif message["role"] == "tool":
            raise ValueError("native tool reply has no pending call")
        if message["role"] == "assistant":
            calls = message.get("tool_calls") or []
            pending = {call["id"] for call in calls}
            if len(pending) != len(calls) or any(not isinstance(p, str) or not p for p in pending):
                raise ValueError("native tool call identifiers must be unique nonempty strings")
    if pending:
        raise ValueError("native tool calls require matching contiguous tool replies")


def function_tools(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools = []
    for definition in definitions:
        function = {
            "name": definition["name"],
            "description": definition["description"],
            "parameters": definition["input_schema"],
        }
        if definition.get("strict"):
            function["strict"] = True
        tools.append({"type": "function", "function": function})
    return tools


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _arguments(value: Any) -> dict[str, Any]:
    try:
        if not isinstance(value, str):
            raise ValueError("tool arguments must be a JSON string")
        parsed = json.loads(value, parse_constant=_reject_constant)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be an object")
        canonical_json(parsed)
        return parsed
    except (ValueError, TypeError):
        # This deliberately fails typed tool or terminal validation. The raw
        # arguments remain in the opaque message for the correction round.
        return {"invalid_provider_arguments": "expected a finite JSON object"}


def _token_count(usage: dict[str, Any], name: str) -> int:
    value = usage[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid usage count {name}")
    return value


def provider_usage(payload: Any) -> ProviderUsage:
    if not isinstance(payload, dict):
        return ProviderUsage(usage_known=False)
    try:
        prompt_tokens = _token_count(payload, "prompt_tokens")
        completion_tokens = _token_count(payload, "completion_tokens")
        prompt_details = payload.get("prompt_tokens_details") or {}
        completion_details = payload.get("completion_tokens_details") or {}
        cached_tokens = _token_count(
            {"cached_tokens": prompt_details.get("cached_tokens", 0)}, "cached_tokens"
        )
        reasoning_tokens = _token_count(
            {"reasoning_tokens": completion_details.get("reasoning_tokens", 0)}, "reasoning_tokens"
        )
        if cached_tokens > prompt_tokens or reasoning_tokens > completion_tokens:
            raise ValueError("usage subsets cannot exceed their totals")
        billed_usd = payload.get("cost")
        if billed_usd is not None:
            billed_usd = float(billed_usd)
            if not math.isfinite(billed_usd) or billed_usd < 0:
                raise ValueError("invalid billed cost")
        return ProviderUsage(
            usage_known=True,
            input_tokens=prompt_tokens - cached_tokens,
            output_tokens=completion_tokens,
            cache_read_input_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            billed_usd=billed_usd,
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return ProviderUsage(usage_known=False)


class OpenRouterAdapter:
    execution_mode = ExecutionMode.live

    def __init__(self, model_id: str, client: httpx.Client | None = None):
        self.profile = model_profile(model_id)
        self.model_id = model_id
        self.client = client

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "adapter_version": ADAPTER_VERSION,
            "model_id": self.model_id,
            "routing": self.profile.routing(),
            "client_retries": 0,
            "reasoning": {"effort": "high", "exclude": False},
            "sampling_settings_applied": False,
            "terminal_protocol": "native submission tool",
        }

    def request_body(self, request: ProviderRequest) -> dict[str, Any]:
        if request.effort != "high":
            raise ValueError("OpenRouter profiles require reasoning effort high")
        return {
            "model": self.model_id,
            "messages": chat_messages(request),
            "tools": function_tools(request.tools),
            "tool_choice": "auto",
            "provider": self.profile.routing(),
            "reasoning": {"effort": "high", "exclude": False},
            "max_tokens": request.maximum_output_tokens,
            "stream": False,
        }

    def estimate_input_tokens(self, request: ProviderRequest) -> TokenEstimate:
        # Two UTF8 bytes per token is an engineering estimate, not a certified
        # bound. The estimate includes repeated history and all schemas.
        byte_count = len(canonical_json(self.request_body(request)).encode("utf-8"))
        return TokenEstimate((byte_count + 1) // 2 + 256, BYTE_ESTIMATE_METHOD)

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        body = self.request_body(request)
        headers = {"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"}
        try:
            if self.client is None:
                with httpx.Client(transport=httpx.HTTPTransport(retries=0)) as client:
                    response = client.post(
                        CHAT_COMPLETIONS_URL,
                        json=body,
                        headers=headers,
                        timeout=request.timeout_seconds,
                    )
            else:
                response = self.client.post(
                    CHAT_COMPLETIONS_URL,
                    json=body,
                    headers=headers,
                    timeout=request.timeout_seconds,
                )
        except (httpx.TimeoutException, httpx.TransportError) as failure:
            raise ProviderFailure(
                FailureKind.uncertain,
                f"OpenRouter transport outcome unknown: {type(failure).__name__}",
            ) from failure
        if response.status_code >= 400:
            failure_kind = (
                FailureKind.transient
                if response.status_code == 429
                else (
                    FailureKind.uncertain if response.status_code >= 500 else FailureKind.permanent
                )
            )
            raise ProviderFailure(
                failure_kind,
                f"OpenRouter HTTP {response.status_code}: {error_category(response)}",
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            )
        try:
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("error"):
                raise ValueError("provider returned no completion")
            return self._normalize(payload, request)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as failure:
            raise ProviderFailure(
                FailureKind.uncertain,
                "OpenRouter response unusable with unknown execution usage",
                request_id=response.headers.get("x-request-id"),
            ) from failure

    def _normalize(self, payload: dict[str, Any], request: ProviderRequest) -> ProviderResponse:
        choices = payload["choices"]
        if len(choices) != 1:
            raise ValueError("expected one completion")
        choice = choices[0]
        original = choice["message"]
        if original["role"] != "assistant":
            raise ValueError("expected assistant message")
        tool_calls = []
        blocks = [{"type": "openrouter", "message": copy.deepcopy(original)}]
        identifiers: set[str] = set()
        for raw_call in original.get("tool_calls") or []:
            identifier = raw_call["id"]
            name = raw_call["function"]["name"]
            if raw_call["type"] != "function" or not isinstance(identifier, str) or not identifier:
                raise ValueError("invalid function call identity")
            if identifier in identifiers or not isinstance(name, str):
                raise ValueError("duplicate function identity or invalid name")
            identifiers.add(identifier)
            arguments = _arguments(raw_call["function"]["arguments"])
            tool_calls.append(ToolCall(identifier, name, arguments))
            blocks.append({"type": "tool_use", "id": identifier, "name": name, "input": arguments})
        stop_reason = str(choice.get("finish_reason") or "unknown")
        if original.get("refusal"):
            stop_reason = "refusal"
        settings_errors = []
        if payload.get("model") != self.model_id:
            settings_errors.append("response model does not match requested model")
        if payload.get("provider") != self.profile.provider_name:
            settings_errors.append(
                "response provider missing or different from the restricted provider"
            )
        reasoning_details = (
            original.get("reasoning_details")
            or original.get("reasoning")
            or original.get("reasoning_content")
        )
        usage = provider_usage(payload.get("usage"))
        return ProviderResponse(
            assistant_content=blocks,
            tool_calls=tool_calls,
            text=original.get("content") or "",
            stop_reason=stop_reason,
            usage=usage,
            reported_model=payload.get("model"),
            request_id=payload.get("id"),
            execution_mode=ExecutionMode.live,
            effective_settings={
                "requested_model": self.model_id,
                "reported_model": payload.get("model"),
                "reported_provider": payload.get("provider"),
                "requested_effort": request.effort,
                "effective_effort": None,
                "effort_verification": "request sent, effective effort not echoed",
                "reasoning_observed": bool(reasoning_details or usage.reasoning_tokens),
                "maximum_output_tokens": request.maximum_output_tokens,
                "settings_errors": settings_errors,
            },
        )
