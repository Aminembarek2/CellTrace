"""Provider boundary shared by the live and scripted adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from celltrace.contracts.common import ExecutionMode


@dataclass(frozen=True)
class ProviderRequest:
    role: str
    system_prompt: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    maximum_output_tokens: int
    timeout_seconds: float
    effort: str | None


@dataclass(frozen=True)
class ToolCall:
    tool_use_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ProviderUsage:
    usage_known: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    billed_usd: float | None = None


@dataclass(frozen=True)
class ProviderResponse:
    assistant_content: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    text: str
    stop_reason: str
    usage: ProviderUsage
    reported_model: str | None
    request_id: str | None
    execution_mode: ExecutionMode
    effective_settings: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "assistant_content": self.assistant_content,
            "tool_calls": [
                {"tool_use_id": call.tool_use_id, "name": call.name, "arguments": call.arguments}
                for call in self.tool_calls
            ],
            "text": self.text,
            "stop_reason": self.stop_reason,
            "usage": self.usage.__dict__,
            "reported_model": self.reported_model,
            "request_id": self.request_id,
            "execution_mode": self.execution_mode.value,
            "effective_settings": self.effective_settings,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ProviderResponse:
        return cls(
            assistant_content=record["assistant_content"],
            tool_calls=[ToolCall(**call) for call in record["tool_calls"]],
            text=record["text"],
            stop_reason=record["stop_reason"],
            usage=ProviderUsage(**record["usage"]),
            reported_model=record["reported_model"],
            request_id=record["request_id"],
            execution_mode=ExecutionMode(record["execution_mode"]),
            effective_settings=record.get("effective_settings", {}),
        )


# Estimation method of the byte based prompt bound, calibrated by the workflow.
BYTE_ESTIMATE_METHOD = "utf8_two_byte_estimate_with_margin"


class FailureKind(StrEnum):
    transient = "transient"
    permanent = "permanent"
    uncertain = "uncertain"


class ProviderFailure(Exception):
    """A provider attempt that produced no usable response.

    ``transient`` means the provider answered with a retryable error, so the
    request is known to have failed. ``uncertain`` means we cannot tell
    whether it executed (timeout or connection loss), so it is never
    resubmitted automatically.
    """

    def __init__(
        self,
        kind: FailureKind,
        message: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.request_id = request_id

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "message": str(self),
            "status_code": self.status_code,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class TokenEstimate:
    input_tokens: int
    method: str


class ModelAdapter(Protocol):
    # Read only members, so an adapter with a fixed str model id or class level mode conforms.
    @property
    def execution_mode(self) -> ExecutionMode: ...

    @property
    def model_id(self) -> str | None: ...

    def estimate_input_tokens(self, request: ProviderRequest) -> TokenEstimate: ...

    def complete(self, request: ProviderRequest) -> ProviderResponse: ...

    def describe(self) -> dict[str, Any]: ...
