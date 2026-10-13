"""Deterministic tool failure injection for tests, the demo and failure cases."""

from __future__ import annotations

from typing import Any

from celltrace.contracts.common import ErrorCode
from celltrace.contracts.evidence import ToolError
from celltrace.tools.dispatch import ToolDispatcher, ToolOutcome


class FaultInjectingDispatcher(ToolDispatcher):
    """Fails selected tools with a fixed error code, every time they are called."""

    failing_tools: frozenset[str] = frozenset()
    failure_code: ErrorCode = ErrorCode.store_unavailable

    def dispatch(self, tool_name: str, raw_arguments: dict[str, Any]) -> ToolOutcome:
        if tool_name in self.failing_tools:
            return ToolOutcome(
                tool_name=tool_name,
                error=ToolError(
                    code=self.failure_code,
                    message="injected measurement store failure",
                    retryable=self.failure_code in (ErrorCode.store_unavailable, ErrorCode.timeout),
                    details={"injected": True},
                ),
                physical_query=True,
            )
        return super().dispatch(tool_name, raw_arguments)


def failing_dispatcher_factory(
    tools: set[str], code: ErrorCode = ErrorCode.store_unavailable
) -> type[FaultInjectingDispatcher]:
    return type(
        "ConfiguredFaultInjectingDispatcher",
        (FaultInjectingDispatcher,),
        {"failing_tools": frozenset(tools), "failure_code": code},
    )
