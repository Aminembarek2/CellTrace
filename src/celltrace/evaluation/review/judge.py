"""One paid judge call per item, charged to the durable run ledger.

A completed reply may receive one correction request, charged to the same run. A reply without
a tool call is never resent, because the provider rejects that continuation, so the next attempt
repeats the original request. An attempt with unknown outcome is never dispatched again.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from celltrace.canonical import canonical_json
from celltrace.evaluation.review.persistence import write_json_atomic
from celltrace.evaluation.review.response_contract import (
    REQUEST_RUBRIC,
    decode_assessment,
    tool_schema,
)
from celltrace.evaluation.review.rubric import TOOL_NAME, Assessment
from celltrace.providers.base import (
    FailureKind,
    ModelAdapter,
    ProviderFailure,
    ProviderRequest,
    ProviderResponse,
)
from celltrace.providers.rate_card import RateCard
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    BudgetExhausted,
    Reservation,
    RunLimits,
    RunStore,
)

ROLE = "semantic_judge"
ASSESSMENT_METHOD = "automated model review, not a human annotation"
VALIDATION_ERRORS = (ValueError, KeyError, IndexError, TypeError)


@dataclass(frozen=True)
class CallLimits:
    maximum_output_tokens: int = 16384
    maximum_attempts: int = 2
    maximum_usd_per_item: float = 0.30
    request_timeout_seconds: int = 120

    def __post_init__(self) -> None:
        for name in ("maximum_output_tokens", "maximum_attempts", "request_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.maximum_usd_per_item) or self.maximum_usd_per_item <= 0:
            raise ValueError("per-item monetary ceiling must be positive and finite")

    def run_limits(self) -> RunLimits:
        return RunLimits(
            0,
            self.maximum_attempts,
            100000,
            self.maximum_attempts * self.maximum_output_tokens,
            self.maximum_usd_per_item,
            self.maximum_attempts * self.request_timeout_seconds * 1000,
            0,
            1,
        )


@dataclass(frozen=True)
class JudgeCall:
    """Everything one judged item needs, so the call has no hidden global state."""

    run_id: str
    experiment_id: str
    model_id: str
    item: dict[str, Any]
    input_hash: str
    plan_hash: str
    limits: CallLimits


def build_request(item: dict[str, Any], limits: CallLimits) -> ProviderRequest:
    return ProviderRequest(
        role=ROLE,
        system_prompt=REQUEST_RUBRIC,
        messages=[{"role": "user", "content": canonical_json(item)}],
        tools=[
            {
                "name": TOOL_NAME,
                "description": "Submit the three bounded semantic judgments with exact references.",
                "input_schema": tool_schema(item),
                "strict": True,
            }
        ],
        maximum_output_tokens=limits.maximum_output_tokens,
        timeout_seconds=limits.request_timeout_seconds,
        effort="high",
    )


def append_correction(
    request: ProviderRequest, response: ProviderResponse, error: Exception
) -> None:
    """Answer every submitted tool call with the validation error. Truncated replies are skipped."""
    if not response.tool_calls:
        return
    message = (
        "Assessment failed validation. Submit every required field. Select only catalog "
        "finding_reference_ids and supplied evidence_ids; use empty lists when needed. "
        + str(error).splitlines()[0][:300]
    )
    feedback = [
        {
            "type": "tool_result",
            "tool_use_id": call.tool_use_id,
            "content": canonical_json({"error": "INVALID_ASSESSMENT", "message": message}),
            "is_error": True,
        }
        for call in response.tool_calls
    ]
    request.messages.extend(
        [
            {"role": "assistant", "content": response.assistant_content},
            {"role": "user", "content": feedback},
        ]
    )


def _validated(response: ProviderResponse, item: dict[str, Any]) -> Assessment:
    if response.stop_reason != "tool_calls" or len(response.tool_calls) != 1:
        raise ValueError(
            f"expected one complete terminal assessment tool call, stop reason "
            f"{response.stop_reason}, {len(response.tool_calls)} tool calls"
        )
    call = response.tool_calls[0]
    if call.name != TOOL_NAME:
        raise ValueError("unexpected terminal tool")
    return decode_assessment(call.arguments, item)


def judge_item(
    store: RunStore, adapter: ModelAdapter, rates: RateCard, call: JudgeCall, destination: Path
) -> dict[str, Any]:
    """Judge one item, persist the outcome next to its ledger run, and return it."""
    if destination.exists():
        saved: dict[str, Any] = json.loads(destination.read_text())
        return saved
    item_id = call.item["item_id"]
    if (
        store.connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (call.run_id,)).fetchone()
        is None
    ):
        store.create_run(
            call.run_id,
            item_id,
            call.item["candidate"]["event_id"],
            "semantic_review",
            adapter.execution_mode.value,
            call.model_id,
            call.input_hash,
            {"plan_hash": call.plan_hash, "input_hash": call.input_hash},
            {"item_id": item_id, "assessment_method": ASSESSMENT_METHOD},
            call.limits.run_limits(),
            experiment_id=call.experiment_id,
        )
    result: dict[str, Any] = {
        "item_id": item_id,
        "run_id": call.run_id,
        "model_id": call.model_id,
        "assessment_method": ASSESSMENT_METHOD,
        "assessment": None,
        "status": "unassessed",
    }
    if store.actions(call.run_id, ActionKind.provider_attempt):
        _recover_prior_attempts(store, call, result)
    else:
        _dispatch_attempts(store, adapter, rates, call, result)
    result["usage"] = store.usage(call.run_id).__dict__
    store.mark_terminal(
        call.run_id, "completed" if result["assessment"] else "failed", result["status"]
    )
    write_json_atomic(destination, result)
    return result


def _recover_prior_attempts(store: RunStore, call: JudgeCall, result: dict[str, Any]) -> None:
    """Reuse a persisted valid completion. Never redispatch an attempt with unknown outcome."""
    for action in store.actions(call.run_id, ActionKind.provider_attempt):
        if action.state == ActionState.started:
            store.mark_uncertain(action.action_id, "prior dispatch has no persisted completion")
        elif action.state == ActionState.reserved:
            store.complete_action(
                action.action_id,
                ActionState.failed,
                0,
                error_code="NOT_DISPATCHED",
                usage={"usage_known": True, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
            )
        if action.response is None:
            continue
        response = ProviderResponse.from_record(action.response)
        if not response.usage.usage_known or response.effective_settings.get("settings_errors"):
            continue
        try:
            assessment = _validated(response, call.item)
        except VALIDATION_ERRORS:
            continue
        result.update(status="assessed", assessment=assessment.model_dump(mode="json"))
    if result["assessment"] is None:
        result["status"] = "unresolved_prior_attempt"


def _dispatch_attempts(
    store: RunStore,
    adapter: ModelAdapter,
    rates: RateCard,
    call: JudgeCall,
    result: dict[str, Any],
) -> None:
    request = build_request(call.item, call.limits)
    logical_key = f"{call.run_id}:{ROLE}"
    for attempt_number in range(1, call.limits.maximum_attempts + 1):
        estimate = adapter.estimate_input_tokens(request)
        reservation = Reservation(
            input_tokens=estimate.input_tokens,
            output_tokens=request.maximum_output_tokens,
            usd=rates.reservation_cost(estimate.input_tokens, request.maximum_output_tokens),
            estimation_method=estimate.method,
        )
        try:
            action_id = store.reserve_action(
                call.run_id,
                f"{logical_key}:{attempt_number}",
                logical_key,
                ROLE,
                ActionKind.provider_attempt,
                {"input_hash": call.input_hash, "plan_hash": call.plan_hash},
                reservation,
                attempt_number=attempt_number,
            )
        except BudgetExhausted as exhausted:
            result.update(status="budget_prevented", reason=exhausted.limit_name)
            return
        store.mark_started(action_id)
        started = store.clock.monotonic_ms()
        try:
            response = adapter.complete(request)
        except ProviderFailure as failure:
            duration = store.clock.monotonic_ms() - started
            store.add_active_ms(call.run_id, duration)
            if failure.kind == FailureKind.uncertain:
                store.mark_uncertain(action_id, str(failure), duration)
                result.update(status="provider_uncertain", reason=str(failure))
                return
            store.complete_action(
                action_id,
                ActionState.failed,
                duration,
                error_code="PROVIDER_FAILURE",
                error=failure.to_record(),
                usage={"usage_known": True, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
            )
            result.update(status="provider_failed", reason=str(failure))
            if failure.kind == FailureKind.permanent:
                return
            continue
        duration = store.clock.monotonic_ms() - started
        store.complete_action(
            action_id,
            ActionState.completed,
            duration,
            response=response.to_record(),
            usage={**response.usage.__dict__, "usd": rates.cost(response.usage)},
            provider_request_id=response.request_id,
        )
        store.add_active_ms(call.run_id, duration)
        if not response.usage.usage_known or response.effective_settings.get("settings_errors"):
            result.update(
                status="unverified_provider_response", reason="usage or routing not established"
            )
            return
        try:
            assessment = _validated(response, call.item)
        except VALIDATION_ERRORS as error:
            result.update(
                status="invalid_assessment", reason=f"{type(error).__name__}: {error}"[:500]
            )
            append_correction(request, response, error)
            continue
        result.update(status="assessed", assessment=assessment.model_dump(mode="json"))
        result.pop("reason", None)
        return
