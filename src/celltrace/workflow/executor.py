"""One observable role turn: a provider attempt and the tool calls it requested.

Crash safety comes from the ledger, not from the graph checkpoint:
- A provider attempt is reserved, then marked started, then completed together
  with its assistant message in one transaction.
- On entry, a turn first finishes any assistant message whose tool calls have
  no results yet, reusing completed tool actions by their operation keys.
- A provider attempt left in ``started`` state is uncertain. It may have run
  and been billed, so it is never resubmitted automatically.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from celltrace.agents.roles import ROLE_SPECS, RoleSpec, tool_result_block
from celltrace.canonical import canonical_json
from celltrace.clock import Clock
from celltrace.config import DetectorSection, ObservationSection, RuntimeSection
from celltrace.contracts.common import (
    MEASUREMENT_TOOLS,
    RETRYABLE_ERROR_CODES,
    ErrorCode,
    Role,
)
from celltrace.contracts.evidence import RunContext, ToolError
from celltrace.providers.base import (
    BYTE_ESTIMATE_METHOD,
    FailureKind,
    ModelAdapter,
    ProviderFailure,
    ProviderRequest,
    ProviderResponse,
    ToolCall,
)
from celltrace.providers.rate_card import RateCard
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    BudgetExhausted,
    Reservation,
    RunStore,
)
from celltrace.tools.dispatch import ToolDispatcher, ToolOutcome

CALIBRATION_MARGIN = 1.25
PROTOCOL_REMINDER = (
    "Protocol error: plain text is not accepted as findings or a review. "
    "Call one measurement tool or the terminal submission tool."
)


class TurnStatus(StrEnum):
    continued = "continued"
    submitted = "submitted"
    invalid_submission = "invalid_submission"
    protocol_error = "protocol_error"
    budget_exhausted = "budget_exhausted"
    provider_failed = "provider_failed"
    provider_uncertain = "provider_uncertain"


@dataclass
class TurnResult:
    status: TurnStatus
    submission: dict[str, Any] | None = None
    submission_action_id: int | None = None
    detail: str = ""
    measurement_actions: list[int] = field(default_factory=list)


def pending_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool use blocks of the final assistant message that have no results yet."""
    if not messages or messages[-1]["role"] != "assistant":
        return []
    return [block for block in messages[-1]["content"] if block.get("type") == "tool_use"]


class RoleTurnExecutor:
    def __init__(
        self,
        context: RunContext,
        store: RunStore,
        dispatcher: ToolDispatcher,
        adapter: ModelAdapter,
        runtime: RuntimeSection,
        clock: Clock,
        rate_card: RateCard | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_source: random.Random | None = None,
        fault_hook: Callable[[str], None] | None = None,
        detector_configuration: DetectorSection | None = None,
        observation_configuration: ObservationSection | None = None,
    ):
        self.context = context
        self.run_id = context.run_id
        self.store = store
        self.dispatcher = dispatcher
        self.adapter = adapter
        self.runtime = runtime
        self.detector_configuration = detector_configuration or DetectorSection()
        self.observation_configuration = observation_configuration or ObservationSection()
        self.clock = clock
        self.rate_card = rate_card
        self.sleep = sleep
        self.random_source = random_source or random.Random(0)
        # Tests use named points to simulate a process stopping at exact boundaries.
        self.fault_hook = fault_hook or (lambda point: None)

    # Entry point

    def turn(self, role: Role) -> TurnResult:
        spec = ROLE_SPECS[role]
        messages = self.store.messages(self.run_id, role.value)
        pending = pending_tool_calls(messages)
        if pending:
            # Resume or finish the tool calls of an already completed provider attempt.
            source_action = self._latest_provider_action(role)
            return self._handle_tool_calls(
                spec,
                [ToolCall(block["id"], block["name"], dict(block["input"])) for block in pending],
                source_action,
            )
        uncertain = self._reconcile_open_attempts(role)
        if uncertain is not None:
            return uncertain
        turn_number = sum(1 for message in messages if message["role"] == "assistant") + 1
        response, action_id_or_result = self._provider_attempts(spec, messages, turn_number)
        if response is None:
            assert isinstance(action_id_or_result, TurnResult)
            return action_id_or_result
        assert isinstance(action_id_or_result, int)
        settings_errors = response.effective_settings.get("settings_errors", [])
        if settings_errors:
            return TurnResult(TurnStatus.provider_failed, detail=canonical_json(settings_errors))
        if response.stop_reason in ("length", "max_tokens"):
            detail = "provider completion truncated"
            if response.usage.reasoning_tokens and not response.text and not response.tool_calls:
                detail = "reasoning exhausted completion allowance before a usable response"
            return TurnResult(TurnStatus.provider_failed, detail=detail)
        if not response.tool_calls:
            if response.stop_reason == "refusal":
                return TurnResult(TurnStatus.provider_failed, detail="model refusal")
            self.store.append_message(
                self.run_id,
                role.value,
                {"role": "user", "content": [{"type": "text", "text": PROTOCOL_REMINDER}]},
            )
            return TurnResult(TurnStatus.protocol_error, detail=response.stop_reason)
        return self._handle_tool_calls(spec, response.tool_calls, action_id_or_result)

    # Provider attempts

    def _latest_provider_action(self, role: Role) -> int:
        actions = [
            action
            for action in self.store.actions(self.run_id, ActionKind.provider_attempt)
            if action.role == role.value and action.state == ActionState.completed
        ]
        return actions[-1].action_id

    def _reconcile_open_attempts(self, role: Role) -> TurnResult | None:
        for action in self.store.actions(self.run_id, ActionKind.provider_attempt):
            if action.role != role.value:
                continue
            if action.state == ActionState.reserved:
                # Reserved but never dispatched: no provider call happened.
                self.store.complete_action(
                    action.action_id,
                    ActionState.failed,
                    0,
                    error_code="NOT_DISPATCHED",
                    error={"reason": "process stopped after reservation, before dispatch"},
                    usage={"usage_known": True, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
                )
                self.store.record_event(
                    self.run_id, "reservation_not_dispatched", {"action_id": action.action_id}
                )
            elif action.state == ActionState.started:
                self.store.mark_uncertain(action.action_id, "started without a persisted result")
                self.store.record_event(
                    self.run_id,
                    "provider_outcome_uncertain",
                    {
                        "action_id": action.action_id,
                        "recovery": "celltrace resume --run <run-id> --retry-uncertain "
                        "creates a newly charged attempt after you accept unknown prior usage",
                    },
                )
                return TurnResult(TurnStatus.provider_uncertain, detail="uncertain prior attempt")
            elif action.state == ActionState.uncertain and not self._uncertain_released(action):
                return TurnResult(TurnStatus.provider_uncertain, detail="uncertain prior attempt")
        return None

    def _uncertain_released(self, action: Any) -> bool:
        return any(
            event["event_type"] == "uncertain_retry_authorized"
            and event["payload"].get("action_id") == action.action_id
            for event in self.store.run_events(self.run_id)
        )

    def _request(self, spec: RoleSpec, messages: list[dict[str, Any]]) -> ProviderRequest:
        usage = self.store.usage(self.run_id)
        remaining_output = self.runtime.maximum_output_tokens - usage.output_tokens
        resources = self.store.resource_allowance(self.run_id, spec.role.value)
        remaining_output = min(remaining_output, resources["output_tokens"].role_remaining)
        if self.runtime.show_remaining_budget:
            # This dynamic suffix is outside the fixed cached system prefix. Persist its
            # exact content in the action request, not in the role's replay conversation.
            note = {
                "type": "text",
                "text": "Current ledger budget (input includes cached tokens; estimated "
                "admission is not an invoice guarantee): "
                + canonical_json(self._budget_view(spec.role.value)),
            }
            if messages and messages[-1]["role"] == "user":
                messages = [
                    *messages[:-1],
                    messages[-1] | {"content": [*messages[-1]["content"], note]},
                ]
            else:
                messages = [*messages, {"role": "user", "content": [note]}]
        remaining_seconds = self.runtime.maximum_active_seconds - usage.active_ms / 1000
        # The limits are fixed for the run, so the cached prompt prefix stays stable.
        budget_note = (
            f"\n\nRun limits: {self.runtime.maximum_tool_requests} tool requests and "
            f"{self.runtime.maximum_provider_attempts} model calls in total for all roles."
        )
        reserve = self.store.run_limits(self.run_id).reviewer_tool_reserve
        if reserve:
            budget_note += (
                f" Investigator lifetime tool ceiling is "
                f"{self.runtime.maximum_tool_requests - reserve}; {reserve} requests are reserved "
                "for the Reviewer. The Reviewer may also use unused Investigator capacity."
            )
        return ProviderRequest(
            role=spec.role.value,
            system_prompt=spec.system_prompt + self._pattern_note() + budget_note,
            messages=messages,
            tools=spec.tool_definitions(strict=self.runtime.strict_tool_schemas),
            maximum_output_tokens=max(
                min(self.runtime.maximum_output_tokens_per_attempt, remaining_output), 1
            ),
            timeout_seconds=max(min(self.runtime.provider_timeout_seconds, remaining_seconds), 1),
            effort=self.runtime.model_effort
            if self.adapter.model_id != "scripted-fixture"
            else None,
        )

    def _pattern_note(self) -> str:
        configuration = self.detector_configuration
        rule = {
            "reference_ms": configuration.reference_ms,
            "event_ms": configuration.event_ms,
            "minimum_coverage": configuration.minimum_coverage,
            "minimum_active_fraction": configuration.minimum_active_fraction,
            "minimum_reference_mbps": configuration.minimum_reference_mbps,
            "minimum_absolute_drop_mbps": configuration.minimum_absolute_drop_mbps,
            "minimum_relative_drop": configuration.minimum_relative_drop,
        }
        radio_rule = self.observation_configuration.model_dump(mode="json")
        return (
            "\n\nConfigured active_rate_drop observation rule: "
            + canonical_json(rule)
            + "\nReported radio labels need verified median RSRP changes on the "
            "representative windows with sufficient coverage. Stable means absolute change "
            "at most the tolerance; degradation means a decrease greater than the tolerance. "
            "These labels describe reported RSRP, not all radio health or physical cause. "
            + canonical_json(radio_rule)
        )

    def _calibration_factor(self, role: str) -> float | None:
        """Observed prompt tokens per byte estimate for this role's last completed attempt.

        Replayed reasoning text tokenizes far more densely than two bytes per
        token, so the raw estimate reserved about twice the real prompt in live
        runs. The factor keeps a 25 percent margin and never exceeds the raw bound.
        """
        for row in reversed(self.store.raw_actions(self.run_id)):
            if row["kind"] != ActionKind.provider_attempt.value or row["role"] != role:
                continue
            if row["state"] != ActionState.completed.value or not row["usage_known"]:
                continue
            byte_estimate = int(json.loads(row["request_json"]).get("byte_estimate") or 0)
            if byte_estimate <= 0:
                return None
            observed = int(
                (row["actual_input_tokens"] or 0)
                + (row["cache_read_input_tokens"] or 0)
                + (row["cache_creation_input_tokens"] or 0)
            )
            return min(1.0, observed / byte_estimate * CALIBRATION_MARGIN)
        return None

    def _reservation(self, request: ProviderRequest, role: str) -> tuple[Reservation, int]:
        estimate = self.adapter.estimate_input_tokens(request)
        input_tokens = estimate.input_tokens
        method = estimate.method
        if method == BYTE_ESTIMATE_METHOD:
            factor = self._calibration_factor(role)
            if factor is not None:
                input_tokens = math.ceil(estimate.input_tokens * factor)
                method = f"{method}_calibrated_by_last_role_attempt"
        usd = 0.0
        if self.rate_card is not None:
            if estimate.method != "provider_count_tokens":
                input_tokens += self.rate_card.tool_use_system_prompt_tokens_auto
            usd = self.rate_card.reservation_cost(input_tokens, request.maximum_output_tokens)
        reservation = Reservation(
            input_tokens=input_tokens,
            output_tokens=request.maximum_output_tokens,
            usd=usd,
            estimation_method=method,
        )
        return reservation, estimate.input_tokens

    def _provider_attempts(
        self, spec: RoleSpec, messages: list[dict[str, Any]], turn_number: int
    ) -> tuple[ProviderResponse | None, int | TurnResult]:
        role = spec.role.value
        logical_key = f"{self.run_id}:{role}:turn:{turn_number}"
        previous = [
            action
            for action in self.store.actions(self.run_id, ActionKind.provider_attempt)
            if action.logical_key == logical_key
        ]
        attempt = len(previous) + 1
        maximum_attempts = 1 + self.runtime.maximum_request_retries
        while True:
            request = self._request(spec, messages)
            reservation, byte_estimate = self._reservation(request, role)
            try:
                action_id = self.store.reserve_action(
                    self.run_id,
                    f"{logical_key}:attempt:{attempt}",
                    logical_key,
                    role,
                    ActionKind.provider_attempt,
                    {
                        "role": role,
                        "turn": turn_number,
                        "message_count": len(messages),
                        "byte_estimate": byte_estimate,
                        "maximum_output_tokens": request.maximum_output_tokens,
                        "resource_budget": self._budget_view(role)
                        if self.runtime.show_remaining_budget
                        else None,
                    },
                    reservation,
                    attempt_number=attempt,
                )
            except BudgetExhausted as exhausted:
                return None, TurnResult(TurnStatus.budget_exhausted, detail=exhausted.limit_name)
            self.fault_hook("after_provider_reservation")
            self.store.mark_started(action_id)
            started = self.clock.monotonic_ms()
            try:
                self.fault_hook("during_provider_call")
                response = self.adapter.complete(request)
            except ProviderFailure as failure:
                duration = self.clock.monotonic_ms() - started
                self.store.add_active_ms(self.run_id, duration)
                if failure.kind == FailureKind.uncertain:
                    self.store.mark_uncertain(action_id, str(failure), duration)
                    self.store.record_event(
                        self.run_id, "provider_outcome_uncertain", failure.to_record()
                    )
                    return None, TurnResult(TurnStatus.provider_uncertain, detail=str(failure))
                self.store.complete_action(
                    action_id,
                    ActionState.failed,
                    duration,
                    error_code=f"PROVIDER_{failure.kind.value.upper()}",
                    error=failure.to_record(),
                    usage={"usage_known": True, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
                    provider_request_id=failure.request_id,
                )
                if failure.kind == FailureKind.transient and attempt < maximum_attempts:
                    attempt += 1
                    delay_seconds = min(2.0**attempt, 8.0) * (0.5 + self.random_source.random())
                    self.sleep(delay_seconds)
                    self.store.add_active_ms(self.run_id, int(delay_seconds * 1000))
                    continue
                return None, TurnResult(TurnStatus.provider_failed, detail=str(failure))
            duration = self.clock.monotonic_ms() - started
            usage = response.usage
            usd = self.rate_card.cost(usage) if self.rate_card is not None else 0.0
            sequence = len(messages)
            self.store.complete_action(
                action_id,
                ActionState.completed,
                duration,
                response=response.to_record(),
                usage={**usage.__dict__, "usd": usd},
                provider_request_id=response.request_id,
                message=(
                    self.run_id,
                    role,
                    sequence,
                    {"role": "assistant", "content": response.assistant_content},
                ),
            )
            self.store.add_active_ms(self.run_id, duration)
            self.fault_hook("after_provider_result")
            return response, action_id

    # Tool calls

    def _handle_tool_calls(
        self, spec: RoleSpec, calls: list[ToolCall], source_action_id: int
    ) -> TurnResult:
        role = spec.role.value
        results: list[dict[str, Any]] = []
        measurement_actions: list[int] = []
        for call in calls:
            if call.name == spec.terminal_tool:
                if len(calls) > 1:
                    results.append(
                        tool_result_block(
                            call.tool_use_id,
                            self._with_budget(
                                role, {"error": "submit alone, not together with other tool calls"}
                            ),
                            True,
                        )
                    )
                    continue
                try:
                    validated = spec.terminal_model.model_validate(call.arguments)
                except ValidationError as validation_error:
                    problems = [
                        {"location": list(map(str, problem["loc"])), "message": problem["msg"]}
                        for problem in validation_error.errors()[:20]
                    ]
                    if "invalid_provider_arguments" in call.arguments:
                        # A cut off object otherwise reads as every field missing.
                        problems = [
                            {
                                "location": [],
                                "message": "The arguments were not a complete JSON object. "
                                "If the answer reached the completion limit, keep reasoning "
                                "short and submit the whole object.",
                            }
                        ]
                    self.store.append_message(
                        self.run_id,
                        role,
                        {
                            "role": "user",
                            "content": [
                                tool_result_block(
                                    call.tool_use_id,
                                    self._with_budget(
                                        role,
                                        {
                                            "error": ErrorCode.invalid_output.value,
                                            "problems": problems,
                                        },
                                    ),
                                    True,
                                )
                            ],
                        },
                    )
                    return TurnResult(
                        TurnStatus.invalid_submission, detail=canonical_json(problems)
                    )
                return TurnResult(
                    TurnStatus.submitted,
                    submission={
                        "tool_use_id": call.tool_use_id,
                        "payload": validated.model_dump(mode="json"),
                    },
                    submission_action_id=source_action_id,
                )
            outcome, action_id = self._execute_tool(role, call)
            self.fault_hook("after_tool_result")
            is_measurement = call.name in {tool.value for tool in MEASUREMENT_TOOLS}
            if action_id is not None and is_measurement and outcome.error is None:
                measurement_actions.append(action_id)
            results.append(
                tool_result_block(
                    call.tool_use_id,
                    self._tool_payload(role, outcome),
                    is_error=outcome.error is not None,
                )
            )
            if outcome.error is not None and outcome.error.code == ErrorCode.budget_exhausted:
                self.store.append_message(self.run_id, role, {"role": "user", "content": results})
                return TurnResult(
                    TurnStatus.budget_exhausted,
                    detail=str(outcome.error.details.get("limit_name", "tool_requests")),
                )
        self.store.append_message(self.run_id, role, {"role": "user", "content": results})
        self.fault_hook("end_of_turn")
        return TurnResult(TurnStatus.continued, measurement_actions=measurement_actions)

    def _tool_payload(self, role: str, outcome: ToolOutcome) -> dict[str, Any]:
        return self._with_budget(role, outcome.model_payload())

    def _with_budget(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.runtime.show_remaining_budget:
            # Keep historical tool results compact; the full fresh ledger view is
            # supplied ephemerally before each call. Repeating it in history grows
            # input quadratically as the conversation is replayed.
            resources = self.store.resource_allowance(self.run_id, role)
            budget = self.store.tool_allowance(self.run_id, role) | {
                "role": role,
                "provider_attempts_remaining": resources["provider_attempts"].role_remaining,
                "input_tokens_remaining": resources["input_tokens"].role_remaining,
                "output_tokens_remaining": resources["output_tokens"].role_remaining,
            }
            payload = payload | {"remaining_budget": budget}
        return payload

    def _budget_view(self, role: str) -> dict[str, Any]:
        resources = self.store.resource_allowance(self.run_id, role)
        return self.store.tool_allowance(self.run_id, role) | {
            "role": role,
            "provider_attempts_remaining": resources["provider_attempts"].role_remaining,
            "resources": {name: asdict(value) for name, value in resources.items()},
        }

    def _execute_tool(self, role: str, call: ToolCall) -> tuple[ToolOutcome, int | None]:
        logical_key = f"{self.run_id}:{role}:tool:{call.tool_use_id}"
        previous = [
            action
            for action in self.store.actions(self.run_id, ActionKind.tool_request)
            if action.logical_key == logical_key
        ]
        if previous and previous[-1].state == ActionState.completed:
            return self._replay_tool(previous[-1]), previous[-1].action_id
        attempt = len(previous) + 1
        maximum_attempts = 1 + self.runtime.maximum_request_retries
        while True:
            try:
                action_id = self.store.reserve_action(
                    self.run_id,
                    f"{logical_key}:attempt:{attempt}",
                    logical_key,
                    role,
                    ActionKind.tool_request,
                    {"tool": call.name, "arguments": call.arguments},
                    Reservation(tool_units=1),
                    attempt_number=attempt,
                    tool_name=call.name,
                )
            except BudgetExhausted as exhausted:
                error = ToolError(
                    code=ErrorCode.budget_exhausted,
                    message=f"{exhausted.limit_name} allowance is used up",
                    details={"limit_name": exhausted.limit_name},
                )
                return ToolOutcome(tool_name=call.name, error=error), None
            self.store.mark_started(action_id)
            started = self.clock.monotonic_ms()
            outcome = self.dispatcher.dispatch(call.name, call.arguments)
            duration = self.clock.monotonic_ms() - started
            if outcome.evidence is not None and not outcome.cache_hit:
                # Evidence is persisted before the model can see it.
                self.store.save_evidence(outcome.evidence.model_dump(mode="json"))
            state = ActionState.completed if outcome.error is None else ActionState.failed
            self.store.complete_action(
                action_id,
                state,
                duration,
                response={"payload": outcome.model_payload()},
                evidence_id=None if outcome.evidence is None else outcome.evidence.evidence_id,
                error_code=None if outcome.error is None else outcome.error.code.value,
                error=None if outcome.error is None else outcome.error.model_dump(mode="json"),
                cache_hit=outcome.cache_hit,
                physical_query=outcome.physical_query,
            )
            self.store.add_active_ms(self.run_id, duration)
            retryable = (
                outcome.error is not None
                and outcome.error.retryable
                and outcome.error.code in RETRYABLE_ERROR_CODES
            )
            if retryable and attempt < maximum_attempts:
                attempt += 1
                continue
            return outcome, action_id

    def _replay_tool(self, action: Any) -> ToolOutcome:
        self.store.record_event(
            self.run_id,
            "tool_result_replayed",
            {"action_id": action.action_id, "operation_key": action.operation_key},
        )
        payload = action.response["payload"]
        if action.evidence_id is not None:
            from celltrace.contracts.evidence import EvidenceRecord

            record = self.store.get_evidence(action.evidence_id)
            assert record is not None
            return ToolOutcome(
                tool_name=action.tool_name, evidence=EvidenceRecord.model_validate(record)
            )
        if "evidence_record" in payload:
            return ToolOutcome(tool_name=action.tool_name, retrieved=payload["evidence_record"])
        return ToolOutcome(
            tool_name=action.tool_name, error=ToolError.model_validate(payload["error"])
        )
