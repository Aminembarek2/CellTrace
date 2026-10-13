"""Charged native tool continuation and terminal schema probe for both roles."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from celltrace.agents.roles import INVESTIGATOR, REVIEWER, RoleSpec, tool_result_block
from celltrace.canonical import canonical_json
from celltrace.clock import Clock, SystemClock
from celltrace.config import AppConfig, api_key_present
from celltrace.contracts.common import ToolName
from celltrace.contracts.tools import TOOL_ARGUMENT_MODELS
from celltrace.providers.base import FailureKind, ProviderFailure, ProviderRequest, ProviderResponse
from celltrace.providers.openrouter_adapter import OpenRouterAdapter
from celltrace.providers.rate_card import RateCard
from celltrace.providers.readiness import capability_identity, verified_rate_card
from celltrace.storage.lock import RunLock
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    BudgetExhausted,
    Reservation,
    RunStore,
)
from celltrace.workflow.runner import run_limits

PROBE_EXPERIMENT_ID = "exp_openrouter_capability_probes"
PROBE_PREAMBLE = (
    "Protocol compatibility test. This is not an investigation or a review and there is "
    "nothing to evaluate. Follow the instruction literally. "
)


def _terminal_fixture(specification: RoleSpec) -> dict[str, Any]:
    payload: dict[str, Any]
    if specification == REVIEWER:
        payload = {
            "decision": "unresolved",
            "reason_summary": "Protocol probe without measurement data",
            "review_limitation": "No measurement snapshot in this probe",
        }
    else:
        payload = {
            "event_id": "evt_protocol_probe",
            "eligibility": {"status": "unknown", "supporting_claim_ids": [], "reason": "probe"},
            "factual_claims": [],
            "observation_patterns": [],
            "hypotheses": [],
            "missing_evidence": ["No measurement snapshot in this probe"],
            "next_measurements": [],
            "decision_summary": "Protocol probe",
            "cause_status": "not_assessed",
        }
    return specification.terminal_model.model_validate(payload).model_dump(mode="json")


def _charged_attempt(
    adapter: OpenRouterAdapter,
    request: ProviderRequest,
    store: RunStore,
    run_id: str,
    logical_key: str,
    clock: Clock,
    rate_card: RateCard,
) -> ProviderResponse:
    estimate = adapter.estimate_input_tokens(request)
    action_id = store.reserve_action(
        run_id,
        logical_key,
        logical_key,
        request.role,
        ActionKind.provider_attempt,
        {"request": logical_key},
        Reservation(
            input_tokens=estimate.input_tokens,
            output_tokens=request.maximum_output_tokens,
            usd=rate_card.reservation_cost(estimate.input_tokens, request.maximum_output_tokens),
            estimation_method=estimate.method,
        ),
    )
    store.mark_started(action_id)
    started_ms = clock.monotonic_ms()
    try:
        response = adapter.complete(request)
    except ProviderFailure as failure:
        duration_ms = clock.monotonic_ms() - started_ms
        store.add_active_ms(run_id, duration_ms)
        if failure.kind == FailureKind.uncertain:
            store.mark_uncertain(action_id, str(failure), duration_ms)
        else:
            store.complete_action(
                action_id,
                ActionState.failed,
                duration_ms,
                error=failure.to_record(),
                error_code=failure.kind.value,
                usage={"usage_known": True, "input_tokens": 0, "output_tokens": 0, "usd": 0.0},
            )
        raise
    duration_ms = clock.monotonic_ms() - started_ms
    store.complete_action(
        action_id,
        ActionState.completed,
        duration_ms,
        response=response.to_record(),
        usage={**response.usage.__dict__, "usd": rate_card.cost(response.usage)},
        provider_request_id=response.request_id,
    )
    store.add_active_ms(run_id, duration_ms)
    return response


def run_live_probe(
    model_id: str,
    config: AppConfig,
    out_dir: Path,
    adapter: OpenRouterAdapter | None = None,
) -> dict[str, Any]:
    if not config.runtime.enable_live_calls or not api_key_present():
        raise ValueError("a live probe requires explicit live activation and OPENROUTER_API_KEY")
    if model_id != config.runtime.model_id:
        raise ValueError("probe model must match the validated configuration")
    rate_card = verified_rate_card(config.runtime)
    adapter = adapter or OpenRouterAdapter(model_id)
    clock = SystemClock()
    run_id = f"probe_{uuid.uuid4().hex}"
    identity = capability_identity(config.runtime)
    results: dict[str, Any] = {
        "run_id": run_id,
        "model_id": model_id,
        "identity": identity,
        "checked_at": clock.utc_now_iso(),
        "checks": {},
        "status": "failed",
        "scope": "provider protocol only, no adaptive diagnostic performance measured",
    }
    store = RunStore(config.runtime.run_store, clock)
    store.create_experiment(
        PROBE_EXPERIMENT_ID,
        config.runtime.maximum_estimated_usd_per_experiment,
        {"purpose": "explicit provider capability probes"},
    )
    store.create_run(
        run_id,
        run_id,
        "evt_protocol_probe",
        "capability_probe",
        "live",
        model_id,
        "no_measurement_snapshot",
        identity,
        {},
        run_limits(config, rate_card),
        experiment_id=PROBE_EXPERIMENT_ID,
    )
    try:
        with RunLock(config.runtime.run_store.parent / "locks", run_id):
            for specification in (INVESTIGATOR, REVIEWER):
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": "Call the tool"}]}
                ]
                terminal_fixture = _terminal_fixture(specification)
                for phase in ("measurement", "terminal"):
                    usage = store.usage(run_id)
                    expected_tool = (
                        "get_session_context"
                        if phase == "measurement"
                        else specification.terminal_tool
                    )
                    instruction = "Call get_session_context with no arguments."
                    if phase == "terminal":
                        # Without this framing a model treated the probe as a real review.
                        instruction = (
                            "Copy every value verbatim and add no other fields. "
                            f"Call {expected_tool} with exactly {canonical_json(terminal_fixture)}"
                        )
                    request = ProviderRequest(
                        role=specification.role.value,
                        system_prompt=PROBE_PREAMBLE + instruction,
                        messages=messages,
                        tools=specification.tool_definitions(
                            strict=config.runtime.strict_tool_schemas
                        ),
                        maximum_output_tokens=config.runtime.maximum_output_tokens_per_attempt,
                        timeout_seconds=min(
                            config.runtime.provider_timeout_seconds,
                            max(
                                0.001,
                                config.runtime.maximum_active_seconds - usage.active_ms / 1000,
                            ),
                        ),
                        effort=config.runtime.model_effort,
                    )
                    logical_key = f"{run_id}:{specification.role.value}:{phase}"
                    response = _charged_attempt(
                        adapter, request, store, run_id, logical_key, clock, rate_card
                    )
                    passed = (
                        len(response.tool_calls) == 1
                        and response.tool_calls[0].name == expected_tool
                        and response.stop_reason == "tool_calls"
                        and response.usage.usage_known
                        and not response.effective_settings.get("settings_errors")
                    )
                    if passed and phase == "terminal":
                        submitted = specification.terminal_model.model_validate(
                            response.tool_calls[0].arguments
                        )
                        passed = submitted.model_dump(mode="json") == terminal_fixture
                    elif passed:
                        validated = TOOL_ARGUMENT_MODELS[
                            ToolName.get_session_context
                        ].model_validate(
                            response.tool_calls[0].arguments,
                        )
                        passed = validated.model_dump(exclude_none=True) == {}
                    results["checks"][f"{specification.role.value}_{phase}"] = {
                        "passed": passed,
                        "stop_reason": response.stop_reason,
                        "usage": response.usage.__dict__,
                        "effective_settings": response.effective_settings,
                        "provider_request_id": response.request_id,
                    }
                    if not passed:
                        raise ValueError(
                            "provider did not demonstrate the requested native tool protocol"
                        )
                    messages.append({"role": "assistant", "content": response.assistant_content})
                    if phase == "measurement":
                        tool_call = response.tool_calls[0]
                        tool_key = f"{logical_key}:tool:{tool_call.tool_use_id}"
                        tool_action = store.reserve_action(
                            run_id,
                            tool_key,
                            tool_key,
                            specification.role.value,
                            ActionKind.tool_request,
                            {"tool": expected_tool, "arguments": tool_call.arguments},
                            Reservation(tool_units=1),
                            tool_name=expected_tool,
                        )
                        store.mark_started(tool_action)
                        store.complete_action(
                            tool_action,
                            ActionState.completed,
                            0,
                            response={"probe": "no measurements supplied"},
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    tool_result_block(
                                        tool_call.tool_use_id,
                                        {"probe": "No measurement snapshot"},
                                        False,
                                    ),
                                    {
                                        "type": "text",
                                        "text": "Current ledger budget: "
                                        + canonical_json(store.usage(run_id).__dict__),
                                    },
                                ],
                            }
                        )
            results["status"] = "passed"
    except (ProviderFailure, BudgetExhausted, ValueError) as failure:
        results["failure"] = {"type": type(failure).__name__, "message": str(failure)}
    finally:
        results["usage"] = store.usage(run_id).__dict__
        store.mark_terminal(run_id, results["status"], "capability_probe_" + results["status"])
        store.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    results["saved_to"] = str(path)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results
