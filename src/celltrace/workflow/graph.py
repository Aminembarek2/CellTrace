"""LangGraph state machine for one investigation run.

The controller is code. Nodes: investigate, review, verify, revise, render.
Graph state only names the next step and the last turn status. Counters,
submissions and budgets are read from the run store, which is authoritative
after a crash, so a resumed node that repeats work finds it already recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from celltrace.agents.roles import reviewer_opening, tool_result_block
from celltrace.canonical import content_hash
from celltrace.contracts.common import MEASUREMENT_TOOLS, Arm, Role
from celltrace.contracts.findings import InvestigationFindings, ReviewDecision, ReviewResult
from celltrace.contracts.report import VerificationResult
from celltrace.storage.run_store import ActionKind, ActionState, RunStore
from celltrace.verification.verifier import Verifier
from celltrace.workflow.executor import RoleTurnExecutor, TurnStatus, pending_tool_calls

MAXIMUM_CONSECUTIVE_PROTOCOL_ERRORS = 3

TERMINAL_STATUSES = {
    TurnStatus.budget_exhausted,
    TurnStatus.provider_failed,
    TurnStatus.provider_uncertain,
}


class RunState(TypedDict, total=False):
    run_id: str
    last_status: str
    last_detail: str
    consecutive_protocol_errors: int
    terminal_reason: str


@dataclass
class GraphServices:
    run_id: str
    arm: Arm
    store: RunStore
    executor: RoleTurnExecutor
    verifier: Verifier
    maximum_review_rounds: int
    maximum_revisions: int
    render: Any  # callable(run_id, terminal_reason) -> dict


def _events(store: RunStore, run_id: str, event_type: str) -> list[dict[str, Any]]:
    return [event for event in store.run_events(run_id) if event["event_type"] == event_type]


def latest_findings(store: RunStore, run_id: str) -> tuple[str, int, dict[str, Any]] | None:
    rows = store.findings(run_id)
    return rows[-1] if rows else None


def revision_count(store: RunStore, run_id: str) -> int:
    return len(_events(store, run_id, "revision_requested"))


def build_graph(services: GraphServices) -> Any:
    store, run_id = services.store, services.run_id

    def after_turn(status: TurnStatus, state: RunState, detail: str) -> RunState:
        errors = state.get("consecutive_protocol_errors", 0)
        errors = errors + 1 if status == TurnStatus.protocol_error else 0
        update: RunState = {
            "last_status": status.value,
            "last_detail": detail,
            "consecutive_protocol_errors": errors,
        }
        if status in TERMINAL_STATUSES:
            update["terminal_reason"] = f"{status.value}:{detail}"
        elif errors >= MAXIMUM_CONSECUTIVE_PROTOCOL_ERRORS:
            update["terminal_reason"] = "protocol_errors_repeated"
        return update

    # Nodes

    def investigate(state: RunState) -> RunState:
        result = services.executor.turn(Role.investigator)
        if result.status == TurnStatus.submitted:
            assert result.submission is not None and result.submission_action_id is not None
            store.save_findings(run_id, result.submission["payload"], result.submission_action_id)
        return after_turn(result.status, state, result.detail)

    def ensure_reviewer_handoff(findings_id: str, findings: dict[str, Any]) -> int:
        handoffs = _events(store, run_id, "reviewer_handoff")
        for event in handoffs:
            if event["payload"]["findings_id"] == findings_id:
                return int(event["payload"]["action_watermark"])
        review_round = len(handoffs) + 1
        opening = reviewer_opening(services.executor.context, findings, review_round)
        if services.executor.runtime.require_review_measurement:
            opening["content"][0]["text"] += (
                "\nChoose a relevant measurement query in this review round before submitting. "
                "An earlier round's query is not a check of this revision. If a chosen "
                "measurement fails, record the limitation; that review remains limited."
            )
        pending = pending_tool_calls(store.messages(run_id, Role.reviewer.value))
        if pending:
            text = opening["content"][0]["text"]
            opening = {
                "role": "user",
                "content": [
                    tool_result_block(
                        pending[0]["id"], {"status": "revised findings", "next": text}, False
                    )
                ],
            }
        store.append_message(run_id, Role.reviewer.value, opening)
        actions = store.actions(run_id)
        watermark = actions[-1].action_id if actions else 0
        store.record_event(
            run_id,
            "reviewer_handoff",
            {"findings_id": findings_id, "round": review_round, "action_watermark": watermark},
        )
        return watermark

    def independent_queries(watermark: int) -> int:
        return sum(
            1
            for action in store.actions(run_id, ActionKind.tool_request)
            if action.role == Role.reviewer.value
            and action.action_id > watermark
            and action.state == ActionState.completed
            and action.tool_name in {tool.value for tool in MEASUREMENT_TOOLS}
            and action.evidence_id is not None
        )

    def review(state: RunState) -> RunState:
        latest = latest_findings(store, run_id)
        assert latest is not None
        findings_id, _, findings = latest
        watermark = ensure_reviewer_handoff(findings_id, findings)
        result = services.executor.turn(Role.reviewer)
        if result.status == TurnStatus.submitted:
            assert result.submission is not None and result.submission_action_id is not None
            payload = result.submission["payload"]
            try:
                ReviewResult.model_validate(payload).validate_finding_paths(findings)
            except ValueError as invalid_path:
                store.append_message(
                    run_id,
                    Role.reviewer.value,
                    {
                        "role": "user",
                        "content": [
                            tool_result_block(
                                result.submission["tool_use_id"],
                                {"error": "INVALID_OUTPUT", "message": str(invalid_path)},
                                True,
                            )
                        ],
                    },
                )
                return after_turn(TurnStatus.invalid_submission, state, "invalid_finding_path")
            count = independent_queries(watermark)
            attempted = any(
                action.role == Role.reviewer.value
                and action.action_id > watermark
                and action.tool_name in {tool.value for tool in MEASUREMENT_TOOLS}
                for action in store.actions(run_id, ActionKind.tool_request)
            )
            require_attempt = (
                services.executor.runtime.require_review_measurement
                and count == 0
                and not attempted
                and store.tool_allowance(run_id, Role.reviewer.value)["role_remaining"] > 0
            )
            if require_attempt or (
                payload["decision"] == ReviewDecision.accept.value
                and count == 0
                and not payload.get("review_limitation")
            ):
                store.append_message(
                    run_id,
                    Role.reviewer.value,
                    {
                        "role": "user",
                        "content": [
                            tool_result_block(
                                result.submission["tool_use_id"],
                                {
                                    "error": "INVALID_OUTPUT",
                                    "message": (
                                        "Choose a relevant measurement query in this round. "
                                        "Earlier queries and generic data gaps do not satisfy "
                                        "final-revision review."
                                        if require_attempt
                                        else "accept requires at least one independent "
                                        "measurement query in this round, or a review_limitation"
                                    ),
                                },
                                True,
                            )
                        ],
                    },
                )
                return after_turn(
                    TurnStatus.protocol_error if require_attempt else TurnStatus.invalid_submission,
                    state,
                    "review_without_query",
                )
            store.save_review(run_id, payload, count, findings_id, result.submission_action_id)
        return after_turn(result.status, state, result.detail)

    def verify(state: RunState) -> RunState:
        latest = latest_findings(store, run_id)
        assert latest is not None
        findings_id, _, payload = latest
        findings = InvestigationFindings.model_validate(payload)
        reviews = [entry for entry in store.reviews(run_id) if entry["findings_id"] == findings_id]
        result = services.verifier.verify(findings, reviews[-1] if reviews else None)
        store.save_verification(run_id, findings_id, result.passed, result.model_dump(mode="json"))
        return {"last_status": "verified" if result.passed else "verification_failed"}

    def revise(state: RunState) -> RunState:
        latest = latest_findings(store, run_id)
        assert latest is not None
        findings_id = latest[0]
        if any(
            event["payload"]["findings_id"] == findings_id
            for event in _events(store, run_id, "revision_requested")
        ):
            return {"last_status": "revision_requested"}
        verification = [
            entry for entry in store.verifications(run_id) if entry["findings_id"] == findings_id
        ]
        reviews = [entry for entry in store.reviews(run_id) if entry["findings_id"] == findings_id]
        feedback: dict[str, Any] = {"status": "revision requested"}
        if verification and not verification[-1]["passed"]:
            failed = VerificationResult.model_validate(verification[-1]["result"]).failed_checks
            feedback["verification_errors"] = [check.model_dump(mode="json") for check in failed]
        if reviews and reviews[-1]["review"]["decision"] == ReviewDecision.request_more_evidence:
            review_payload = reviews[-1]["review"]
            feedback["reviewer"] = {
                "disputed_claim_ids": review_payload["disputed_claim_ids"],
                "requested_queries": review_payload["requested_queries"],
                "alternative_checks": review_payload["alternative_checks"],
                "reason_summary": review_payload["reason_summary"],
            }
            if review_payload.get("disputed_finding_paths"):
                feedback["reviewer"]["disputed_finding_paths"] = review_payload[
                    "disputed_finding_paths"
                ]
        pending = pending_tool_calls(store.messages(run_id, Role.investigator.value))
        assert pending, "revision needs the pending submit_findings call"
        store.append_message(
            run_id,
            Role.investigator.value,
            {"role": "user", "content": [tool_result_block(pending[0]["id"], feedback, False)]},
        )
        store.record_event(
            run_id,
            "revision_requested",
            {"findings_id": findings_id, "feedback_sha256": content_hash(feedback)},
        )
        return {"last_status": "revision_requested"}

    def render(state: RunState) -> RunState:
        services.render(run_id, state.get("terminal_reason", ""))
        return {"last_status": "rendered"}

    # Routing

    def route_after_investigate(state: RunState) -> str:
        if state.get("terminal_reason"):
            return "render"
        if state["last_status"] != TurnStatus.submitted.value:
            return "investigate"
        if services.arm == Arm.investigator_raw:
            return "render"
        if services.arm == Arm.investigator_reviewer_gated:
            reviews_done = len(store.reviews(run_id))
            if reviews_done < services.maximum_review_rounds:
                return "review"
        return "verify"

    def route_after_review(state: RunState) -> str:
        if state.get("terminal_reason"):
            return "render"
        if state["last_status"] == TurnStatus.submitted.value:
            return "verify"
        return "review"

    def route_after_verify(state: RunState) -> str:
        if revision_count(store, run_id) >= services.maximum_revisions:
            return "render"
        latest = latest_findings(store, run_id)
        assert latest is not None
        reviews = [entry for entry in store.reviews(run_id) if entry["findings_id"] == latest[0]]
        wants_evidence = (
            services.arm == Arm.investigator_reviewer_gated
            and bool(reviews)
            and reviews[-1]["review"]["decision"] == ReviewDecision.request_more_evidence
        )
        if state["last_status"] == "verification_failed" or wants_evidence:
            return "revise"
        return "render"

    graph = StateGraph(RunState)
    graph.add_node("investigate", investigate)
    graph.add_node("review", review)
    graph.add_node("verify", verify)
    graph.add_node("revise", revise)
    graph.add_node("render", render)
    graph.add_edge(START, "investigate")
    graph.add_conditional_edges(
        "investigate", route_after_investigate, ["investigate", "review", "verify", "render"]
    )
    graph.add_conditional_edges("review", route_after_review, ["review", "verify", "render"])
    graph.add_conditional_edges("verify", route_after_verify, ["revise", "render"])
    graph.add_edge("revise", "investigate")
    graph.add_edge("render", END)
    return graph
