"""Graph behavior, budgets, provider failures and crash recovery on the fixture."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from celltrace.agents.scripted_policies import (
    ScriptedInvestigator,
    ScriptedReviewer,
    build_findings,
)
from celltrace.clock import ManualClock
from celltrace.contracts.common import Arm
from celltrace.data.fixture import RADIO_DROP_WITH_CELL_CHANGE_MEMBER, STABLE_RADIO_DROP_MEMBER
from celltrace.providers.base import (
    BYTE_ESTIMATE_METHOD,
    FailureKind,
    ProviderRequest,
    TokenEstimate,
)
from celltrace.providers.scripted import ScriptedModelAdapter, ScriptedTurn
from celltrace.storage.lock import RunLock, RunLockedError
from celltrace.storage.run_store import ActionKind, ActionState
from celltrace.workflow.demo import LIVE_EXPERIMENT_INPUT_TOKENS
from celltrace.workflow.runner import InvestigationRunner, ResumeRefused, RunOptions

from .conftest import FixtureSnapshot, make_config

CORRECTION_ALLOWANCE = {("runtime", "maximum_input_tokens"): LIVE_EXPERIMENT_INPUT_TOKENS}


class SimulatedCrash(BaseException):
    """BaseException so no application handler can swallow it."""


@dataclass
class CrashAt:
    point: str
    occurrence: int
    seen: int = 0

    def __call__(self, point: str) -> None:
        if point == self.point:
            self.seen += 1
            if self.seen == self.occurrence:
                raise SimulatedCrash(point)


@dataclass
class AlwaysCorruptInvestigator(ScriptedInvestigator):
    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        turn = super().next_turn(request)
        if turn.tool_name == "submit_findings":
            from celltrace.agents.scripted_policies import _candidate

            return ScriptedTurn(
                turn.text,
                "submit_findings",
                build_findings(_candidate(request.messages), request.messages, corrupt=True),
            )
        return turn


@dataclass
class HastyReviewer(ScriptedReviewer):
    """Accepts without any query on its first turn, then behaves like the scripted reviewer."""

    hasty_turns: list[int] = field(default_factory=list)

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        if not self.hasty_turns:
            self.hasty_turns.append(1)
            return ScriptedTurn(
                "Looks fine.",
                "submit_review",
                {"decision": "accept", "reason_summary": "No check made."},
            )
        return super().next_turn(request)


@dataclass
class HastySecondReviewer(ScriptedReviewer):
    skipped: bool = False

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        submitted = sum(
            block.get("name") == "submit_review"
            for message in request.messages
            for block in message["content"]
            if block.get("type") == "tool_use"
        )
        if submitted == 1 and not self.skipped:
            self.skipped = True
            return ScriptedTurn(
                "Earlier checks suffice.",
                "submit_review",
                {
                    "decision": "accept",
                    "reason_summary": "No new measurement in this round.",
                    "review_limitation": "Physical cause telemetry is missing.",
                },
            )
        return super().next_turn(request)


def runner_for(
    tmp_path: Path,
    investigator: Any = None,
    reviewer: Any = None,
    failures: dict[int, FailureKind] | None = None,
    fault_hook: Any = None,
    overrides: dict[tuple[str, ...], Any] | None = None,
) -> InvestigationRunner:
    adapter = ScriptedModelAdapter(
        {
            "investigator": investigator or ScriptedInvestigator(),
            "reviewer": reviewer or ScriptedReviewer(),
        },
        failures=failures,
    )
    options: dict[str, Any] = {"sleep": lambda seconds: None}
    if fault_hook is not None:
        options["fault_hook"] = fault_hook
    return InvestigationRunner(
        make_config(tmp_path, overrides),
        adapter,
        clock=ManualClock(),
        executor_options=options,
    )


def start(
    runner: InvestigationRunner, fixture: FixtureSnapshot, member: str, arm: Arm, run_id: str
) -> Any:
    return runner.start(
        fixture.snapshot_dir,
        fixture.first_candidate(member),
        RunOptions(arm=arm, run_id=run_id),
    )


def counts(runner: InvestigationRunner, run_id: str) -> tuple[int, int, int]:
    usage = runner.store.usage(run_id)
    evidence = len(runner.store.run_evidence(run_id))
    return usage.provider_attempts, usage.tool_requests, evidence


# Arms and roles


def test_final_round_cannot_relabel_earlier_queries_as_independent_review(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(
        tmp_path,
        reviewer=HastySecondReviewer(request_more_in_first_round=True),
        overrides=CORRECTION_ALLOWANCE
        | {
            ("runtime", "require_review_measurement"): True,
            ("runtime", "maximum_tool_requests"): 24,
            ("runtime", "maximum_provider_attempts"): 20,
            ("runtime", "show_remaining_budget"): True,
        },
    )
    try:
        report = start(
            runner,
            fixture_snapshot,
            STABLE_RADIO_DROP_MEMBER,
            Arm.investigator_reviewer_gated,
            "final_query",
        )
        assert report.workflow_outcome.value == "completed"
        reviews = runner.store.reviews("final_query")
        assert len(reviews) == 2 and all(r["independent_query_count"] > 0 for r in reviews)
        assert any(
            "generic data gaps" in json.dumps(m)
            for m in runner.store.messages("final_query", "reviewer")
        )
        calls = runner.store.raw_actions("final_query")
        recorded = [
            json.loads(a["request_json"]).get("resource_budget")
            for a in calls
            if a["kind"] == "provider_attempt"
        ]
        assert all(r is not None for r in recorded)
        assert (
            recorded[-1]["resources"]["provider_attempts"]["used"]
            > recorded[0]["resources"]["provider_attempts"]["used"]
        )
    finally:
        runner.close()


def test_reviewed_run_has_separate_contexts_and_an_independent_query(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(tmp_path)
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_reviewer_gated, "run_c"
    )
    assert (report.workflow_outcome.value, report.fact_check_status.value) == (
        "completed",
        "passed",
    )
    assert report.review_status.value == "independently_checked"
    assert report.cause_status.value == "unresolved"
    investigator_messages = runner.store.messages("run_c", "investigator")
    reviewer_messages = runner.store.messages("run_c", "reviewer")
    reviewer_text = json.dumps(reviewer_messages)
    # The reviewer sees findings, never the investigator's tool transcript.
    assert "get_session_context" not in reviewer_text
    assert "Findings:" in reviewer_messages[0]["content"][0]["text"]
    assert "Findings:" not in json.dumps(investigator_messages)
    reviewer_tools = [
        action
        for action in runner.store.actions("run_c", ActionKind.tool_request)
        if action.role == "reviewer"
    ]
    assert reviewer_tools and reviewer_tools[0].tool_name == "compare_windows"
    assert runner.store.reviews("run_c")[0]["independent_query_count"] >= 1
    runner.close()


@dataclass
class PromptRecordingInvestigator(ScriptedInvestigator):
    prompts: list[str] = field(default_factory=list)

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        self.prompts.append(request.system_prompt)
        return super().next_turn(request)


@dataclass
class PromptRecordingReviewer(ScriptedReviewer):
    prompts: list[str] = field(default_factory=list)

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        self.prompts.append(request.system_prompt)
        return super().next_turn(request)


def test_both_roles_are_told_the_configured_run_limits(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    investigator = PromptRecordingInvestigator()
    reviewer = PromptRecordingReviewer()
    runner = runner_for(
        tmp_path,
        investigator=investigator,
        reviewer=reviewer,
        overrides={("runtime", "maximum_tool_requests"): 21},
    )
    start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_reviewer_gated, "run_l"
    )
    expected = "Run limits: 21 tool requests and 12 model calls in total for all roles."
    assert investigator.prompts and reviewer.prompts
    assert all(prompt.endswith(expected) for prompt in investigator.prompts + reviewer.prompts)
    assert len(set(investigator.prompts)) == 1
    runner.close()


class ByteEstimateAdapter(ScriptedModelAdapter):
    """Reports a byte bound twice the usage it later bills, like the live endpoints."""

    def estimate_input_tokens(self, request: ProviderRequest) -> TokenEstimate:
        scripted = super().estimate_input_tokens(request)
        return TokenEstimate(scripted.input_tokens * 2, BYTE_ESTIMATE_METHOD)

    def complete(self, request: ProviderRequest) -> Any:
        response = super().complete(request)
        actual = ScriptedModelAdapter.estimate_input_tokens(self, request).input_tokens
        return replace(response, usage=replace(response.usage, input_tokens=actual))


def test_input_reservation_is_calibrated_by_the_same_role(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    adapter = ByteEstimateAdapter(
        {"investigator": ScriptedInvestigator(), "reviewer": ScriptedReviewer()}
    )
    runner = InvestigationRunner(
        make_config(tmp_path),
        adapter,
        clock=ManualClock(),
        executor_options={"sleep": lambda seconds: None},
    )
    start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_reviewer_gated, "run_e"
    )
    attempts = [
        row for row in runner.store.raw_actions("run_e") if row["kind"] == "provider_attempt"
    ]
    seen_roles: set[str] = set()
    for row in attempts:
        byte_estimate = json.loads(row["request_json"])["byte_estimate"]
        if row["role"] not in seen_roles:
            assert row["reserved_input_tokens"] == byte_estimate
            assert row["estimation_method"] == BYTE_ESTIMATE_METHOD
        else:
            # Billed usage is half the bound, so the reservation keeps a 25 percent margin.
            assert row["reserved_input_tokens"] == math.ceil(byte_estimate * 0.625)
            assert row["estimation_method"].endswith("calibrated_by_last_role_attempt")
        seen_roles.add(row["role"])
    assert seen_roles == {"investigator", "reviewer"}
    runner.close()


@dataclass
class TruncatedFirstSubmission(ScriptedInvestigator):
    truncated: list[int] = field(default_factory=list)

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        turn = super().next_turn(request)
        if turn.tool_name == "submit_findings" and not self.truncated:
            self.truncated.append(1)
            return ScriptedTurn(
                turn.text,
                "submit_findings",
                {"invalid_provider_arguments": "expected a finite JSON object"},
            )
        return turn


def test_cut_off_submission_is_reported_as_incomplete_json(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(
        tmp_path, investigator=TruncatedFirstSubmission(), overrides=CORRECTION_ALLOWANCE
    )
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_t"
    )
    investigator_text = json.dumps(runner.store.messages("run_t", "investigator"))
    assert "not a complete JSON object" in investigator_text
    assert "Field required" not in investigator_text
    assert report.workflow_outcome.value == "completed"
    runner.close()


def test_reviewer_acceptance_cannot_waive_failed_facts(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(tmp_path, investigator=AlwaysCorruptInvestigator())
    report = start(
        runner,
        fixture_snapshot,
        RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
        Arm.investigator_reviewer_gated,
        "run_waive",
    )
    assert report.review is not None and report.review.decision.value == "accept"
    assert report.fact_check_status.value == "failed"
    assert report.workflow_outcome.value == "exhausted"
    assert report.terminal_reason == "budget_exhausted:input_tokens"
    rejected_codes = {entry["check"]["code"] for entry in report.rejected_claims}
    assert {"VALUE_MISMATCH", "UNKNOWN_EVIDENCE"} <= rejected_codes
    assert all(claim.claim_id != "invented_statistic" for claim in report.verified_claims)
    runner.close()


def test_acceptance_without_a_query_is_sent_back(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    # A correction round needs more than the 40000 token default once schemas show limits.
    runner = runner_for(tmp_path, reviewer=HastyReviewer(), overrides=CORRECTION_ALLOWANCE)
    report = start(
        runner,
        fixture_snapshot,
        STABLE_RADIO_DROP_MEMBER,
        Arm.investigator_reviewer_gated,
        "run_hasty",
    )
    reviewer_text = json.dumps(runner.store.messages("run_hasty", "reviewer"))
    assert "accept requires at least one independent measurement" in reviewer_text
    assert runner.store.reviews("run_hasty")[0]["independent_query_count"] >= 1
    assert report.review_status.value == "independently_checked"
    runner.close()


def test_raw_arm_never_runs_the_verifier(fixture_snapshot: FixtureSnapshot, tmp_path: Path) -> None:
    runner = runner_for(tmp_path, investigator=ScriptedInvestigator(corrupt_first_submission=True))
    report = start(
        runner, fixture_snapshot, RADIO_DROP_WITH_CELL_CHANGE_MEMBER, Arm.investigator_raw, "run_a"
    )
    assert runner.store.verifications("run_a") == []
    assert report.fact_check_status.value == "not_run"
    assert any(claim.claim_id == "invented_statistic" for claim in report.unverified_claims)
    assert report.verified_claims == []
    runner.close()


def test_gated_arm_corrects_after_code_feedback(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(
        tmp_path,
        investigator=ScriptedInvestigator(corrupt_first_submission=True),
        overrides=CORRECTION_ALLOWANCE,
    )
    report = start(
        runner,
        fixture_snapshot,
        RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
        Arm.investigator_gated,
        "run_b",
    )
    assert [entry["passed"] for entry in report.verification_history] == [False, True]
    assert report.fact_check_status.value == "passed"
    assert report.usage.revision_rounds == 1
    assert report.initial_findings is not None and report.final_findings is not None
    assert len(report.initial_findings.factual_claims) != len(report.final_findings.factual_claims)
    runner.close()


# Budgets and provider failures


def test_budget_exhaustion_is_terminal(fixture_snapshot: FixtureSnapshot, tmp_path: Path) -> None:
    runner = runner_for(tmp_path, overrides={("runtime", "maximum_tool_requests"): 2})
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_budget"
    )
    assert report.workflow_outcome.value == "exhausted"
    assert report.usage.tool_requests == 2
    assert runner.store.get_run("run_budget")["status"] == "terminal"
    runner.close()


def test_reviewer_cut_off_by_the_budget_is_not_reported_as_never_started(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    # Expert audit: a live run whose Reviewer spent 12 tool requests reported only not_run.
    (tmp_path / "probe").mkdir()
    (tmp_path / "cut").mkdir()
    probe = runner_for(tmp_path / "probe")
    start(probe, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_reviewer_gated, "p")
    tools = probe.store.actions("p", ActionKind.tool_request)
    investigator_tools = sum(action.role == "investigator" for action in tools)
    assert sum(action.role == "reviewer" for action in tools) >= 1
    probe.close()
    runner = runner_for(
        tmp_path / "cut", overrides={("runtime", "maximum_tool_requests"): investigator_tools}
    )
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_reviewer_gated, "c"
    )
    assert report.workflow_outcome.value == "exhausted"
    assert report.review_status.value == "not_run"
    cut_off = "The Reviewer started (0 tool requests"
    assert any(note.startswith(cut_off) for note in report.limitations)
    assert runner.store.reviews("c") == []
    runner.close()


def test_transient_failure_is_retried_once_and_both_attempts_are_charged(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(tmp_path, failures={2: FailureKind.transient})
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_retry"
    )
    attempts = runner.store.actions("run_retry", ActionKind.provider_attempt)
    assert [action.state for action in attempts[:3]] == [
        ActionState.completed,
        ActionState.failed,
        ActionState.completed,
    ]
    assert attempts[1].logical_key == attempts[2].logical_key
    assert report.workflow_outcome.value == "completed"
    runner.close()


def test_repeated_transient_failures_stop_after_the_retry_limit(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(tmp_path, failures={2: FailureKind.transient, 3: FailureKind.transient})
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_stop"
    )
    assert report.terminal_reason.startswith("provider_failed")
    assert report.usage.provider_attempts == 3
    runner.close()


def test_permanent_failure_is_not_retried(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(tmp_path, failures={1: FailureKind.permanent})
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_permanent"
    )
    assert report.usage.provider_attempts == 1
    assert report.workflow_outcome.value == "failed"
    runner.close()


def test_uncertain_failure_keeps_reservation_and_records_its_monotonic_duration(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(tmp_path, failures={2: FailureKind.uncertain})
    report = start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_lost"
    )
    assert report.terminal_reason.startswith("provider_uncertain")
    assert report.usage.provider_attempts == 2
    uncertain = runner.store.raw_actions("run_lost")
    lost = [action for action in uncertain if action["state"] == "uncertain"]
    assert len(lost) == 1 and lost[0]["usage_known"] == 0
    # The manual clock does not advance, so the recorded duration is zero rather than missing.
    assert lost[0]["duration_ms"] == 0
    runner.close()


# Crash recovery


def baseline(fixture_snapshot: FixtureSnapshot, tmp_path: Path) -> tuple[int, int, int]:
    runner = runner_for(tmp_path / "baseline")
    start(runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_base")
    result = counts(runner, "run_base")
    runner.close()
    return result


def crash_then_resume(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path, hook: CrashAt, run_id: str
) -> tuple[InvestigationRunner, Any]:
    crashing = runner_for(tmp_path, fault_hook=hook)
    with pytest.raises(SimulatedCrash):
        start(crashing, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, run_id)
    crashing.close()
    resumed = runner_for(tmp_path)
    return resumed, resumed.resume(run_id)


def test_crash_after_reservation_before_dispatch(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    attempts, tools, evidence = baseline(fixture_snapshot, tmp_path)
    runner, report = crash_then_resume(
        fixture_snapshot, tmp_path, CrashAt("after_provider_reservation", 3), "run_crash_reserved"
    )
    assert report.workflow_outcome.value == "completed"
    # The undispatched reservation stays counted and is marked, nothing is repeated.
    assert counts(runner, "run_crash_reserved") == (attempts + 1, tools, evidence)
    events = [event["event_type"] for event in runner.store.run_events("run_crash_reserved")]
    assert "reservation_not_dispatched" in events
    runner.close()


def test_crash_after_persisted_result_reuses_it(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    attempts, tools, evidence = baseline(fixture_snapshot, tmp_path)
    runner, report = crash_then_resume(
        fixture_snapshot, tmp_path, CrashAt("after_provider_result", 3), "run_crash_result"
    )
    assert report.workflow_outcome.value == "completed"
    assert counts(runner, "run_crash_result") == (attempts, tools, evidence)
    runner.close()


def test_crash_after_tool_result_before_message(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    attempts, tools, evidence = baseline(fixture_snapshot, tmp_path)
    runner, report = crash_then_resume(
        fixture_snapshot, tmp_path, CrashAt("after_tool_result", 2), "run_crash_tool"
    )
    assert report.workflow_outcome.value == "completed"
    assert counts(runner, "run_crash_tool") == (attempts, tools, evidence)
    events = [event["event_type"] for event in runner.store.run_events("run_crash_tool")]
    assert "tool_result_replayed" in events
    runner.close()


def test_crash_before_checkpoint_commit(fixture_snapshot: FixtureSnapshot, tmp_path: Path) -> None:
    attempts, tools, evidence = baseline(fixture_snapshot, tmp_path)
    runner, report = crash_then_resume(
        fixture_snapshot, tmp_path, CrashAt("end_of_turn", 3), "run_crash_checkpoint"
    )
    assert report.workflow_outcome.value == "completed"
    assert counts(runner, "run_crash_checkpoint") == (attempts, tools, evidence)
    runner.close()


def test_crash_during_provider_call_is_uncertain_and_needs_explicit_retry(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    attempts, _, _ = baseline(fixture_snapshot, tmp_path)
    runner, report = crash_then_resume(
        fixture_snapshot, tmp_path, CrashAt("during_provider_call", 3), "run_uncertain"
    )
    assert report.terminal_reason.startswith("provider_uncertain")
    # Facts verified before the stop are kept through the deterministic fallback.
    assert report.workflow_outcome.value == "partial"
    assert report.report_basis.value == "deterministic_fallback"
    usage_before = runner.store.usage("run_uncertain")
    assert usage_before.unknown_usage_attempts == 1
    # A plain resume does not resubmit the uncertain attempt.
    again = runner.resume("run_uncertain")
    assert again.terminal_reason == report.terminal_reason
    assert runner.store.usage("run_uncertain").provider_attempts == usage_before.provider_attempts
    retried = runner.resume("run_uncertain", retry_uncertain=True)
    assert retried.workflow_outcome.value == "completed"
    final_usage = runner.store.usage("run_uncertain")
    assert final_usage.provider_attempts == attempts + 1
    assert final_usage.unknown_usage_attempts == 1
    assert final_usage.input_tokens > 0
    runner.close()


def test_resume_refuses_changed_identity_and_wrong_snapshot(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    crashing = runner_for(tmp_path, fault_hook=CrashAt("end_of_turn", 1))
    with pytest.raises(SimulatedCrash):
        start(
            crashing,
            fixture_snapshot,
            STABLE_RADIO_DROP_MEMBER,
            Arm.investigator_gated,
            "run_identity",
        )
    crashing.close()
    changed = runner_for(tmp_path, overrides={("runtime", "maximum_tool_requests"): 5})
    with pytest.raises(ResumeRefused, match="runtime_limits_hash"):
        changed.resume("run_identity")
    changed.close()
    changed_model = runner_for(tmp_path, overrides={("runtime", "model_id"): "z-ai/glm-5.3-flash"})
    with pytest.raises(ResumeRefused, match="runtime_limits_hash"):
        changed_model.resume("run_identity")
    changed_model.close()
    other_snapshot = tmp_path / "other_snapshot"
    shutil.copytree(fixture_snapshot.snapshot_dir, other_snapshot)
    manifest_path = other_snapshot / "snapshot_manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["logical_hash"] = "1" * 64
    manifest["snapshot_id"] = "snap_" + "1" * 16
    manifest_path.write_text(json.dumps(manifest))
    same_config = runner_for(tmp_path)
    with pytest.raises(ResumeRefused, match="snapshot"):
        same_config.resume("run_identity", snapshot_dir=other_snapshot)
    same_config.close()


def test_a_second_owner_cannot_resume_a_locked_run(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    crashing = runner_for(tmp_path, fault_hook=CrashAt("end_of_turn", 1))
    with pytest.raises(SimulatedCrash):
        start(
            crashing,
            fixture_snapshot,
            STABLE_RADIO_DROP_MEMBER,
            Arm.investigator_gated,
            "run_locked",
        )
    crashing.close()
    lock_directory = Path(make_config(tmp_path).runtime.run_store).parent / "locks"
    with RunLock(lock_directory, "run_locked"):
        second = runner_for(tmp_path)
        with pytest.raises(RunLockedError):
            second.resume("run_locked")
        second.close()


def test_instruction_text_in_data_is_returned_as_a_value(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    from celltrace.data.fixture import NETWORK_MODE_INSTRUCTION_TEXT

    runner = runner_for(tmp_path)
    start(
        runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_gated, "run_injection"
    )
    stored = runner.store.get_run("run_injection")["context"]
    # The run context and tool definitions are code owned and identical whatever the data says.
    assert NETWORK_MODE_INSTRUCTION_TEXT not in json.dumps(stored)
    connection = __import__("duckdb").connect(
        str(fixture_snapshot.snapshot_dir / "measurements.duckdb"), read_only=True
    )
    try:
        modes = {
            row[0]
            for row in connection.execute("SELECT DISTINCT network_mode FROM samples").fetchall()
        }
    finally:
        connection.close()
    assert NETWORK_MODE_INSTRUCTION_TEXT in modes
    runner.close()


def test_transition_after_onset_gets_a_chronology_warning(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(tmp_path)
    report = start(
        runner,
        fixture_snapshot,
        RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
        Arm.investigator_reviewer_gated,
        "run_chronology",
    )
    onset = fixture_snapshot.first_candidate(
        RADIO_DROP_WITH_CELL_CHANGE_MEMBER
    ).event_window.start_ms
    transition = next(claim for claim in report.verified_claims if claim.claim_type == "transition")
    assert transition.earlier_time_ms >= onset
    assert any(item.startswith("Chronology warning") for item in report.limitations)
    runner.close()


def test_remaining_allowance_is_from_the_ledger_for_each_role(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(
        tmp_path,
        overrides={
            ("runtime", "maximum_tool_requests"): 24,
            ("runtime", "reviewer_tool_reserve"): 4,
            ("runtime", "show_remaining_budget"): True,
            # This test inspects both roles, rather than the old prompt's token
            # size. Fresh multi-resource reminders also consume input tokens.
            ("runtime", "maximum_input_tokens"): 60_000,
        },
    )
    report = start(
        runner,
        fixture_snapshot,
        STABLE_RADIO_DROP_MEMBER,
        Arm.investigator_reviewer_gated,
        "with_reminders",
    )
    assert report.workflow_outcome.value == "completed"
    for role in ("investigator", "reviewer"):
        remaining = []
        for message in runner.store.messages("with_reminders", role):
            for block in message.get("content", []):
                if block.get("type") == "tool_result":
                    allowance = json.loads(block["content"])["remaining_budget"]
                    assert allowance["role"] == role
                    assert allowance["total_used"] + allowance["total_remaining"] == 24
                    if role == "investigator":
                        assert allowance["role_remaining"] <= 20 - allowance["role_used"]
                    remaining.append(allowance["total_remaining"])
        assert remaining and remaining == sorted(remaining, reverse=True)
    runner.close()


def test_completion_admission_cannot_spend_future_review_output(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(
        tmp_path,
        overrides={
            ("runtime", "maximum_input_tokens"): 240_000,
            ("runtime", "maximum_output_tokens_per_attempt"): 11_000,
            ("runtime", "review_round_output_reserve"): 1_000,
            ("runtime", "show_remaining_budget"): True,
        },
    )
    start(
        runner,
        fixture_snapshot,
        STABLE_RADIO_DROP_MEMBER,
        Arm.investigator_reviewer_gated,
        "clamped",
    )
    first = runner.store.actions("clamped", ActionKind.provider_attempt)[0]
    assert first.request["maximum_output_tokens"] == 10_000
    assert (
        runner.store.connection.execute(
            "SELECT reserved_output_tokens FROM actions WHERE action_id=?", (first.action_id,)
        ).fetchone()[0]
        == 10_000
    )
    assert (
        first.request["resource_budget"]["resources"]["output_tokens"]["protected_future"] == 2_000
    )
    assert all(
        "Current ledger budget" not in json.dumps(message)
        for message in runner.store.messages("clamped", "investigator")
    )
    runner.close()
