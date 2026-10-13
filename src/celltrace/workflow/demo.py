"""Offline end to end demonstration on the original synthetic fixture.

fixture archive -> ingest -> detect -> tools -> investigator -> reviewer ->
verifier -> report, with scripted policies. Every report says
``scripted_fixture``. The demo checks its own expectations and fails loudly
when a scenario does not behave as declared.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from celltrace.agents.scripted_policies import ScriptedInvestigator, ScriptedReviewer
from celltrace.analysis.detector import run_detect
from celltrace.clock import SystemClock
from celltrace.config import AppConfig, load_config
from celltrace.contracts.common import Arm
from celltrace.contracts.evidence import CandidateEvent
from celltrace.contracts.report import InvestigationReport
from celltrace.data.fixture import (
    RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
    STABLE_RADIO_DROP_MEMBER,
    write_fixture_archive,
)
from celltrace.data.snapshot import open_snapshot, run_ingest
from celltrace.providers.base import FailureKind
from celltrace.providers.scripted import ScriptedModelAdapter
from celltrace.storage.run_store import RunStore
from celltrace.tools.faults import failing_dispatcher_factory
from celltrace.workflow.runner import InvestigationRunner, RunOptions, export_run

DEFAULT_PRIVATE_DIRECTORY = Path("artifacts/private/demo")


# Live experiment input allowance, chosen after measured truncation failures.
LIVE_EXPERIMENT_INPUT_TOKENS = 240_000


@dataclass
class Scenario:
    name: str
    purpose: str
    member: str
    arm: Arm
    investigator: ScriptedInvestigator
    reviewer: ScriptedReviewer
    expectations: dict[str, Any]
    overrides: dict[tuple[str, ...], Any] = field(default_factory=dict)
    provider_failures: dict[int, FailureKind] = field(default_factory=dict)
    failing_tools: set[str] = field(default_factory=set)


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="observation_recovered_after_correction",
            purpose=(
                "Radio drop with a valid cell change. The first findings contain a wrong number "
                "and an invented evidence id. Code rejects the wrong claims and the revised "
                "findings pass within the live input allowance. The 40000 "
                "token default cannot fit a correction round once schemas show their limits."
            ),
            member=RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
            arm=Arm.investigator_gated,
            investigator=ScriptedInvestigator(corrupt_first_submission=True),
            reviewer=ScriptedReviewer(),
            overrides={("runtime", "maximum_input_tokens"): LIVE_EXPERIMENT_INPUT_TOKENS},
            expectations={
                "workflow_outcome": "completed",
                "fact_check_status": "passed",
                "review_status": "not_run",
                "cause_status": "unresolved",
                "first_verification_failed": True,
                "has_transition_claim": True,
            },
        ),
        Scenario(
            name="reviewed_correction_stops_at_input_limit",
            purpose=(
                "The reviewer queries evidence and asks for uplink context. Code rejects the "
                "initial wrong claims. Revision is routed but the shared input allowance "
                "prevents another provider attempt. Verified facts remain visible."
            ),
            member=RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
            arm=Arm.investigator_reviewer_gated,
            investigator=ScriptedInvestigator(corrupt_first_submission=True),
            reviewer=ScriptedReviewer(request_more_in_first_round=True),
            expectations={
                "workflow_outcome": "exhausted",
                "fact_check_status": "failed",
                "review_status": "independently_checked",
                "cause_status": "unresolved",
                "first_verification_failed": True,
                "terminal_reason_prefix": "budget_exhausted:input_tokens",
            },
        ),
        Scenario(
            name="stable_radio_rate_drop_unresolved",
            purpose=(
                "Active rate drop while reported RSRP stays within 1 dB. Facts pass, the cause "
                "stays unresolved because load and transport are not measured."
            ),
            member=STABLE_RADIO_DROP_MEMBER,
            arm=Arm.investigator_reviewer_gated,
            investigator=ScriptedInvestigator(),
            reviewer=ScriptedReviewer(),
            expectations={
                "workflow_outcome": "completed",
                "fact_check_status": "passed",
                "review_status": "independently_checked",
                "cause_status": "unresolved",
                "pattern": "stable_reported_radio",
            },
        ),
        Scenario(
            name="raw_arm_keeps_wrong_claims",
            purpose=(
                "Arm A on the same event with the same wrong first findings. Nothing checks the "
                "claims during the run, so they are reported as unverified."
            ),
            member=RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
            arm=Arm.investigator_raw,
            investigator=ScriptedInvestigator(corrupt_first_submission=True),
            reviewer=ScriptedReviewer(),
            expectations={
                "workflow_outcome": "completed",
                "fact_check_status": "not_run",
                "report_basis": "raw_unverified_findings",
            },
        ),
        Scenario(
            name="tool_and_provider_failure",
            purpose=(
                "compare_windows always fails with STORE_UNAVAILABLE and the provider then "
                "returns two overload errors. The run stops with a bounded retry and reports "
                "only facts code could verify from the evidence it did collect."
            ),
            member=RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
            arm=Arm.investigator_gated,
            investigator=ScriptedInvestigator(),
            reviewer=ScriptedReviewer(),
            failing_tools={"compare_windows"},
            provider_failures={6: FailureKind.transient, 7: FailureKind.transient},
            expectations={
                "workflow_outcome": "partial",
                "report_basis": "deterministic_fallback",
                "terminal_reason_prefix": "provider_failed",
            },
        ),
        Scenario(
            name="budget_exhausted",
            purpose="Three tool requests allowed. The run stops honestly as exhausted.",
            member=STABLE_RADIO_DROP_MEMBER,
            arm=Arm.investigator_reviewer_gated,
            investigator=ScriptedInvestigator(),
            reviewer=ScriptedReviewer(),
            overrides={("runtime", "maximum_tool_requests"): 3},
            expectations={"workflow_outcome": "exhausted", "terminal_reason_prefix": "budget"},
        ),
    ]


def _check(report: InvestigationReport, store: RunStore, scenario: Scenario) -> list[str]:
    problems = []
    expected = scenario.expectations
    for key in ("workflow_outcome", "fact_check_status", "review_status", "cause_status"):
        if key in expected and getattr(report, key).value != expected[key]:
            problems.append(f"{key} was {getattr(report, key).value}, expected {expected[key]}")
    if "report_basis" in expected and report.report_basis.value != expected["report_basis"]:
        problems.append(f"report_basis was {report.report_basis.value}")
    if "terminal_reason_prefix" in expected and not report.terminal_reason.startswith(
        expected["terminal_reason_prefix"]
    ):
        problems.append(f"terminal_reason was {report.terminal_reason}")
    if expected.get("first_verification_failed"):
        verifications = store.verifications(report.run_id)
        if not verifications or verifications[0]["passed"]:
            problems.append("the first verification did not reject the wrong claims")
    if expected.get("has_transition_claim") and not any(
        claim.claim_type == "transition" for claim in report.verified_claims
    ):
        problems.append("no verified transition claim")
    if "pattern" in expected and expected["pattern"] not in report.observation_patterns:
        problems.append(f"pattern {expected['pattern']} missing")
    if report.execution_mode.value != "scripted_fixture":
        problems.append("demo report is not labeled scripted_fixture")
    return problems


def _demo_config(private_directory: Path, overrides: dict[tuple[str, ...], Any]) -> AppConfig:
    base: dict[tuple[str, ...], Any] = {
        ("runtime", "run_store"): str(private_directory / "runs.sqlite3"),
        ("runtime", "checkpoint_store"): str(private_directory / "checkpoints.sqlite3"),
    }
    return load_config(cli_overrides={**base, **overrides}, environment={})


def run_offline_demo(
    out_dir: Path, private_directory: Path = DEFAULT_PRIVATE_DIRECTORY
) -> dict[str, Any]:
    execution_id = f"execution_{uuid.uuid4().hex[:16]}"
    private_directory = private_directory / execution_id
    out_dir = out_dir / execution_id
    private_directory.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = private_directory / "raw"
    write_fixture_archive(archive_dir)
    snapshot_dir = run_ingest(archive_dir, private_directory / "processed")
    base_config = _demo_config(private_directory, {})
    candidates_path = run_detect(
        snapshot_dir, out_dir / "candidates", base_config.analysis.detector
    )
    candidates = [
        CandidateEvent.model_validate(entry)
        for entry in json.loads(candidates_path.read_text())["candidates"]
    ]
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        session_by_member = {
            str(row[1]): str(row[0])
            for row in connection.execute(
                "SELECT session_id, source_relative_path FROM sessions"
            ).fetchall()
        }
    finally:
        connection.close()

    results: list[dict[str, Any]] = []
    for scenario in scenarios():
        session_id = session_by_member[scenario.member]
        session_candidates = sorted(
            (candidate for candidate in candidates if candidate.session_id == session_id),
            key=lambda candidate: candidate.first_confirmation_ms,
        )
        if not session_candidates:
            results.append(
                {"scenario": scenario.name, "status": "failed", "problems": ["no candidate"]}
            )
            continue
        config = _demo_config(private_directory, scenario.overrides)
        adapter = ScriptedModelAdapter(
            {"investigator": scenario.investigator, "reviewer": scenario.reviewer},
            failures=scenario.provider_failures,
        )
        dispatcher_factory = (
            failing_dispatcher_factory(scenario.failing_tools) if scenario.failing_tools else None
        )
        runner = InvestigationRunner(
            config,
            adapter,
            clock=SystemClock(),
            executor_options={"sleep": lambda seconds: None},
            dispatcher_factory=dispatcher_factory,
        )
        try:
            report = runner.start(
                snapshot_dir,
                session_candidates[0],
                RunOptions(
                    arm=scenario.arm,
                    run_id=f"run_demo_{scenario.name}",
                    case_id=f"demo_{scenario.name}",
                    report_directory=out_dir / "reports",
                ),
            )
            problems = _check(report, runner.store, scenario)
            export_run(runner.store, report.run_id, private_directory / "exports")
        finally:
            runner.close()
        results.append(
            {
                "scenario": scenario.name,
                "purpose": scenario.purpose,
                "arm": scenario.arm.value,
                "event_id": session_candidates[0].event_id,
                "run_id": report.run_id,
                "execution_mode": report.execution_mode.value,
                "workflow_outcome": report.workflow_outcome.value,
                "fact_check_status": report.fact_check_status.value,
                "review_status": report.review_status.value,
                "cause_status": report.cause_status.value,
                "report_basis": report.report_basis.value,
                "terminal_reason": report.terminal_reason,
                "verified_claims": len(report.verified_claims),
                "rejected_claims": len(report.rejected_claims),
                "tool_requests": report.usage.tool_requests,
                "provider_attempts": report.usage.provider_attempts,
                "report_markdown": str(out_dir / "reports" / f"{report.run_id}.md"),
                "status": "passed" if not problems else "failed",
                "problems": problems,
            }
        )
    summary = {
        "label": "OFFLINE SCRIPTED FIXTURE DEMO. Not live model behavior and not real data.",
        "snapshot_id": manifest["snapshot_id"],
        "fixture_source": "celltrace.data.fixture (original synthetic values)",
        "candidates": str(candidates_path),
        "candidate_count": len(candidates),
        "scenarios": results,
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
    }
    (out_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
