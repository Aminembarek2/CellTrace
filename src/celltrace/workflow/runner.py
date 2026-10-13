"""Compose and run one investigation: start, resume and export.

Identity check on resume: snapshot, dictionary, prompts, tool schemas,
contract and limits must match the stored run. Any difference refuses the
resume, because continuing would silently turn a historical run into a
different experiment.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from celltrace.agents.roles import investigator_opening, role_identity
from celltrace.canonical import content_hash
from celltrace.clock import Clock, SystemClock
from celltrace.config import AppConfig
from celltrace.contracts.common import CONTRACT_SCHEMA_VERSION, Arm, Role
from celltrace.contracts.evidence import CandidateEvent, RunContext
from celltrace.contracts.report import InvestigationReport
from celltrace.data.snapshot import open_snapshot
from celltrace.providers.base import ModelAdapter
from celltrace.providers.rate_card import RateCard
from celltrace.reporting.build import build_report
from celltrace.reporting.render import render_markdown
from celltrace.storage.lock import RunLock
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    Reservation,
    RunLimits,
    RunStore,
)
from celltrace.tools.context import build_run_context
from celltrace.tools.dispatch import TOOL_VERSION, ToolDispatcher
from celltrace.tools.repository import MeasurementRepository
from celltrace.verification.verifier import Verifier
from celltrace.workflow.executor import RoleTurnExecutor
from celltrace.workflow.graph import GraphServices, build_graph

GRAPH_RECURSION_LIMIT = 250


class ResumeRefused(RuntimeError):
    pass


@dataclass
class RunOptions:
    arm: Arm
    case_id: str | None = None
    run_id: str | None = None
    experiment_id: str | None = None
    report_directory: Path | None = None


def run_limits(config: AppConfig, rate_card: RateCard | None, arm: Arm | None = None) -> RunLimits:
    runtime = config.runtime
    return RunLimits(
        maximum_tool_requests=runtime.maximum_tool_requests,
        maximum_provider_attempts=runtime.maximum_provider_attempts,
        maximum_input_tokens=runtime.maximum_input_tokens,
        maximum_output_tokens=runtime.maximum_output_tokens,
        maximum_estimated_usd=runtime.maximum_estimated_usd_per_run if rate_card else None,
        maximum_active_ms=runtime.maximum_active_seconds * 1000,
        maximum_review_rounds=runtime.maximum_review_rounds,
        maximum_request_retries=runtime.maximum_request_retries,
        reviewer_tool_reserve=runtime.reviewer_tool_reserve
        if arm == Arm.investigator_reviewer_gated
        else 0,
        review_round_tool_reserve=runtime.review_round_tool_reserve
        if arm == Arm.investigator_reviewer_gated
        else 0,
        review_round_provider_reserve=runtime.review_round_provider_reserve
        if arm == Arm.investigator_reviewer_gated
        else 0,
        review_round_input_reserve=runtime.review_round_input_reserve
        if arm == Arm.investigator_reviewer_gated
        else 0,
        review_round_output_reserve=runtime.review_round_output_reserve
        if arm == Arm.investigator_reviewer_gated
        else 0,
    )


def run_identity(
    context: RunContext, config: AppConfig, adapter: ModelAdapter, arm: Arm
) -> dict[str, Any]:
    identity = {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "snapshot_hash": context.snapshot_hash,
        "dictionary_hash": context.dictionary_hash,
        "roles": role_identity(),
        "runtime_limits_hash": content_hash(config.runtime.model_dump(mode="json")),
        "detector_config_hash": content_hash(config.analysis.detector.model_dump(mode="json")),
        "observation_config_hash": content_hash(
            config.analysis.observation.model_dump(mode="json")
        ),
        "execution_mode": adapter.execution_mode.value,
        "model_id": adapter.model_id,
        "arm": arm.value,
    }
    if adapter.execution_mode.value == "live":
        identity["provider"] = adapter.describe()
        identity["rate_card_sha256"] = (
            None
            if config.runtime.verified_rate_card is None
            else content_hash(config.runtime.verified_rate_card.read_text())
        )
    return identity


class InvestigationRunner:
    def __init__(
        self,
        config: AppConfig,
        adapter: ModelAdapter,
        clock: Clock | None = None,
        rate_card: RateCard | None = None,
        executor_options: dict[str, Any] | None = None,
        dispatcher_factory: Any = None,
    ):
        if adapter.execution_mode.value == "live":
            from celltrace.config import api_key_present
            from celltrace.providers.readiness import require_capability_probe, verified_rate_card

            if not config.runtime.enable_live_calls or not api_key_present():
                raise ValueError("live runner requires explicit activation and OPENROUTER_API_KEY")
            if adapter.model_id != config.runtime.model_id:
                raise ValueError("live adapter model differs from the configured profile")
            checked_rates = verified_rate_card(config.runtime)
            require_capability_probe(config.runtime)
            if rate_card != checked_rates:
                raise ValueError("live adapter rates differ from the verified rate card")
        self.config = config
        self.adapter = adapter
        self.clock = clock or SystemClock()
        self.rate_card = rate_card
        self.executor_options = executor_options or {}
        self.dispatcher_factory = dispatcher_factory or ToolDispatcher
        self.store = RunStore(config.runtime.run_store, self.clock)

    def close(self) -> None:
        self.store.close()

    def _checkpointer(self) -> tuple[SqliteSaver, sqlite3.Connection]:
        path = self.config.runtime.checkpoint_store
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(connection), connection

    def _lock(self, run_id: str) -> RunLock:
        return RunLock(self.config.runtime.run_store.parent / "locks", run_id)

    # Start

    def start(
        self, snapshot_dir: Path, candidate: CandidateEvent, options: RunOptions
    ) -> InvestigationReport:
        run_id = options.run_id or f"run_{uuid.uuid4().hex[:16]}"
        case_id = options.case_id or f"case_{candidate.event_id}"
        connection, manifest = open_snapshot(snapshot_dir)
        try:
            repository = MeasurementRepository(connection)
            context = build_run_context(
                run_id,
                case_id,
                candidate,
                manifest["logical_hash"],
                str(snapshot_dir),
                repository,
                self.config.runtime,
            )
            identity = run_identity(context, self.config, self.adapter, options.arm)
            self.store.save_candidate(
                candidate.event_id,
                context.snapshot_hash,
                candidate.detector_config_hash,
                candidate.model_dump(mode="json"),
            )
            with self._lock(run_id):
                self.store.create_run(
                    run_id,
                    case_id,
                    candidate.event_id,
                    options.arm.value,
                    self.adapter.execution_mode.value,
                    self.adapter.model_id,
                    context.snapshot_hash,
                    identity,
                    context.model_dump(mode="json"),
                    run_limits(self.config, self.rate_card, options.arm),
                    experiment_id=options.experiment_id,
                )
                self.store.record_event(
                    run_id,
                    "run_started",
                    {
                        "resolved_config": self.config.redacted_summary(),
                        "adapter": self.adapter.describe(),
                        "identity": identity,
                    },
                )
                self.store.append_message(
                    run_id, Role.investigator.value, investigator_opening(context)
                )
                return self._execute(context, repository, options)
        finally:
            connection.close()

    # Resume

    def resume(
        self, run_id: str, snapshot_dir: Path | None = None, retry_uncertain: bool = False
    ) -> InvestigationReport:
        run = self.store.get_run(run_id)
        reopen = False
        if run["status"] == "terminal":
            existing = self.store.get_report(run_id)
            uncertain_stop = (run["terminal_reason"] or "").startswith("provider_uncertain")
            if not (retry_uncertain and uncertain_stop) and existing is not None:
                # A finished run is never silently restarted as if it were new.
                return InvestigationReport.model_validate(existing)
            reopen = True
        context = RunContext.model_validate(run["context"])
        snapshot_path = snapshot_dir or Path(context.snapshot_path)
        connection, manifest = open_snapshot(snapshot_path)
        try:
            if manifest["logical_hash"] != run["snapshot_hash"]:
                raise ResumeRefused(
                    "snapshot differs from the one this run started with, start a new run instead"
                )
            arm = Arm(run["arm"])
            identity = run_identity(context, self.config, self.adapter, arm)
            if content_hash(identity) != run["identity_hash"]:
                differences = sorted(
                    key for key in identity if identity[key] != run["identity"].get(key)
                )
                raise ResumeRefused(f"run identity changed: {differences}. Fork a new run.")
            with self._lock(run_id):
                if retry_uncertain:
                    self._authorize_uncertain_retry(run_id)
                if reopen:
                    self.store.reopen_run(run_id, self.store.get_report(run_id))
                self.store.record_event(run_id, "run_resumed", {"retry_uncertain": retry_uncertain})
                return self._execute(
                    context,
                    MeasurementRepository(connection),
                    RunOptions(arm=arm, run_id=run_id, case_id=context.case_id),
                )
        finally:
            connection.close()

    def _authorize_uncertain_retry(self, run_id: str) -> None:
        for action in self.store.actions(run_id, ActionKind.provider_attempt):
            if action.state == ActionState.uncertain:
                self.store.record_event(
                    run_id,
                    "uncertain_retry_authorized",
                    {
                        "action_id": action.action_id,
                        "warning": "the earlier attempt may have run and been billed. Its "
                        "reserved usage stays charged and the new attempt is charged again.",
                    },
                )

    # Shared execution

    def _execute(
        self, context: RunContext, repository: MeasurementRepository, options: RunOptions
    ) -> InvestigationReport:
        dispatcher = self.dispatcher_factory(context, repository, self.store, self.clock)
        executor = RoleTurnExecutor(
            context,
            self.store,
            dispatcher,
            self.adapter,
            self.config.runtime,
            self.clock,
            rate_card=self.rate_card,
            detector_configuration=self.config.analysis.detector,
            observation_configuration=self.config.analysis.observation,
            **self.executor_options,
        )
        verifier = Verifier(
            context,
            self.store,
            repository,
            on_query=self._verification_logger(context),
            detector_configuration=self.config.analysis.detector,
            observation_configuration=self.config.analysis.observation,
        )
        run_id = context.run_id

        def render(run_identifier: str, terminal_reason: str) -> dict[str, Any]:
            report = build_report(
                self.store,
                context,
                options.arm,
                self.adapter.execution_mode,
                self.adapter.model_id,
                terminal_reason,
                verifier=verifier,
            )
            payload = report.model_dump(mode="json")
            self.store.save_report(run_identifier, payload)
            self.store.mark_terminal(
                run_identifier, report.workflow_outcome.value, report.terminal_reason
            )
            if options.report_directory is not None:
                write_report_files(report, options.report_directory)
            return payload

        services = GraphServices(
            run_id=run_id,
            arm=options.arm,
            store=self.store,
            executor=executor,
            verifier=verifier,
            maximum_review_rounds=self.config.runtime.maximum_review_rounds,
            maximum_revisions=self.config.runtime.maximum_review_rounds,
            render=render,
        )
        reopened = sum(
            1 for event in self.store.run_events(run_id) if event["event_type"] == "run_reopened"
        )
        # A reopened run continues on a fresh graph thread. Its nodes replay recorded
        # submissions from the ledger, so no completed work is charged again.
        thread_id = run_id if reopened == 0 else f"{run_id}:reopened:{reopened}"
        saver, checkpoint_connection = self._checkpointer()
        try:
            graph = build_graph(services).compile(checkpointer=saver)
            graph_config: Any = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": GRAPH_RECURSION_LIMIT,
            }
            snapshot = graph.get_state(graph_config)
            graph.invoke(None if snapshot.next else {"run_id": run_id}, graph_config)
        finally:
            checkpoint_connection.close()
        stored = self.store.get_report(run_id)
        assert stored is not None
        return InvestigationReport.model_validate(stored)

    def _verification_logger(self, context: RunContext) -> Any:
        def log(description: str) -> None:
            key = f"{context.run_id}:verifier:{uuid.uuid4().hex[:12]}"
            action_id = self.store.reserve_action(
                context.run_id,
                key,
                key,
                Role.verifier.value,
                ActionKind.verification_query,
                {"description": description},
                reservation=Reservation(),
            )
            self.store.complete_action(action_id, ActionState.completed, 0, physical_query=True)

        return log


def write_report_files(report: InvestigationReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{report.run_id}.json"
    markdown_path = directory / f"{report.run_id}.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def export_run(store: RunStore, run_id: str, directory: Path) -> Path:
    """JSONL export derived from the run store, never a second source of truth."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.jsonl"
    records: list[dict[str, Any]] = [{"record": "run", **store.get_run(run_id)}]
    records += [{"record": "event", **event} for event in store.run_events(run_id)]
    records += [{"record": "action", **action} for action in store.raw_actions(run_id)]
    records += [{"record": "evidence", **evidence} for evidence in store.run_evidence(run_id)]
    for role in (Role.investigator, Role.reviewer):
        for message in store.messages(run_id, role.value):
            records.append({"record": "message", "role_owner": role.value, **message})
    records += [
        {"record": "findings", "findings_id": row[0], "revision": row[1], "payload": row[2]}
        for row in store.findings(run_id)
    ]
    records += [{"record": "review", **review} for review in store.reviews(run_id)]
    records += [{"record": "verification", **entry} for entry in store.verifications(run_id)]
    report = store.get_report(run_id)
    if report is not None:
        records.append({"record": "report", "payload": report})
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return path
