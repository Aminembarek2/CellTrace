"""Prepare, run and reproduce the planted-interpretation development experiment."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from celltrace.agents.roles import reviewer_opening, tool_result_block
from celltrace.agents.scripted_policies import ScriptedInvestigator, ScriptedReviewer
from celltrace.canonical import content_hash, file_sha256
from celltrace.clock import SystemClock
from celltrace.config import AppConfig, load_config
from celltrace.contracts.common import MEASUREMENT_TOOLS, Arm, Role
from celltrace.contracts.evidence import CandidateEvent, RunContext
from celltrace.contracts.findings import InvestigationFindings, ReviewResult
from celltrace.data.snapshot import open_snapshot
from celltrace.evaluation.planted_errors import (
    BASE_FAMILIES,
    DRAWS,
    METHOD_VERSION,
    METHOD_VERSION_4,
    SUPPORTED_METHODS,
    qualified_control,
    summarize,
    variants,
)
from celltrace.providers.base import ModelAdapter
from celltrace.providers.rate_card import RateCard
from celltrace.providers.scripted import ScriptedModelAdapter
from celltrace.release import source_tree_hash
from celltrace.storage.lock import RunLock
from celltrace.storage.run_store import ActionKind, ActionState, RunStore
from celltrace.tools.dispatch import ToolDispatcher, evidence_content_hash
from celltrace.tools.repository import MeasurementRepository
from celltrace.verification.verifier import Verifier
from celltrace.workflow.executor import RoleTurnExecutor, TurnStatus
from celltrace.workflow.runner import InvestigationRunner, RunOptions, run_limits

MODEL_ID = "deepseek/deepseek-v4.1-flash"
STUDY_CEILING_USD = 0.90
LIMITS = {
    "maximum_tool_requests": 6,
    "maximum_provider_attempts": 8,
    "maximum_input_tokens": 120000,
    "maximum_output_tokens": 32000,
    "maximum_output_tokens_per_attempt": 16384,
    "maximum_active_seconds": 600,
    "provider_timeout_seconds": 120,
    "maximum_request_retries": 0,
    "maximum_estimated_usd_per_run": 0.15,
    "maximum_estimated_usd_per_experiment": STUDY_CEILING_USD,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def study_config(
    directory: Path, supplied: AppConfig | None = None, ceiling_usd: float = STUDY_CEILING_USD
) -> AppConfig:
    overrides: dict[tuple[str, ...], Any] = {
        ("runtime", key): value for key, value in LIMITS.items()
    }
    overrides[("runtime", "maximum_estimated_usd_per_experiment")] = ceiling_usd
    overrides.update(
        {
            ("runtime", "run_store"): str(directory / "runs.sqlite3"),
            ("runtime", "checkpoint_store"): str(directory / "checkpoints.sqlite3"),
            ("runtime", "model_id"): MODEL_ID,
            ("runtime", "reviewer_tool_reserve"): 0,
            ("runtime", "review_round_tool_reserve"): 0,
            ("runtime", "review_round_provider_reserve"): 0,
            ("runtime", "review_round_input_reserve"): 0,
            ("runtime", "review_round_output_reserve"): 0,
        }
    )
    if supplied is None:
        return load_config(cli_overrides=overrides, environment={})
    result = supplied.model_dump(mode="json")
    for (_, key), value in overrides.items():
        result["runtime"][key] = value
    return AppConfig.model_validate(result)


def rebind_evidence(
    findings: dict[str, Any], evidence: list[dict[str, Any]], run_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reissue measured input citations for an isolated review, retaining source lineage."""
    findings = copy.deepcopy(findings)
    records = copy.deepcopy(evidence)
    remapping = {}
    for record in records:
        old_id = record["evidence_id"]
        record["run_id"] = run_id
        record["evidence_id"] = "ev_" + content_hash([run_id, record["cache_key"]])[:16]
        record["content_sha256"] = evidence_content_hash(record)
        remapping[old_id] = record["evidence_id"]
    for claim in findings["factual_claims"]:
        claim["evidence_ids"] = [remapping[identifier] for identifier in claim["evidence_ids"]]
    return findings, records


def select_bases(manifest: dict[str, Any], bases_per_family: int) -> list[dict[str, Any]]:
    """Version 4: families present on radio windows, each base on a window no other base uses."""

    def window(case: dict[str, Any]) -> tuple[str, int]:
        return case["source_session_id"], case["source_confirmation_ms"]

    # Planted controls cite measured RSRP, so only windows where it applies can be bases.
    usable = [c for c in manifest["cases"] if c["base_window"]["radio_supported_in_windows"]]
    present = [f for f in BASE_FAMILIES if any(c["family"] == f for c in usable)]
    ordered = sorted(usable, key=lambda item: item["case_id"])
    # The family with the fewest windows chooses first, so a common family cannot take them.
    scarcity = sorted(
        present,
        key=lambda family: (len({window(c) for c in ordered if c["family"] == family}), family),
    )
    used: set[tuple[str, int]] = set()
    chosen: dict[str, list[dict[str, Any]]] = {family: [] for family in present}
    for family in scarcity:
        for case in ordered:
            if len(chosen[family]) >= bases_per_family:
                break
            if case["family"] == family and window(case) not in used:
                chosen[family].append(case)
                used.add(window(case))
    return [case for family in present for case in chosen[family]]


def prepare(
    manifest_path: Path,
    out: Path,
    bases_per_family: int | None = None,
    phrasings: tuple[str, ...] | None = None,
    draws: int = DRAWS,
    ceiling_usd: float = STUDY_CEILING_USD,
) -> dict[str, Any]:
    from celltrace.evaluation.commands import frozen_runtime_settings

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") != "dev":
        raise ValueError("planted-error selection requires a development manifest")
    version_4 = bases_per_family is not None or phrasings is not None or draws != DRAWS
    selected = []
    if version_4:
        selected = select_bases(manifest, bases_per_family or 2)
        if not selected:
            raise ValueError("no development bases of the planted families")
    else:
        for family in BASE_FAMILIES:
            choices = sorted(
                (case for case in manifest["cases"] if case["family"] == family),
                key=lambda case: case["case_id"],
            )
            if len(choices) < 2:
                raise ValueError(f"need two development cases from {family}")
            selected.extend(choices[:2])
    method = METHOD_VERSION_4 if version_4 else METHOD_VERSION
    out.mkdir(parents=True, exist_ok=False)
    config = study_config(out / "offline", ceiling_usd=ceiling_usd)
    seed_config = study_config(out / "preparation")
    seed_data = seed_config.model_dump(mode="json")
    seed_data["runtime"].update(maximum_tool_requests=12, maximum_provider_attempts=12)
    seed_config = AppConfig.model_validate(seed_data)
    adapter = ScriptedModelAdapter({"investigator": ScriptedInvestigator()})
    runner = InvestigationRunner(seed_config, adapter)
    items: list[dict[str, Any]] = []
    inputs = {str(manifest_path.resolve()): file_sha256(manifest_path)}
    verifier_controls = []
    control_repairs: list[dict[str, Any]] = []
    try:
        for case in selected:
            directory = manifest_path.parent.parent / "cases" / case["case_id"]
            for name in (
                "candidate.json",
                "snapshot/snapshot_manifest.json",
                "snapshot/measurements.duckdb",
            ):
                path = directory / name
                if not path.is_file():
                    # The canonical snapshot currently uses measurements.duckdb.
                    raise ValueError(f"missing frozen input: {path}")
                inputs[str(path.resolve())] = file_sha256(path)
            candidate = CandidateEvent.model_validate_json(
                (directory / "candidate.json").read_text()
            )
            if content_hash(candidate.model_dump(mode="json")) != case["candidate_sha256"]:
                raise ValueError("development candidate changed")
            seed_id = "run_seed_" + content_hash(case["case_id"])[:16]
            report = runner.start(
                directory / "snapshot",
                candidate,
                RunOptions(arm=Arm.investigator_gated, case_id=case["case_id"], run_id=seed_id),
            )
            if report.rejected_claims or not report.verified_claims:
                raise ValueError("seed contains rejected or absent measured claims")
            context = RunContext.model_validate(runner.store.get_run(seed_id)["context"])
            connection, snapshot = open_snapshot(directory / "snapshot")
            try:
                if snapshot["logical_hash"] != case["case_snapshot_logical_hash"]:
                    raise ValueError("development snapshot identity changed")
                verifier = Verifier(context, runner.store, MeasurementRepository(connection))
                seed_findings = runner.store.findings(seed_id)[-1][2]
                seed_check = verifier.verify(InvestigationFindings.model_validate(seed_findings))
                failed = [check.claim_id for check in seed_check.failed_checks]
                # Only a rejected observation label may be repaired. Any failed measured
                # claim, provenance or policy check still stops the freeze.
                if any(not claim_id.startswith("pattern:") for claim_id in failed):
                    raise ValueError("seed findings fail a measured or provenance check")
                removed = frozenset(claim_id.removeprefix("pattern:") for claim_id in failed)
                control = qualified_control(seed_findings, removed)
                control_repairs.append(
                    {"base_id": case["case_id"], "removed_unsupported_patterns": sorted(removed)}
                )
                for family, target, findings in variants(control, phrasings):
                    if not verifier.verify(InvestigationFindings.model_validate(findings)).passed:
                        raise ValueError(f"{family} did not pass the numerical/provenance gate")
                    finding_id = "finding_" + content_hash([case["case_id"], family, method])[:20]
                    # Each draw is a separate conversation over identical findings and evidence.
                    for draw in range(draws):
                        item_id = "item_" + content_hash([finding_id, draw])[:20]
                        item_context = context.model_copy(update={"run_id": "run_" + item_id[5:]})
                        rebound, records = rebind_evidence(
                            findings, runner.store.run_evidence(seed_id), item_context.run_id
                        )
                        items.append(
                            {
                                "item_id": item_id,
                                "finding_id": finding_id,
                                "draw": draw,
                                "base_id": case["case_id"],
                                "context": item_context.model_dump(mode="json"),
                                "findings": rebound,
                                "evidence": records,
                                "label": {"family": family, "target_path": target},
                            }
                        )
                for kind in (
                    "wrong_number",
                    "unknown_citation",
                    "wrong_event",
                    "unobservable_exclusion",
                ):
                    broken = copy.deepcopy(control)
                    rate = next(
                        c for c in broken["factual_claims"] if c["claim_type"] == "metric_change"
                    )
                    if kind == "wrong_number":
                        rate["event_value"] += 100
                        rate["delta"] += 100
                    elif kind == "unknown_citation":
                        rate["evidence_ids"] = ["ev_unknown_control"]
                    elif kind == "wrong_event":
                        broken["event_id"] = "evt_wrong_control"
                    else:
                        # Radio impairment marked ruled out from reported RSRP alone.
                        broken["hypotheses"][0]["status"] = "contradicted"
                    check = verifier.verify(InvestigationFindings.model_validate(broken))
                    if check.passed:
                        raise ValueError(f"verifier failed its {kind} negative control")
                    verifier_controls.append(
                        {"base_id": case["case_id"], "kind": kind, "rejected": True}
                    )
            finally:
                connection.close()
    finally:
        runner.close()
    random.Random(20261010).shuffle(items)
    plan: dict[str, Any] = {
        "method": method,
        "design": {
            "bases_per_family": bases_per_family,
            "families": sorted({case["family"] for case in selected}),
            "phrasings": list(phrasings) if phrasings else None,
            "draws": draws,
            "distinct_windows": version_4,
        },
        "source_sha256": source_tree_hash(),
        "runtime": frozen_runtime_settings(config),
        "analysis": config.analysis.model_dump(mode="json"),
        "inputs": inputs,
        "items": items,
        "verifier_controls": verifier_controls,
        "control_repairs": control_repairs,
        "source_sessions": sorted({case["source_session_id"] for case in selected}),
        "provider": {
            "model_id": MODEL_ID,
            "provider": "deepseek",
            "gateway": "OpenRouter",
            "effort": "high",
        },
        "study_ceiling_usd": ceiling_usd,
        "semantic_accuracy_claim": "excluded",
    }
    plan["plan_sha256"] = content_hash(plan)
    write_json(out / "plan.json", plan)
    write_json(
        out / "payload_inventory.json",
        {
            "plan_sha256": plan["plan_sha256"],
            "reviews": len(items),
            "draws_per_finding": draws,
            "bases": len(selected),
            "source_sessions": len(plan["source_sessions"]),
            "destination": plan["provider"],
            "fields_sent": [
                "case candidate and bounds",
                "generated structured findings",
                "same-case measurement evidence and query results",
            ],
            "fields_excluded": [
                "hidden labels",
                "corruption recipes",
                "arm/family names",
                "other cases",
                "manual annotations",
                "PAM data",
            ],
            "study_ceiling_usd": ceiling_usd,
            "capability_probe_ceiling_usd": 0.10,
            "combined_planned_ceiling_usd": round(ceiling_usd + 0.10, 2),
            "new_paid_calls": 0,
        },
    )
    return plan


def load_plan(out: Path, config: AppConfig | None = None) -> dict[str, Any]:
    from celltrace.evaluation.commands import frozen_runtime_settings

    plan: dict[str, Any] = json.loads((out / "plan.json").read_text())
    digest = content_hash({key: value for key, value in plan.items() if key != "plan_sha256"})
    if digest != plan["plan_sha256"] or plan["source_sha256"] != source_tree_hash():
        raise ValueError("planted-error plan or source changed; prepare a new freeze")
    if plan["method"] not in SUPPORTED_METHODS:
        raise ValueError("unsupported planted-error method")
    for name, expected in plan["inputs"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError("frozen planted-error input changed")
    config = config or study_config(out / "offline", ceiling_usd=plan["study_ceiling_usd"])
    if plan["runtime"] != frozen_runtime_settings(config) or plan[
        "analysis"
    ] != config.analysis.model_dump(mode="json"):
        raise ValueError("planted-error execution settings differ from the freeze")
    return plan


def recorded_outcomes(store: RunStore, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outcomes = {}
    for item in plan["items"]:
        run_id = item["context"]["run_id"]
        for event in store.run_events(run_id):
            if event["event_type"] == "planted_outcome":
                outcomes[item["item_id"]] = event["payload"]
    return outcomes


def review_one(
    item: dict[str, Any],
    config: AppConfig,
    adapter: ModelAdapter,
    rates: RateCard | None,
    store: RunStore,
    experiment_id: str,
) -> dict[str, Any]:
    context = RunContext.model_validate(item["context"])
    run_id = context.run_id
    connection, snapshot = open_snapshot(Path(context.snapshot_path))
    try:
        if snapshot["logical_hash"] != context.snapshot_hash:
            raise ValueError("review snapshot changed")
        repository = MeasurementRepository(connection)
        existing = next((r for r in store.list_runs(experiment_id) if r["run_id"] == run_id), None)
        if existing is None:
            store.create_run(
                run_id,
                context.case_id,
                context.candidate.event_id,
                Arm.investigator_reviewer_gated.value,
                adapter.execution_mode.value,
                adapter.model_id,
                context.snapshot_hash,
                {"study": experiment_id, "input_sha256": content_hash(item)},
                context.model_dump(mode="json"),
                run_limits(config, rates),
                experiment_id=experiment_id,
            )
            for record in item["evidence"]:
                store.save_evidence(record)
            store.record_event(
                run_id,
                "prepared_input",
                {
                    "input_sha256": content_hash(item),
                    "origin": "offline measurement preparation; no runtime investigator",
                },
            )
            store.append_message(
                run_id, Role.reviewer.value, reviewer_opening(context, item["findings"], 1)
            )
        check = Verifier(context, store, repository).verify(
            InvestigationFindings.model_validate(item["findings"])
        )
        if not check.passed:
            raise ValueError("scheduled findings no longer pass code verification")
        clock = SystemClock()
        executor = RoleTurnExecutor(
            context,
            store,
            ToolDispatcher(context, repository, store, clock),
            adapter,
            config.runtime,
            clock,
            rates,
        )
        review = None
        status = "not_run"
        for _ in range(config.runtime.maximum_provider_attempts + 2):
            result = executor.turn(Role.reviewer)
            status = result.status.value
            if result.status == TurnStatus.submitted:
                assert result.submission is not None
                review = ReviewResult.model_validate(result.submission["payload"])
                try:
                    review.validate_finding_paths(item["findings"])
                    known = {claim["claim_id"] for claim in item["findings"]["factual_claims"]}
                    if not set(review.disputed_claim_ids) <= known:
                        raise ValueError("disputed claim is absent from findings")
                except ValueError as invalid:
                    store.append_message(
                        run_id,
                        Role.reviewer.value,
                        {
                            "role": "user",
                            "content": [
                                tool_result_block(
                                    result.submission["tool_use_id"],
                                    {"error": "INVALID_OUTPUT", "message": str(invalid)},
                                    True,
                                )
                            ],
                        },
                    )
                    review = None
                    status = "invalid_submission"
                    continue
                break
            if result.status in (
                TurnStatus.provider_uncertain,
                TurnStatus.provider_failed,
                TurnStatus.budget_exhausted,
            ):
                break
        actions = store.actions(run_id, ActionKind.tool_request)
        independent_ids = {
            action.evidence_id
            for action in actions
            if action.role == "reviewer"
            and action.state == ActionState.completed
            and action.tool_name in {tool.value for tool in MEASUREMENT_TOOLS}
            and action.evidence_id is not None
        }
        payload = {
            "status": status,
            "review": None if review is None else review.model_dump(mode="json"),
            "successful_independent_queries": len(independent_ids),
            "cited_independent_query": bool(
                review and set(review.independent_evidence_ids) & independent_ids
            ),
            "usage": asdict(store.usage(run_id)),
        }
        store.record_event(run_id, "planted_outcome", payload)
        store.mark_terminal(run_id, "completed" if review else "partial", status)
        return payload
    finally:
        connection.close()


def run_study(
    out: Path, execution: Path, config: AppConfig, adapter: ModelAdapter, rates: RateCard | None
) -> dict[str, Any]:
    plan = load_plan(out, config)
    if adapter.execution_mode.value == "live":
        from celltrace.config import api_key_present
        from celltrace.providers.readiness import require_capability_probe, verified_rate_card

        if execution.resolve() != (out / "live").resolve():
            raise ValueError("live study uses its single canonical ledger at <plan>/live")
        if (
            not config.runtime.enable_live_calls
            or not api_key_present()
            or adapter.model_id != MODEL_ID
        ):
            raise ValueError("live planted-error run requires explicit exact-profile activation")
        require_capability_probe(config.runtime)
        if rates != verified_rate_card(config.runtime):
            raise ValueError("live planted-error rates differ from verified metadata")
    elif rates is not None:
        raise ValueError("offline planted-error run cannot have paid rates")
    execution.mkdir(parents=True, exist_ok=True)
    store = RunStore(execution / "runs.sqlite3", SystemClock())
    experiment_id = "exp_planted_" + plan["plan_sha256"][:16]
    try:
        with RunLock(execution / "locks", "study"):
            store.create_experiment(
                experiment_id,
                plan["study_ceiling_usd"],
                {
                    "plan_sha256": plan["plan_sha256"],
                    "execution_mode": adapter.execution_mode.value,
                },
            )
            for run in store.list_runs(experiment_id):
                if (
                    run["execution_mode"] != adapter.execution_mode.value
                    or run["model_id"] != adapter.model_id
                ):
                    raise ValueError("execution mode or model changed in the study ledger")
                for action in store.actions(run["run_id"], ActionKind.provider_attempt):
                    if action.state == ActionState.reserved:
                        store.complete_action(
                            action.action_id,
                            ActionState.failed,
                            0,
                            error_code="NOT_DISPATCHED",
                            usage={
                                "usage_known": True,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "usd": 0.0,
                            },
                        )
                    elif action.state == ActionState.started:
                        store.mark_uncertain(action.action_id, "interrupted planted-error dispatch")
            outcomes = recorded_outcomes(store, plan)
            uncertain = any(
                store.usage(r["run_id"]).unknown_usage_attempts
                for r in store.list_runs(experiment_id)
            )
            for item in plan["items"]:
                if item["item_id"] in outcomes:
                    continue
                if uncertain or (execution / "STOP").exists():
                    break
                outcome = review_one(item, config, adapter, rates, store, experiment_id)
                outcomes[item["item_id"]] = outcome
                uncertain = bool(outcome["usage"]["unknown_usage_attempts"])
                write_json(
                    execution / "progress.json",
                    {
                        "recorded": len(outcomes),
                        "scheduled": len(plan["items"]),
                        "uncertain": uncertain,
                    },
                )
            design = plan.get("design") or {}
            phrasings = design.get("phrasings")
            summary = summarize(
                plan["items"],
                outcomes,
                adapter.execution_mode.value,
                method=plan["method"],
                draws=design.get("draws") or DRAWS,
                phrasings=tuple(phrasings) if phrasings else None,
            )
            summary.update(
                plan_sha256=plan["plan_sha256"],
                source_sha256=plan["source_sha256"],
                source_sessions=len(plan["source_sessions"]),
                bases=len({i["base_id"] for i in plan["items"]}),
                all_outcomes_known=not uncertain,
                known_or_reserved_usd=sum(
                    store.usage(r["run_id"]).estimated_usd for r in store.list_runs(experiment_id)
                ),
                status="blocked_unknown_usage"
                if uncertain
                else ("finished" if len(outcomes) == len(plan["items"]) else "incomplete"),
            )
            write_json(execution / "summary.json", summary)
            return summary
    finally:
        store.close()


def command_prepare(arguments: argparse.Namespace) -> int:
    plan = prepare(
        arguments.manifest.resolve(),
        arguments.out.resolve(),
        bases_per_family=arguments.bases_per_family,
        phrasings=tuple(arguments.wording) if arguments.wording else None,
        draws=arguments.draws,
        ceiling_usd=arguments.ceiling_usd,
    )
    print(
        json.dumps(
            {
                "plan": str(arguments.out / "plan.json"),
                "plan_sha256": plan["plan_sha256"],
                "scheduled": len(plan["items"]),
                "new_paid_calls": 0,
            },
            indent=2,
        )
    )
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    from celltrace.cli import build_adapter

    if arguments.live == arguments.offline:
        raise ValueError("choose exactly one of --offline or --live")
    supplied = None
    if arguments.live:
        if arguments.config is None:
            raise ValueError("live planted-error run needs a verified configuration")
        supplied = load_config(
            arguments.config, cli_overrides={("runtime", "enable_live_calls"): True}
        )
    frozen_plan = json.loads((arguments.plan.resolve() / "plan.json").read_text())
    config = study_config(
        arguments.execution.resolve(), supplied, ceiling_usd=frozen_plan["study_ceiling_usd"]
    )
    if arguments.live:
        adapter, rates = build_adapter(config, True)
    else:
        adapter, rates = ScriptedModelAdapter({"reviewer": ScriptedReviewer()}), None
    summary = run_study(
        arguments.plan.resolve(), arguments.execution.resolve(), config, adapter, rates
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "finished" else 3


def register(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "planted-errors", help="frozen matched development reviewer controls"
    )
    actions = command.add_subparsers(dest="action", required=True)
    prepare_parser = actions.add_parser(
        "prepare", help="freeze development inputs without paid calls"
    )
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--bases-per-family", type=int, help="version 4 design")
    prepare_parser.add_argument(
        "--wording", choices=("original", "new"), action="append", help="version 4 design"
    )
    prepare_parser.add_argument("--draws", type=int, default=DRAWS)
    prepare_parser.add_argument("--ceiling-usd", type=float, default=STUDY_CEILING_USD)
    prepare_parser.set_defaults(handler=command_prepare)
    run_parser = actions.add_parser(
        "run", help="review frozen items using the durable role executor"
    )
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--execution", type=Path, required=True)
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--offline", action="store_true")
    run_parser.add_argument("--live", action="store_true")
    run_parser.set_defaults(handler=command_run)
