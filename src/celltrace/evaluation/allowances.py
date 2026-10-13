"""Frozen, model-specific development calibration using the existing benchmark and ledger.

A percentile is a candidate, not an optimality claim. Failed trajectories remain censored
observations; changing capacity and dividing equal capacity are separate comparisons.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from celltrace.canonical import content_hash, file_sha256
from celltrace.config import AppConfig, load_config
from celltrace.contracts.common import Arm
from celltrace.evaluation.benchmark import experiment_directory, run_benchmark, verify_freeze
from celltrace.evaluation.commands import freeze_hashes
from celltrace.evaluation.review.persistence import write_json_atomic
from celltrace.providers.base import ModelAdapter
from celltrace.providers.rate_card import RateCard
from celltrace.release import source_tree_hash
from celltrace.storage.lock import RunLock

METHOD = "role_tool_calibration_1"
PROFILE_TOOLS = 48
BASELINE_TOOLS = 24
MINIMUM_REVIEWER_RESERVE = 3
PERCENTILE = 0.9


class CalibrationRule(TypedDict):
    profile_tools: int
    percentile: float
    minimum_reviewer_reserve: int
    baseline_tools: int
    interpretation: str


RULE: CalibrationRule = {
    "profile_tools": 48,
    "percentile": 0.9,
    "minimum_reviewer_reserve": 3,
    "baseline_tools": 24,
    "interpretation": "development candidate; censored tails are conditional",
}
CEILINGS = {"profile": 0.60, "baseline": 1.20, "shared": 1.20, "reserved": 0.60}
ARMS = [Arm.investigator_raw, Arm.investigator_gated, Arm.investigator_reviewer_gated]
ConfiguredAdapterFactory = Callable[[AppConfig], tuple[ModelAdapter, RateCard | None]]


def nearest_rank(values: list[int], fraction: float = 0.9) -> int:
    if not values or not 0 < fraction <= 1:
        raise ValueError("a percentile needs observations and a fraction in (0, 1]")
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def phase_rows(index_path: Path, manifest_path: Path, model_id: str) -> list[dict[str, Any]]:
    """Read every indexed outcome; pending/prevented/missing reports are never successes."""
    index = json.loads(index_path.read_text())
    cases = {c["case_id"]: c for c in json.loads(manifest_path.read_text())["cases"]}
    connection = sqlite3.connect(
        index_path.with_name("runs.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    rows = []
    seen = set()
    try:
        for entry in index["runs"]:
            identity = (entry["case_id"], entry["arm"])
            if identity in seen:
                raise ValueError("a calibration phase has duplicate case/arm observations")
            seen.add(identity)
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (entry["run_id"],)
            ).fetchone()
            if run and run["model_id"] != model_id:
                raise ValueError("calibration cannot pool different models")
            tools = {
                r["role"]: int(r["used"])
                for r in connection.execute(
                    "SELECT role, SUM(reserved_tool_units) used FROM actions "
                    "WHERE run_id = ? AND kind = 'tool_request' GROUP BY role",
                    (entry["run_id"],),
                )
            }
            cost = connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN usage_known=1 THEN COALESCE(actual_usd,0) "
                "ELSE reserved_usd END),0), SUM(CASE WHEN usage_known=0 THEN 1 ELSE 0 END) "
                "FROM actions WHERE run_id=? AND kind='provider_attempt'",
                (entry["run_id"],),
            ).fetchone()
            outcome = str(run["workflow_outcome"] if run else entry["status"])
            report = connection.execute(
                "SELECT 1 FROM reports WHERE run_id=?", (entry["run_id"],)
            ).fetchone()
            rows.append(
                {
                    "case_id": entry["case_id"],
                    "arm": entry["arm"],
                    "source_session_id": cases[entry["case_id"]]["source_session_id"],
                    "outcome": outcome,
                    "terminal_reason": run["terminal_reason"] if run else None,
                    "investigator": tools.get("investigator", 0),
                    "reviewer": tools.get("reviewer", 0),
                    "known_or_reserved_usd": float(cost[0]),
                    "unknown_attempts": int(cost[1] or 0),
                    "censored": outcome != "completed" or report is None,
                    "report_missing": report is None,
                }
            )
    finally:
        connection.close()
    return rows


def derive_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if not r["censored"] and not r["unknown_attempts"]]
    if not complete:
        return {"status": "blocked_no_completed_profile", "rows": rows}
    investigator = nearest_rank([r["investigator"] for r in complete])
    reserve = max(RULE["minimum_reviewer_reserve"], nearest_rank([r["reviewer"] for r in complete]))
    total = investigator + reserve
    sensitivity = {}
    for session in sorted({r["source_session_id"] for r in complete}):
        selected = [r for r in complete if r["source_session_id"] == session]
        sensitivity[session] = {
            role: nearest_rank([r[role] for r in selected]) for role in ("investigator", "reviewer")
        }
    return {
        "status": "blocked_invalid_candidate"
        if not 0 < investigator < total <= 100
        else "conditional_censored"
        if len(complete) != len(rows)
        else "profile_complete",
        "investigator_tools": investigator,
        "reviewer_tool_reserve": reserve,
        "total_tools": total,
        "completed": len(complete),
        "scheduled": len(rows),
        "completed_fraction": len(complete) / len(rows),
        "censored_rows": [r for r in rows if r not in complete],
        "session_percentile_sensitivity": sensitivity,
        "percentile_interval": None,
        "interval_note": "too few source sessions; conditional descriptive percentiles only",
    }


def prepare_study(manifest_path: Path, config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preserve the existing allowance study")
    config = load_config(config_path, environment={})
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") != "dev" or not manifest.get("frozen"):
        raise ValueError("allowance calibration requires a frozen development manifest")
    if len(manifest["cases"]) != 18:
        raise ValueError("this protocol profiles all eighteen declared development cases")
    if len({case["source_session_id"] for case in manifest["cases"]}) != 2:
        raise ValueError("this protocol uses exactly two development source sessions")
    if (
        config.runtime.maximum_tool_requests != RULE["profile_tools"]
        or config.runtime.reviewer_tool_reserve
        or not config.runtime.show_remaining_budget
        or config.runtime.maximum_estimated_usd_per_experiment != CEILINGS["profile"]
    ):
        raise ValueError("profile configuration differs from the predeclared method")
    verify_freeze(manifest["freeze"], config)
    paths = [manifest_path, config_path, Path("uv.lock")]
    paths.extend(p for p in (manifest_path.parent.parent / "cases").rglob("*") if p.is_file())
    for dependency in (
        config.runtime.verified_rate_card,
        config.runtime.verified_endpoint_record,
        config.runtime.verified_capability_record,
    ):
        if dependency is not None:
            paths.append(dependency)
    plan = {
        "method": METHOD,
        "rule": RULE,
        "ceilings": CEILINGS,
        "model_id": config.runtime.model_id,
        "configuration": str(config_path),
        "manifest": str(manifest_path),
        "source_tree_sha256": source_tree_hash(),
        "inputs": {str(p): file_sha256(p) for p in sorted(set(paths))},
        "scope": "development only; not held-out performance or an optimal budget",
    }
    plan["plan_hash"] = content_hash(plan)
    output.mkdir(parents=True)
    write_json_atomic(output / "plan.json", plan)
    return plan


def load_study(output: Path) -> dict[str, Any]:
    plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
    supplied = plan.pop("plan_hash")
    if content_hash(plan) != supplied or plan["method"] != METHOD:
        raise ValueError("allowance plan content or method changed")
    plan["plan_hash"] = supplied
    if plan["source_tree_sha256"] != source_tree_hash():
        raise ValueError("source changed after allowance freeze")
    for name, expected in plan["inputs"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"allowance input changed: {name}")
    return plan


def phase_inputs(
    output: Path, plan: dict[str, Any], name: str, total: int, reserve: int = 0
) -> tuple[Path, AppConfig]:
    """Freeze each derived policy before dispatch; observations and case contents stay identical."""
    config = load_config(Path(plan["configuration"]), environment={})
    settings = config.model_dump(mode="json")
    settings["runtime"].update(
        maximum_tool_requests=total,
        reviewer_tool_reserve=reserve,
        maximum_estimated_usd_per_experiment=plan["ceilings"][name],
        enable_live_calls=True,
    )
    effective = AppConfig.model_validate(settings)
    source = Path(plan["manifest"])
    target = output / "policies" / name
    manifest_path = target / "labels/manifest.json"
    manifest = json.loads(source.read_text())
    manifest["freeze"] = freeze_hashes(effective)
    manifest["derived_from_manifest_sha256"] = file_sha256(source)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        verify_freeze(prior["freeze"], effective)
        manifest["freeze"]["frozen_at"] = prior["freeze"]["frozen_at"]
        if content_hash(prior) != content_hash(manifest):
            raise ValueError("derived allowance policy changed")
        if (target / "cases").resolve() != (source.parent.parent / "cases").resolve():
            raise ValueError("derived case inputs changed")
        if json.loads((target / "configuration.json").read_text()) != settings:
            raise ValueError("derived configuration changed")
    else:
        target.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(exist_ok=True)
        # Immutable case inputs are shared; agents never receive filesystem access.
        case_link = target / "cases"
        case_source = (source.parent.parent / "cases").resolve()
        if case_link.exists():
            if case_link.resolve() != case_source:
                raise ValueError("derived case inputs changed")
        else:
            case_link.symlink_to(case_source, target_is_directory=True)
        configuration_path = target / "configuration.json"
        if configuration_path.exists() and json.loads(configuration_path.read_text()) != settings:
            raise ValueError("derived configuration changed")
        write_json_atomic(target / "configuration.json", settings)
        write_json_atomic(manifest_path, manifest)
    return manifest_path, effective


def run_study(output: Path, adapter_factory: ConfiguredAdapterFactory) -> dict[str, Any]:
    """Serial paid phases reuse the benchmark's durable budgets, locks and cooperative stops."""
    with RunLock(output / "locks", "allowance-study"):
        plan = load_study(output)
        progress = output / "progress.json"
        state: dict[str, Any] = (
            json.loads(progress.read_text())
            if progress.exists()
            else {"plan_hash": plan["plan_hash"], "phases": {}}
        )
        if state["plan_hash"] != plan["plan_hash"]:
            raise ValueError("progress belongs to another allowance plan")
        if (output / "STOP").exists():
            state["status"] = "stopped_between_phases"
            write_json_atomic(progress, state)
            return state
        index = _execute_phase(
            output,
            plan,
            "profile",
            PROFILE_TOOLS,
            0,
            [Arm.investigator_reviewer_gated],
            adapter_factory,
        )
        state["phases"]["profile"] = str(index)
        rows = phase_rows(index, Path(plan["manifest"]), plan["model_id"])
        if len(rows) != 18:
            state["status"] = "profile_incomplete"
            write_json_atomic(output / "progress.json", state)
            return state
        candidate = derive_candidate(rows)
        candidate_path = output / "candidate.json"
        if candidate_path.exists() and json.loads(candidate_path.read_text()) != candidate:
            raise ValueError("a frozen candidate cannot be changed by later outcomes")
        if not candidate_path.exists():
            write_json_atomic(candidate_path, candidate)
        state["candidate"] = candidate
        write_json_atomic(output / "progress.json", state)
        if candidate["status"].startswith("blocked"):
            state["status"] = candidate["status"]
            write_json_atomic(progress, state)
            return state
        total, reserve = candidate["total_tools"], candidate["reviewer_tool_reserve"]
        comparisons = [("baseline", 24, 0, ARMS)]
        if total != 24:
            comparisons.append(("shared", total, 0, ARMS))
        comparisons.append(("reserved", total, reserve, [Arm.investigator_reviewer_gated]))
        for name, tools, reserved, arms in comparisons:
            if (output / "STOP").exists():
                state["status"] = "stopped_between_phases"
                break
            index = _execute_phase(output, plan, name, tools, reserved, arms, adapter_factory)
            state["phases"][name] = str(index)
            write_json_atomic(output / "progress.json", state)
            if len(json.loads(index.read_text())["runs"]) != 18 * len(arms):
                state["status"] = "comparison_incomplete"
                break
        else:
            state["status"] = "finished"
        write_json_atomic(output / "progress.json", state)
        return state


def _execute_phase(
    output: Path,
    plan: dict[str, Any],
    name: str,
    total: int,
    reserve: int,
    arms: list[Arm],
    adapter_factory: ConfiguredAdapterFactory,
) -> Path:
    load_study(output)
    manifest, config = phase_inputs(output, plan, name, total, reserve)
    experiment = f"exp_allowance_{name}_{plan['plan_hash'][:12]}"
    write_json_atomic(
        output / "active_phase.json",
        {"phase": name, "stop_file": str(experiment_directory(manifest, experiment) / "STOP")},
    )
    return run_benchmark(
        manifest,
        "live",
        arms,
        lambda: adapter_factory(config),
        config=config,
        experiment_id=experiment,
    )


def study_expenses(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Include paid outcomes absent from the latest index after an interruption."""
    expenses = {}
    for name in CEILINGS:
        database = (
            output
            / "policies"
            / name
            / "experiments"
            / (f"exp_allowance_{name}_{plan['plan_hash'][:12]}")
            / "runs.sqlite3"
        )
        if not database.exists():
            continue
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN usage_known=1 "
                "THEN COALESCE(actual_usd,0) ELSE reserved_usd END),0), "
                "COALESCE(SUM(CASE WHEN usage_known=0 THEN 1 ELSE 0 END),0) "
                "FROM actions WHERE kind='provider_attempt'"
            ).fetchone()
            expenses[name] = {"known_or_reserved_usd": row[0], "unknown_attempts": row[1]}
        finally:
            connection.close()
    return expenses


def _unsupported_delivered_claims(row: dict[str, Any]) -> int:
    # A fallback delivers the verified subset, not the rejected final submission.
    if row.get("report_basis") == "deterministic_fallback":
        return 0
    return max(0, int(row.get("checkable_claims") or 0) - int(row.get("supported_claims") or 0))


def _quality_gates(
    metrics: dict[str, Any], shared: str, expenses: dict[str, Any]
) -> dict[str, bool]:
    arm = Arm.investigator_reviewer_gated.value
    before, equal, reserved = (metrics[name][arm] for name in ("baseline", shared, "reserved"))
    compared_gated = [
        record
        for name in {"baseline", shared, "reserved"}
        for selected_arm, record in metrics[name].items()
        if selected_arm != Arm.investigator_raw.value
    ]
    return {
        "completion_not_decreased": reserved["completed"]
        >= max(before["completed"], equal["completed"]),
        "unsupported_claims_not_increased": reserved["unsupported_delivered_claims"]
        <= min(before["unsupported_delivered_claims"], equal["unsupported_delivered_claims"]),
        "no_prohibited_checked_labels": all(
            record["prohibited_active_or_radio_label_runs"] == 0 for record in compared_gated
        ),
        "no_unknown_usage": all(
            record["unknown_attempts"] == 0 and record["missing_outcomes"] == 0
            for record in compared_gated
        )
        and all(record["unknown_attempts"] == 0 for record in expenses.values()),
    }


def summarize_study(output: Path, public: Path) -> dict[str, Any]:
    from celltrace.evaluation.scoring import score_experiment

    if public.exists():
        raise FileExistsError("preserve previous allowance summaries")
    with RunLock(output / "locks", "allowance-study"):
        plan = load_study(output)
        progress = json.loads((output / "progress.json").read_text())
        public.mkdir(parents=True)
        metrics, comparisons = {}, {}
        for name, index in progress["phases"].items():
            manifest = output / "policies" / name / "labels/manifest.json"
            # Per-run identities and private labels stay in the private study.
            scoring = output / "summaries" / public.name / name
            score_experiment(Path(index), manifest, scoring)
            cases = {c["case_id"]: c for c in json.loads(manifest.read_text())["cases"]}
            with (scoring / "runs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            groups = {}
            for arm in sorted({r["arm"] for r in rows}):
                selected = [r for r in rows if r["arm"] == arm]
                false_labels = 0
                for row in selected:
                    forbidden = set(cases[row["case_id"]]["expected_patterns"]["forbidden"])
                    checked = {
                        "active_rate_drop",
                        "reported_radio_degradation",
                        "stable_reported_radio",
                    }
                    false_labels += bool(
                        set(json.loads(row["reported_patterns"] or "[]")) & forbidden & checked
                    )
                groups[arm] = {
                    "runs": len(selected),
                    "completed": sum(r["workflow_outcome"] == "completed" for r in selected),
                    "outcomes": dict(Counter(r["workflow_outcome"] for r in selected)),
                    "unsupported_delivered_claims": sum(
                        _unsupported_delivered_claims(r) for r in selected
                    ),
                    "prohibited_active_or_radio_label_runs": false_labels,
                    "known_or_reserved_usd": sum(float(r["estimated_usd"] or 0) for r in selected),
                    "unknown_attempts": sum(
                        int(r["unknown_usage_attempts"] or 0) for r in selected
                    ),
                    "missing_outcomes": sum(not r.get("report_basis") for r in selected),
                }
            metrics[name] = groups
            comparisons[name] = {
                r["case_id"]: r for r in rows if r["arm"] == Arm.investigator_reviewer_gated.value
            }
        candidate = json.loads((output / "candidate.json").read_text())
        shared = "baseline" if candidate.get("total_tools") == 24 else "shared"
        arm = Arm.investigator_reviewer_gated.value
        required = ("baseline", shared, "reserved")
        expenses = study_expenses(output, plan)
        ready = progress.get("status") == "finished" and all(
            name in metrics
            and expected_arm.value in metrics[name]
            and metrics[name][expected_arm.value]["runs"] == 18
            for name in required
            for expected_arm in ([Arm.investigator_reviewer_gated] if name == "reserved" else ARMS)
        )
        gates = {}
        paired = {}
        if ready:
            gates = _quality_gates(metrics, shared, expenses)
            for name in ("baseline", shared):
                if set(comparisons[name]) != set(comparisons["reserved"]):
                    raise ValueError("comparison cases differ")
                differences = [
                    int(comparisons["reserved"][case]["workflow_outcome"] == "completed")
                    - int(comparisons[name][case]["workflow_outcome"] == "completed")
                    for case in comparisons[name]
                ]
                paired["reserved_minus_" + name] = {
                    "completion_difference": sum(differences) / 18,
                    "pairs": 18,
                    "interval": None,
                    "note": "two source sessions, one draw: development point difference only",
                }
        recommendation = (
            "incomplete"
            if not ready
            else "not_recommended"
            if not all(gates.values())
            else "tested_conditional_candidate"
            if candidate["status"] != ("profile_complete")
            else "tested_development_candidate"
        )
        summary = {
            "method": METHOD,
            "plan_hash": plan["plan_hash"],
            "model_id": plan["model_id"],
            "scope": plan["scope"],
            "candidate": {
                key: value
                for key, value in candidate.items()
                if key not in ("rows", "censored_rows")
            },
            "censored_outcomes": dict(
                Counter(
                    r["outcome"] for r in candidate.get("censored_rows", candidate.get("rows", []))
                )
            ),
            "metrics": metrics,
            "recommendation": recommendation,
            "gates": gates,
            "paired_completion": paired,
            "expenses": expenses,
            "limitations": [
                "No active-default promotion or held-out generalization claim.",
                "Model-specific demand must be measured again after source/prompt changes.",
                "Completed-trajectory percentiles do not identify a censored demand tail.",
                "One draw per case cannot separate sampling noise from small policy differences.",
            ],
        }
        write_json_atomic(public / "summary.json", summary)
        _plot_tradeoff(metrics, public / "completion_cost.png")
        return summary


def _plot_tradeoff(metrics: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#2568b8", "shared": "#ba4b20", "reserved": "#087e58"}
    names = {
        Arm.investigator_raw.value: "A · Investigator",
        Arm.investigator_gated.value: "B · Code checks",
        Arm.investigator_reviewer_gated.value: "C · Reviewer + checks",
    }
    records = [
        (policy, arm, record)
        for policy, groups in metrics.items()
        if policy != "profile"
        for arm, record in groups.items()
    ]
    if not records:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.5), constrained_layout=True)
    labels = [f"{policy} {names[arm]}" for policy, arm, _ in records]
    for index, (policy, _, record) in enumerate(records):
        cost = record["known_or_reserved_usd"]
        fraction = record["completed"] / record["runs"]
        axes[0].barh(index, cost, color=colors[policy], height=0.6)
        axes[1].barh(index, fraction, color=colors[policy], height=0.6)
        axes[0].text(cost, index, f"  ${cost:.3f}", va="center", fontsize=9)
        axes[1].text(
            fraction, index, f"  {record['completed']}/{record['runs']}", va="center", fontsize=9
        )
    for axis in axes:
        axis.set_yticks(range(len(labels)), labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set(xlabel="Provider USD · includes unknown reservations")
    axes[0].set_xlim(0, max(r["known_or_reserved_usd"] for _, _, r in records) * 1.3 + 0.01)
    axes[1].set(xlabel="Completed fraction of all scheduled runs", xlim=(0, 1.18))
    figure.suptitle(
        "Development allowance comparison\n"
        "Two source sessions · descriptive outcomes · no confidence interval",
        fontsize=12,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)
