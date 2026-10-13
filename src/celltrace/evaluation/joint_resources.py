"""Predeclared factorial development contrast, using the normal benchmark ledger.

Policy selection is descriptive and model-specific. No default is promoted and
the two reused development sessions cannot establish generalization.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

from celltrace.canonical import content_hash, file_sha256
from celltrace.config import AppConfig, load_config
from celltrace.contracts.common import Arm
from celltrace.evaluation.allowances import ConfiguredAdapterFactory, _unsupported_delivered_claims
from celltrace.evaluation.benchmark import experiment_directory, run_benchmark, verify_freeze
from celltrace.evaluation.commands import freeze_hashes
from celltrace.evaluation.review.persistence import write_json_atomic
from celltrace.release import source_tree_hash
from celltrace.storage.lock import RunLock

METHOD = "joint_resources_development_3"
POLICIES = {
    "shared": (False, False),
    "query": (True, False),
    "joint": (False, True),
    "combined": (True, True),
}
TOTALS = {
    "maximum_tool_requests": 27,
    "maximum_provider_attempts": 20,
    "maximum_input_tokens": 240000,
    "maximum_output_tokens": 64000,
    "maximum_output_tokens_per_attempt": 16384,
    "maximum_active_seconds": 600,
    "maximum_estimated_usd_per_run": 0.5,
    "maximum_estimated_usd_per_experiment": 0.55,
    "reviewer_tool_reserve": 0,
    "maximum_review_rounds": 2,
    "show_remaining_budget": True,
}
SUMMARIZABLE_STATUSES = frozenset({"finished", "blocked_unknown_usage"})
PROTECTION = {
    "review_round_tool_reserve": 1,
    "review_round_provider_reserve": 2,
    "review_round_input_reserve": 24000,
    "review_round_output_reserve": 16000,
}


def prepare(
    manifest_path: Path, config_path: Path, output: Path, prior_study: Path | None = None
) -> dict[str, Any]:
    """Freeze all four policies together, before seeing any model outcome."""
    if output.exists():
        raise FileExistsError("preserve the existing joint resource plan")
    source = json.loads(manifest_path.read_text())
    if source.get("split") != "dev" or not source.get("frozen"):
        raise ValueError("joint resources require frozen development observations")
    if len(source["cases"]) != 18 or len({c["source_session_id"] for c in source["cases"]}) != 2:
        raise ValueError("the protocol requires all eighteen cases from two development sessions")
    base = load_config(config_path, environment={})
    inputs = {
        str(config_path): file_sha256(config_path),
        str(manifest_path): file_sha256(manifest_path),
    }
    inputs["uv.lock"] = file_sha256(Path("uv.lock"))
    cases = (manifest_path.parent.parent / "cases").resolve()
    carried_usd = 0.0
    if prior_study is not None:
        predecessor = prior_study / "plan.json"
        previous = json.loads(predecessor.read_text())
        previous_hash = previous.pop("plan_hash")
        if content_hash(previous) != previous_hash:
            raise ValueError("predecessor joint resource plan changed")
        previous["plan_hash"] = previous_hash
        if previous["model_id"] != base.runtime.model_id or previous["case_directory"] != str(
            cases
        ):
            raise ValueError("predecessor must use the same model and cases")
        for name, digest in previous["inputs"].items():
            if file_sha256(Path(name)) != digest:
                raise ValueError(f"predecessor joint resource input changed: {name}")
        charges = expenses(previous)
        if any(c["unknown_attempts"] for c in charges.values()):
            raise ValueError("predecessor spending is unknown; do not reset its allowance")
        carried_usd = previous.get("carried_comparison_usd", 0.0) + sum(
            c["known_or_reserved_usd"] for c in charges.values()
        )
        inputs.update(previous["inputs"])
        inputs[str(predecessor)] = file_sha256(predecessor)
        for name in previous["policies"]:
            database = index_path(previous, name).with_name("runs.sqlite3")
            if database.exists():
                inputs[str(database)] = file_sha256(database)
    phase_ceiling = float(TOTALS["maximum_estimated_usd_per_experiment"]) - carried_usd / len(
        POLICIES
    )
    if phase_ceiling <= 0:
        raise ValueError("the original comparison allowance is exhausted")
    for path in sorted(cases.rglob("*")):
        if path.is_file():
            inputs[str(path)] = file_sha256(path)
    for dependency in (
        base.runtime.verified_rate_card,
        base.runtime.verified_endpoint_record,
        base.runtime.verified_capability_record,
    ):
        if dependency is not None:
            inputs[str(dependency)] = file_sha256(dependency)
    output.mkdir(parents=True)
    policies = {}
    for name, (query, joint) in POLICIES.items():
        settings = base.model_dump(mode="json")
        settings["runtime"].update(TOTALS)
        settings["runtime"]["maximum_estimated_usd_per_experiment"] = phase_ceiling
        settings["runtime"].update({k: v if joint else 0 for k, v in PROTECTION.items()})
        settings["runtime"].update(require_review_measurement=query, enable_live_calls=True)
        config = AppConfig.model_validate(settings)
        target = output / "policies" / name
        target.mkdir(parents=True)
        (target / "cases").symlink_to(cases, target_is_directory=True)
        manifest = source | {
            "freeze": freeze_hashes(config),
            "derived_from_manifest_sha256": file_sha256(manifest_path),
            "scope": "reused development observations; current runtime freeze",
        }
        cfg, labels = target / "configuration.json", target / "labels/manifest.json"
        labels.parent.mkdir()
        write_json_atomic(cfg, settings)
        write_json_atomic(labels, manifest)
        inputs.update({str(cfg): file_sha256(cfg), str(labels): file_sha256(labels)})
        policies[name] = {"config": str(cfg), "manifest": str(labels), "ceiling_usd": phase_ceiling}
    plan = {
        "method": METHOD,
        "source_tree_sha256": source_tree_hash(),
        "model_id": base.runtime.model_id,
        "case_directory": str(cases),
        "policies": policies,
        "carried_comparison_usd": carried_usd,
        "authorized_comparison_envelope_usd": 2.2,
        "inputs": inputs,
        "scope": "development only; no optimality or held-out claim",
        "selection": "combined only if all predeclared gates pass; otherwise shared",
        "selection_contract": [
            "Equal positive required-fact and pattern denominators across all policies.",
            "No decrease in completion, final review, required facts or pattern recovery.",
            "No increase in unsupported delivered claims; zero prohibited checked labels.",
            "All scheduled outcomes and paid usage known.",
            "At least one observed quality improvement or lower ledger cost.",
        ],
    }
    plan["plan_hash"] = content_hash(plan)
    write_json_atomic(output / "plan.json", plan)
    return plan


def load(output: Path) -> dict[str, Any]:
    plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
    expected = plan.pop("plan_hash")
    if content_hash(plan) != expected or plan["method"] != METHOD:
        raise ValueError("joint resource plan changed")
    plan["plan_hash"] = expected
    if plan["source_tree_sha256"] != source_tree_hash():
        raise ValueError("source changed after joint resource freeze")
    for name, digest in plan["inputs"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"joint resource input changed: {name}")
    for policy in plan["policies"].values():
        if (Path(policy["manifest"]).parent.parent / "cases").resolve() != Path(
            plan["case_directory"]
        ):
            raise ValueError("joint resource case link changed")
    return plan


def index_path(plan: dict[str, Any], name: str) -> Path:
    manifest = Path(plan["policies"][name]["manifest"])
    return (
        experiment_directory(manifest, f"exp_joint_{name}_{plan['plan_hash'][:12]}") / "index.json"
    )


def expenses(plan: dict[str, Any]) -> dict[str, Any]:
    """Read the ledger, including attempts absent from an interrupted index."""
    result = {}
    for name in plan["policies"]:
        database = index_path(plan, name).with_name("runs.sqlite3")
        if not database.exists():
            continue
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN usage_known=1 THEN COALESCE(actual_usd,0) "
                "ELSE reserved_usd END),0), COALESCE(SUM(usage_known=0),0) "
                "FROM actions WHERE kind='provider_attempt'"
            ).fetchone()
            assert row is not None
            result[name] = {"known_or_reserved_usd": row[0], "unknown_attempts": row[1]}
        finally:
            connection.close()
    return result


def run(output: Path, adapter_factory: ConfiguredAdapterFactory) -> dict[str, Any]:
    with RunLock(output / "locks", "joint-resources"):
        plan = load(output)
        progress = output / "progress.json"
        state: dict[str, Any] = (
            json.loads(progress.read_text())
            if progress.exists()
            else {"plan_hash": plan["plan_hash"], "phases": {}}
        )
        if state["plan_hash"] != plan["plan_hash"]:
            raise ValueError("progress belongs to another joint resource plan")
        for name, policy in plan["policies"].items():
            # load validates all immutable inputs again at every phase boundary.
            load(output)
            if any(record["unknown_attempts"] for record in expenses(plan).values()):
                state["status"] = "blocked_unknown_usage"
                break
            if (output / "STOP").exists():
                state["status"] = "stopped_between_phases"
                break
            config = load_config(Path(policy["config"]), environment={})
            manifest = Path(policy["manifest"])
            verify_freeze(json.loads(manifest.read_text())["freeze"], config)
            experiment = f"exp_joint_{name}_{plan['plan_hash'][:12]}"
            write_json_atomic(
                output / "active_phase.json",
                {
                    "phase": name,
                    "stop_file": str(experiment_directory(manifest, experiment) / "STOP"),
                },
            )
            index = run_benchmark(
                manifest,
                "live",
                [Arm.investigator_reviewer_gated],
                partial(adapter_factory, config),
                config=config,
                experiment_id=experiment,
            )
            state["phases"][name] = str(index)
            state["expenses"] = expenses(plan)
            if len(json.loads(index.read_text())["runs"]) != 18:
                state["status"] = "incomplete"
                break
            if any(record["unknown_attempts"] for record in state["expenses"].values()):
                state["status"] = "blocked_unknown_usage"
                break
            write_json_atomic(progress, state)
        else:
            state["status"] = "finished"
        write_json_atomic(progress, state)
        return state


def gates(metrics: dict[str, Any], paid: dict[str, Any]) -> dict[str, bool]:
    before, after = metrics["shared"], metrics["combined"]
    improved_quality = (
        any(
            after[key] > before[key]
            for key in (
                "completed",
                "final_independent_review",
                "required_facts_recovered",
                "patterns_recovered",
            )
        )
        or after["unsupported_delivered_claims"] < before["unsupported_delivered_claims"]
    )
    lower_cost = {"shared", "combined"} <= set(paid) and paid["combined"][
        "known_or_reserved_usd"
    ] < paid["shared"]["known_or_reserved_usd"]
    return {
        "comparable_quality_denominators": all(
            len({m[key] for m in metrics.values()}) == 1 and before[key] > 0
            for key in ("runs", "required_facts", "pattern_cases")
        ),
        "completion_not_decreased": after["completed"] >= before["completed"],
        "final_independent_review_not_decreased": after["final_independent_review"]
        >= before["final_independent_review"],
        "unsupported_not_increased": after["unsupported_delivered_claims"]
        <= before["unsupported_delivered_claims"],
        "required_fact_recovery_not_decreased": after["required_facts_recovered"]
        >= before["required_facts_recovered"],
        "pattern_recovery_not_decreased": after["patterns_recovered"]
        >= before["patterns_recovered"],
        "zero_prohibited_checked_labels": all(
            m["prohibited_checked_labels"] == 0 for m in metrics.values()
        ),
        "all_outcomes_known": all(m["missing_outcomes"] == 0 for m in metrics.values())
        and set(paid) == set(POLICIES)
        and all(p["unknown_attempts"] == 0 for p in paid.values()),
        "observed_quality_or_cost_benefit": improved_quality or lower_cost,
    }


def require_summarizable(state: dict[str, Any]) -> None:
    """Accept unknown usage after every phase ran, so all_outcomes_known can record it.

    The runner stops with blocked_unknown_usage once a phase holds an uncertain
    attempt. Refusing that state would hide a predeclared gate failure behind a
    missing summary. Earlier stops leave phases missing and are still refused.
    """
    if state.get("status") not in SUMMARIZABLE_STATUSES or set(state["phases"]) != set(POLICIES):
        raise ValueError("all four policies must finish before selection")


def summarize(output: Path, public: Path) -> dict[str, Any]:
    from celltrace.evaluation.scoring import score_experiment

    if public.exists():
        raise FileExistsError("preserve previous joint resource summary")
    with RunLock(output / "locks", "joint-resources"):
        plan = load(output)
        state = json.loads((output / "progress.json").read_text())
        require_summarizable(state)
        metrics = {}
        identities = None
        for name, index in state["phases"].items():
            manifest = Path(plan["policies"][name]["manifest"])
            labels = {c["case_id"]: c for c in json.loads(manifest.read_text())["cases"]}
            destination = output / "scoring" / public.name / name
            score_experiment(Path(index), manifest, destination)
            with (destination / "runs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            seen = {r["case_id"] for r in rows}
            if (
                len(rows) != 18
                or len(seen) != 18
                or (identities is not None and seen != identities)
            ):
                raise ValueError("policies did not compare the same eighteen observations")
            identities = seen
            false_labels = sum(
                bool(
                    set(json.loads(r["reported_patterns"] or "[]"))
                    & set(labels[r["case_id"]]["expected_patterns"]["forbidden"])
                    & {"active_rate_drop", "reported_radio_degradation", "stable_reported_radio"}
                )
                for r in rows
            )
            metrics[name] = {
                "runs": len(rows),
                "completed": sum(r["workflow_outcome"] == "completed" for r in rows),
                "final_independent_review": sum(
                    r["review_status"] == "independently_checked" for r in rows
                ),
                "outcomes": dict(Counter(r["workflow_outcome"] for r in rows)),
                "unsupported_delivered_claims": sum(_unsupported_delivered_claims(r) for r in rows),
                "prohibited_checked_labels": false_labels,
                "missing_outcomes": sum(not r.get("report_basis") for r in rows),
                "known_or_reserved_usd": sum(float(r["estimated_usd"] or 0) for r in rows),
                "required_facts_recovered": sum(
                    int(r["required_facts_recovered"] or 0) for r in rows
                ),
                "required_facts": sum(len(c["required_fact_kinds"]) for c in labels.values()),
                "pattern_cases": sum(
                    bool(c["expected_patterns"]["required"] or c["expected_patterns"]["forbidden"])
                    for c in labels.values()
                ),
                "patterns_recovered": sum(r["pattern_recovered"] == "True" for r in rows),
                "tool_requests": sum(int(r["tool_requests"] or 0) for r in rows),
                "provider_attempts": sum(int(r["provider_attempts"] or 0) for r in rows),
                "input_tokens": sum(int(r["input_tokens"] or 0) for r in rows),
                "output_tokens": sum(int(r["output_tokens"] or 0) for r in rows),
            }
        paid = expenses(plan)
        checks = gates(metrics, paid)
        summary = {
            "method": METHOD,
            "plan_hash": plan["plan_hash"],
            "model_id": plan["model_id"],
            "scope": plan["scope"],
            "study_status": state["status"],
            "metrics": metrics,
            "gates": checks,
            "expenses": paid,
            "carried_comparison_usd": plan["carried_comparison_usd"],
            "total_comparison_usd": plan["carried_comparison_usd"]
            + sum(p["known_or_reserved_usd"] for p in paid.values()),
            "selected_policy": "combined" if all(checks.values()) else "shared",
            "limitations": [
                "Two reused development sessions, one draw per case.",
                "No inferential superiority, optimal allowance or automatic default promotion.",
                "Current-round evidence does not establish physical cause or semantic usefulness.",
            ],
        }
        public.mkdir(parents=True)
        write_json_atomic(public / "summary.json", summary)
        plot(metrics, public / "policy_comparison.png")
        return summary


def plot(metrics: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(POLICIES)
    colors = ["#2568b8", "#ba4b20", "#087e58", "#7656a6"]
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ("completed", "final_independent_review", "known_or_reserved_usd"),
        ("Completed runs / 18", "Final revision independently checked / 18", "Provider cost (USD)"),
        strict=True,
    ):
        values = [metrics[name][key] for name in names]
        ax.barh(names, values, color=colors, height=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 20 if key != "known_or_reserved_usd" else max(values) * 1.35 or 0.1)
        for position, value in enumerate(values):
            label = f" ${value:.3f}" if key == "known_or_reserved_usd" else f" {value}/18"
            ax.text(value, position, label, va="center", fontsize=9)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Equal total resources · two reused development sessions · one draw", fontsize=11
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)
