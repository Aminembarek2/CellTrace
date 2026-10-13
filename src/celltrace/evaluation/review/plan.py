"""Frozen review plan: judges, settings, schedule and the hashes that make it tamper evident.

A plan is written once. Loading it again verifies the plan hash, the runtime source tree, every
source packet file and every prepared judge input, so a changed rubric, schema, judge, setting or
report cannot be mixed into earlier results. An intentional change needs a new plan directory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from celltrace.canonical import canonical_json, content_hash, file_sha256
from celltrace.clock import Clock
from celltrace.evaluation.review.controls import VARIANTS, control_items, presentation_variant
from celltrace.evaluation.review.judge import ASSESSMENT_METHOD, CallLimits
from celltrace.evaluation.review.response_contract import (
    CONTRACT_VERSION,
    REQUEST_RUBRIC,
    WireAssessment,
    prepare_input,
    tool_schema,
)
from celltrace.evaluation.review.rubric import judge_input
from celltrace.providers.profiles import MODEL_PROFILES
from celltrace.providers.rate_card import RateCard
from celltrace.release import source_tree_hash

PROTOCOL_VERSION = "semantic_review_3"
PLAN_FILE = "plan.json"
PACKET_FILES = ("README.md", "annotations.csv", "auditor_mapping.json")
LIMITATIONS = (
    "Verdicts are automated model opinions under a frozen rubric, not human judgments.",
    "Control expectations are rubric assertions written for this project, not incident labels.",
    "Agreement between judges is descriptive and does not establish correctness.",
    "Control calibration is limited to constructed rubric examples.",
    "Phone measurements cannot establish a physical cause, whatever the verdicts say.",
)


@dataclass(frozen=True)
class JudgeProfile:
    """A fixed model profile with verified prices and the files that verified them."""

    model_id: str
    rate_card: RateCard
    configuration_path: str | None = None
    provenance_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_PROFILES:
            raise ValueError(f"judge {self.model_id} is not a fixed model profile")
        if self.rate_card.model_id != self.model_id:
            raise ValueError("judge rate card belongs to another model")


@dataclass(frozen=True)
class QualificationRule:
    """Frozen before any call. Each judge must meet every threshold on the controls."""

    minimum_canonical_agreement: float = 0.9
    minimum_valid_fraction: float = 0.9
    minimum_repeat_consistency: float = 0.9
    minimum_order_consistency: float = 0.9

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")


@dataclass(frozen=True)
class ReviewSettings:
    maximum_estimated_usd: float = 2.5
    repeat_sample_size: int = 6
    invalid_streak_limit: int = 3
    call_limits: CallLimits = field(default_factory=CallLimits)
    qualification: QualificationRule = field(default_factory=QualificationRule)
    controls_only: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_estimated_usd) or self.maximum_estimated_usd <= 0:
            raise ValueError("review ceiling must be positive and finite")
        if (
            isinstance(self.invalid_streak_limit, bool)
            or not isinstance(self.invalid_streak_limit, int)
            or self.invalid_streak_limit < 1
        ):
            raise ValueError("review needs a positive ceiling and invalid streak limit")
        if (
            isinstance(self.repeat_sample_size, bool)
            or not isinstance(self.repeat_sample_size, int)
            or self.repeat_sample_size < 0
        ):
            raise ValueError("repeat sample size cannot be negative")


def experiment_id(plan_hash: str) -> str:
    return "review_" + plan_hash[:16]


def run_id(plan_hash: str, item_id: str) -> str:
    return f"review_{plan_hash[:12]}_{item_id}"


def _opaque_id(source_id: str, model_id: str, stratum: str, variant: str) -> str:
    return "item_" + content_hash([source_id, model_id, stratum, variant])[:24]


def build_schedule(
    reports: list[dict[str, Any]], judges: list[JudgeProfile], settings: ReviewSettings
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Controls first for every judge, then every saved report, then a fixed repeat sample.

    The order matters: the runner checks each judge on its controls before it pays for reports.
    """
    schedule: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, Any]] = {}

    def add(item: dict[str, Any], model_id: str, stratum: str, variant: str) -> None:
        identifier = _opaque_id(item["item_id"], model_id, stratum, variant)
        inputs[identifier] = prepare_input(judge_input(item, identifier))
        schedule.append(
            {
                "item_id": identifier,
                "source_id": item["item_id"],
                "model_id": model_id,
                "stratum": stratum,
                "variant": variant,
            }
        )

    for control in control_items():
        for judge in judges:
            for variant in VARIANTS:
                shown = presentation_variant(control) if variant == "order" else control
                add(shown, judge.model_id, "control", variant)
    for report in [] if settings.controls_only else reports:
        for judge in judges:
            add(report, judge.model_id, "report", "canonical")
    repeat_sources = (
        []
        if settings.controls_only
        else sorted(
            (report for report in reports if report.get("phase") == "final"),
            key=lambda report: content_hash(report["item_id"]),
        )[: settings.repeat_sample_size]
    )
    for report in repeat_sources:
        for judge in judges:
            add(report, judge.model_id, "report", "repeat")
    return schedule, inputs


def prepare_plan(
    packet: Path,
    judges: list[JudgeProfile],
    settings: ReviewSettings,
    output: Path,
    clock: Clock,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"{output} exists. Preserve it and choose a new plan directory")
    if not judges or len({judge.model_id for judge in judges}) != len(judges):
        raise ValueError("provide one or more distinct judges")
    report_paths = sorted((packet / "items").glob("*.json"))
    reports = [json.loads(path.read_text()) for path in report_paths]
    if not reports:
        raise ValueError(f"no review items under {packet / 'items'}")
    if any(
        path.stem != report["item_id"] for path, report in zip(report_paths, reports, strict=True)
    ):
        raise ValueError("review item file names must match their identities")
    source_hashes = {str(path): file_sha256(path) for path in report_paths}
    for name in PACKET_FILES:
        source_hashes[str(packet / name)] = file_sha256(packet / name)
    for judge in judges:
        source_hashes.update(judge.provenance_hashes)
    schedule, inputs = build_schedule(reports, judges, settings)
    plan: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "assessment_method": ASSESSMENT_METHOD,
        "judges": [
            {
                "model_id": judge.model_id,
                "configuration_path": judge.configuration_path,
                "rate_card": judge.rate_card.model_dump(mode="json"),
            }
            for judge in judges
        ],
        "settings": asdict(settings),
        "response_contract": CONTRACT_VERSION,
        "rubric": REQUEST_RUBRIC,
        "rubric_sha256": content_hash(REQUEST_RUBRIC),
        "schema": WireAssessment.model_json_schema(),
        "packet": str(packet),
        "mapping_path": str(packet / "auditor_mapping.json"),
        "source_hashes": source_hashes,
        "runtime_source_sha256": source_tree_hash(),
        "input_hashes": {identifier: content_hash(item) for identifier, item in inputs.items()},
        "tool_schema_hashes": {
            identifier: content_hash(tool_schema(item)) for identifier, item in inputs.items()
        },
        "schedule": schedule,
        "controls": control_items(),
        "limitations": list(LIMITATIONS),
        "created_at": clock.utc_now_iso(),
    }
    plan["plan_hash"] = content_hash(plan)
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    (output / "assessments").mkdir()
    for identifier, item in inputs.items():
        (output / "inputs" / f"{identifier}.json").write_text(canonical_json(item) + "\n")
    (output / PLAN_FILE).write_text(json.dumps(plan, indent=2) + "\n")
    return {
        "plan_hash": plan["plan_hash"],
        "scheduled_items": len(schedule),
        "judges": [judge.model_id for judge in judges],
        "maximum_estimated_usd": settings.maximum_estimated_usd,
    }


def recorded_plan(output: Path) -> dict[str, Any]:
    """Verify the recorded identity for cost replay; this does not authorize live resumption."""
    plan: dict[str, Any] = json.loads((output / PLAN_FILE).read_text())
    recorded = plan.pop("plan_hash")
    if content_hash(plan) != recorded:
        raise ValueError("review plan changed after it was frozen")
    if plan["protocol_version"] not in ("semantic_review_1", "semantic_review_2", PROTOCOL_VERSION):
        raise ValueError(f"plan protocol {plan['protocol_version']} is not {PROTOCOL_VERSION}")
    plan["plan_hash"] = recorded
    return plan


def load_plan(output: Path) -> dict[str, Any]:
    plan = recorded_plan(output)
    if plan["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("historical review protocol is read-only; prepare a new plan")
    if source_tree_hash() != plan["runtime_source_sha256"]:
        raise ValueError("celltrace source changed after the plan was frozen")
    if (
        plan["response_contract"] != CONTRACT_VERSION
        or plan["rubric"] != REQUEST_RUBRIC
        or plan["schema"] != WireAssessment.model_json_schema()
    ):
        raise ValueError("rubric or verdict schema changed after the plan was frozen")
    for file_name, expected in plan["source_hashes"].items():
        if file_sha256(Path(file_name)) != expected:
            raise ValueError(f"frozen review source changed: {file_name}")
    for identifier, expected in plan["input_hashes"].items():
        prepared = json.loads((output / "inputs" / f"{identifier}.json").read_text())
        if content_hash(prepared) != expected:
            raise ValueError(f"frozen judge input changed: {identifier}")
        if content_hash(tool_schema(prepared)) != plan["tool_schema_hashes"][identifier]:
            raise ValueError(f"frozen tool schema changed: {identifier}")
    return plan


def settings_from_plan(plan: dict[str, Any]) -> ReviewSettings:
    stored = plan["settings"]
    return ReviewSettings(
        maximum_estimated_usd=stored["maximum_estimated_usd"],
        repeat_sample_size=stored["repeat_sample_size"],
        invalid_streak_limit=stored["invalid_streak_limit"],
        call_limits=CallLimits(**stored["call_limits"]),
        qualification=QualificationRule(**stored["qualification"]),
        controls_only=stored["controls_only"],
    )
