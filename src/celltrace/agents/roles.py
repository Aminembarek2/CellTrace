"""Role definitions: prompts, tool schemas and handoff messages.

The investigator and reviewer have separate system prompts and transcripts.
They share only the immutable snapshot through tools and the evidence store
through ids. The reviewer never receives the investigator's transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

from pydantic import BaseModel

from celltrace.canonical import canonical_json, content_hash, sha256_text
from celltrace.contracts.common import Role, ToolName
from celltrace.contracts.evidence import RunContext
from celltrace.contracts.findings import InvestigationFindings, ReviewResult
from celltrace.contracts.tools import TOOL_ARGUMENT_MODELS, TOOL_DESCRIPTIONS

SUBMIT_FINDINGS = "submit_findings"
SUBMIT_REVIEW = "submit_review"


def _prompt(name: str) -> str:
    return resources.files("celltrace.agents.prompts").joinpath(name).read_text(encoding="utf-8")


# Keys the model does not need. Count and length limits stay visible because live
# models repeatedly exceeded limits they could not see. Pydantic still
# enforces every constraint on each tool call and submission.
MODEL_FACING_DROPPED_KEYS = frozenset({"title", "discriminator", "default"})


def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema with references inlined and presentation only keywords removed."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = {**definitions[name], **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            result = {}
            for key, value in node.items():
                if key in MODEL_FACING_DROPPED_KEYS:
                    continue
                result["anyOf" if key == "oneOf" else key] = resolve(value)
            return result
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(schema)
    assert isinstance(resolved, dict)
    return resolved


@dataclass(frozen=True)
class RoleSpec:
    role: Role
    system_prompt: str
    tool_names: tuple[ToolName, ...]
    terminal_tool: str
    terminal_model: type[BaseModel]
    terminal_description: str

    def tool_definitions(self, strict: bool = False) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for tool in self.tool_names:
            definitions.append(
                {
                    "name": tool.value,
                    "description": TOOL_DESCRIPTIONS[tool],
                    "input_schema": inline_schema(TOOL_ARGUMENT_MODELS[tool]),
                }
            )
        definitions.append(
            {
                "name": self.terminal_tool,
                "description": self.terminal_description,
                "input_schema": inline_schema(self.terminal_model),
            }
        )
        if strict:
            for definition in definitions:
                definition["strict"] = True
        return definitions

    @property
    def prompt_hash(self) -> str:
        return sha256_text(self.system_prompt)


ALL_MEASUREMENT_TOOLS = tuple(ToolName)

INVESTIGATOR = RoleSpec(
    role=Role.investigator,
    system_prompt=_prompt("investigator.md"),
    tool_names=ALL_MEASUREMENT_TOOLS,
    terminal_tool=SUBMIT_FINDINGS,
    terminal_model=InvestigationFindings,
    terminal_description=(
        "Submit structured findings. This ends your turn. Every factual claim must cite "
        "evidence ids returned in this run."
    ),
)

REVIEWER = RoleSpec(
    role=Role.reviewer,
    system_prompt=_prompt("reviewer.md"),
    tool_names=ALL_MEASUREMENT_TOOLS,
    terminal_tool=SUBMIT_REVIEW,
    terminal_model=ReviewResult,
    terminal_description=(
        "Submit your review decision. Accept only after at least one independent measurement "
        "check, or record why no meaningful check was possible in review_limitation."
    ),
)

ROLE_SPECS: dict[Role, RoleSpec] = {Role.investigator: INVESTIGATOR, Role.reviewer: REVIEWER}


def role_identity() -> dict[str, str]:
    """Hashes of prompts and schemas, recorded with every run."""
    return {
        "investigator_prompt_sha256": INVESTIGATOR.prompt_hash,
        "reviewer_prompt_sha256": REVIEWER.prompt_hash,
        "investigator_tools_sha256": content_hash(INVESTIGATOR.tool_definitions()),
        "reviewer_tools_sha256": content_hash(REVIEWER.tool_definitions()),
    }


def candidate_brief(context: RunContext) -> dict[str, Any]:
    candidate = context.candidate
    return {
        "event_id": candidate.event_id,
        "session_id": context.session_id,
        "allowed_window": context.allowed_window.model_dump(),
        "event_window": candidate.event_window.model_dump(),
        "detector_output_to_check": {
            "detector": f"{candidate.detector_name} {candidate.detector_version}",
            "representative_reference_window": (
                candidate.representative_reference_window.model_dump()
            ),
            "representative_event_window": candidate.representative_event_window.model_dump(),
            "detection_time_ms": candidate.detection_time_ms,
            "reference_median_downlink_mbps_1s_bins": candidate.reference_median_mbps,
            "event_median_downlink_mbps_1s_bins": candidate.event_median_mbps,
            "flag_count": candidate.flag_count,
        },
    }


def investigator_opening(context: RunContext) -> dict[str, Any]:
    text = (
        "Investigate this candidate throughput drop and submit findings.\n"
        f"Candidate: {canonical_json(candidate_brief(context))}"
    )
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def reviewer_opening(
    context: RunContext, findings: dict[str, Any], review_round: int
) -> dict[str, Any]:
    cited = sorted(
        {
            evidence_id
            for claim in findings.get("factual_claims", [])
            for evidence_id in claim.get("evidence_ids", [])
        }
    )
    text = (
        f"Review round {review_round}. Review these structured findings independently.\n"
        f"Candidate: {canonical_json(candidate_brief(context))}\n"
        f"Findings: {canonical_json(findings)}\n"
        f"Cited evidence ids: {canonical_json(cited)}"
    )
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def tool_result_block(tool_use_id: str, payload: dict[str, Any], is_error: bool) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": canonical_json(payload),
    }
    if is_error:
        block["is_error"] = True
    return block
