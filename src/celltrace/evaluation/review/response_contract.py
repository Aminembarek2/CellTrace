"""Item-bound terminal contract: select existing references rather than inventing quotes.

The wire object is deliberately distinct from the canonical historical assessment. Decoding
resolves selected IDs through a deterministic catalog and still invokes its exact-reference
validator. A provider's strict flag is a request, never a substitute for this validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from celltrace.canonical import content_hash
from celltrace.evaluation.review.rubric import RUBRIC, Assessment, Verdict, validate_assessment

CONTRACT_VERSION = "reference_catalog_2"
CATALOG_FIELD = "finding_reference_catalog"
REQUEST_RUBRIC = (
    RUBRIC.split("For every judgment provide", 1)[0]
    + """
For each judgment submit verdict, confidence, rationale, finding_reference_ids and evidence_ids.
Verdict true/false may be strings or JSON Booleans. Unknown/not_applicable must be strings.
Every field is required; use empty lists when no reference is needed. Select finding reference
IDs only from finding_reference_catalog. Code resolves those IDs to exact finding excerpts and
JSON pointers; do not output quotes, pointers or a finding_references field. Measurement objects
are not finding references. Select evidence IDs only from cited_measurement_evidence.
A true verdict needs at least one finding reference. A true missed_alternative also needs a
measurement ID. A true useful_next_measurement must cite a proposed next measurement or a
hypothesis's discriminating measurement and explain what different outcomes would distinguish.
Do not require every imaginable alternative. Use unknown when evidence cannot settle an omission.
Confidence is low/moderate/high. Each rationale is 20 to 600 characters. Select at most two
finding references and eight evidence IDs per criterion. Submit exactly the supplied item_id.
These are automated assessments, not human annotations or physical ground truth.
"""
)


class WireJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    verdict: Verdict | bool
    confidence: Literal["low", "moderate", "high"]
    rationale: str = Field(min_length=20, max_length=600)
    finding_reference_ids: list[str] = Field(max_length=2)
    evidence_ids: list[str] = Field(max_length=8)

    @field_validator("verdict", mode="before")
    @classmethod
    def canonical_verdict(cls, value: Any) -> Any:
        # Exactly two representations of the same Boolean; never coerce numbers or yes/no.
        if isinstance(value, bool):
            return "true" if value else "false"
        return value


class WireAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    item_id: str
    unsupported_causal_claim: WireJudgment
    missed_alternative: WireJudgment
    useful_next_measurement: WireJudgment


def reference_catalog(findings: Any) -> dict[str, dict[str, str]]:
    """Cover all nonempty text leaves in traversal order, with exact bounded chunks."""
    catalog: dict[str, dict[str, str]] = {}

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                escaped = key.replace("~", "~0").replace("/", "~1")
                visit(value[key], pointer + "/" + escaped)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, pointer + f"/{index}")
        elif isinstance(value, str) and value.strip():
            remaining = value
            offset = 0
            while remaining:
                end = min(240, len(remaining))
                if end < len(remaining):
                    boundary = remaining.rfind(" ", 0, end)
                    if boundary > 0:
                        end = boundary
                excerpt = remaining[:end]
                remaining = remaining[end:]
                start = offset
                offset += end
                if not excerpt.strip():
                    continue
                identifier = "ref_" + content_hash([pointer, start, excerpt])[:24]
                catalog[identifier] = {"pointer": pointer, "excerpt": excerpt}

    visit(findings, "")
    return catalog


def prepare_input(item: dict[str, Any]) -> dict[str, Any]:
    return item | {CATALOG_FIELD: reference_catalog(item["findings"])}


def _catalog(item: dict[str, Any]) -> dict[str, dict[str, str]]:
    expected = reference_catalog(item["findings"])
    if item.get(CATALOG_FIELD) != expected:
        raise ValueError("reference catalog differs from the supplied findings")
    return expected


def tool_schema(item: dict[str, Any]) -> dict[str, Any]:
    catalog = _catalog(item)
    schema = WireAssessment.model_json_schema()
    schema["properties"]["item_id"] = {"type": "string", "enum": [item["item_id"]]}
    judgment = schema["$defs"]["WireJudgment"]["properties"]
    evidence_ids = [
        key for key, value in item["cited_measurement_evidence"].items() if value is not None
    ]
    for field, values in (("finding_reference_ids", list(catalog)), ("evidence_ids", evidence_ids)):
        judgment[field]["items"] = {"type": "string"}
        if values:
            judgment[field]["items"]["enum"] = values
        else:
            judgment[field]["maxItems"] = 0
    schema["$defs"]["WireJudgment"]["properties"]["verdict"] = (
        {"anyOf": [{"type": "string", "enum": ["true", "false", "unknown"]}, {"type": "boolean"}]}
        if item["model_submission_available"]
        else {"type": "string", "enum": ["not_applicable"]}
    )
    return schema


def decode_assessment(payload: dict[str, Any], item: dict[str, Any]) -> Assessment:
    catalog = _catalog(item)
    wire = WireAssessment.model_validate(payload)
    normalized: dict[str, Any] = {"item_id": wire.item_id}
    for name in tuple(WireAssessment.model_fields)[1:]:
        judgment: WireJudgment = getattr(wire, name)
        ids = judgment.finding_reference_ids
        if len(set(ids)) != len(ids) or len(set(judgment.evidence_ids)) != len(
            judgment.evidence_ids
        ):
            raise ValueError("duplicate reference or evidence IDs")
        if any(identifier not in catalog for identifier in ids):
            raise ValueError("finding reference ID is absent from this item")
        normalized[name] = {
            "verdict": judgment.verdict,
            "confidence": judgment.confidence,
            "rationale": judgment.rationale,
            "finding_references": [catalog[identifier] for identifier in ids],
            "evidence_ids": judgment.evidence_ids,
        }
    return validate_assessment(normalized, item)
