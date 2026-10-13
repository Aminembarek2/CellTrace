"""Configuration strictness, canonical hashing and contract validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from celltrace.canonical import NonFiniteValueError, canonical_json, content_hash
from celltrace.config import RuntimeSection, load_config
from celltrace.contracts.common import TimeWindow
from celltrace.contracts.findings import InvestigationFindings, ReviewResult
from celltrace.providers.profiles import COMPARISON_MODEL_IDS, PRIMARY_MODEL_ID


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "runtime:\n  maximum_tool_request: 5\n")
    with pytest.raises(ValidationError, match="maximum_tool_request"):
        load_config(path, environment={})


def test_negative_and_inconsistent_limits_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, "runtime:\n  maximum_tool_requests: -1\n"), environment={})
    with pytest.raises(ValidationError, match="per attempt"):
        load_config(
            write_yaml(
                tmp_path,
                "runtime:\n  maximum_output_tokens: 100\n"
                "  maximum_output_tokens_per_attempt: 200\n",
            ),
            environment={},
        )


def test_live_calls_need_model_and_rate_card(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "runtime:\n  enable_live_calls: true\n")
    with pytest.raises(ValidationError, match="rate_card"):
        load_config(path, environment={})
    with pytest.raises(ValidationError, match="model_id"):
        load_config(path, environment={"CELLTRACE_MODEL": "openrouter/auto"})


def test_precedence_cli_over_environment_over_file(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, f"runtime:\n  model_id: {PRIMARY_MODEL_ID}\n")
    assert load_config(path, environment={}).runtime.model_id == PRIMARY_MODEL_ID
    from_environment = load_config(path, environment={"CELLTRACE_MODEL": COMPARISON_MODEL_IDS[0]})
    assert from_environment.runtime.model_id == COMPARISON_MODEL_IDS[0]
    from_cli = load_config(
        path,
        cli_overrides={("runtime", "model_id"): COMPARISON_MODEL_IDS[1]},
        environment={"CELLTRACE_MODEL": COMPARISON_MODEL_IDS[0]},
    )
    assert from_cli.runtime.model_id == COMPARISON_MODEL_IDS[1]


def test_redacted_summary_has_no_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value-123")
    summary = canonical_json(load_config(environment={}).redacted_summary())
    assert "secret-value-123" not in summary
    assert '"openrouter_api_key_present":true' in summary


@pytest.mark.parametrize(
    "model_id",
    ["openrouter/auto", "deepseek/latest", "vendor/model-x", "deepseek/deepseek-v4.1-flash:free"],
)
def test_unapproved_profiles_are_rejected(model_id: str) -> None:
    with pytest.raises(ValidationError, match="model_id"):
        RuntimeSection(model_id=model_id)


def test_fixed_effort_and_probe_requirement_cannot_be_silently_disabled() -> None:
    assert RuntimeSection().maximum_input_tokens == 40_000
    assert RuntimeSection().model_effort == "high"
    with pytest.raises(ValidationError):
        RuntimeSection.model_validate({"model_effort": "low"})
    with pytest.raises(ValidationError):
        RuntimeSection.model_validate({"require_live_capability_probe": False})
    with pytest.raises(ValidationError):
        RuntimeSection.model_validate({"maximum_estimated_usd_per_run": math.nan})


def test_canonical_json_is_order_independent_and_rejects_nonfinite() -> None:
    assert content_hash({"b": 1, "a": [1.5, None]}) == content_hash({"a": [1.5, None], "b": 1})
    with pytest.raises(NonFiniteValueError):
        canonical_json({"value": math.nan})
    with pytest.raises(NonFiniteValueError):
        canonical_json([math.inf])


def test_time_window_is_half_open_and_nonempty() -> None:
    assert TimeWindow(start_ms=0, end_ms=1).length_ms == 1
    with pytest.raises(ValidationError):
        TimeWindow(start_ms=5, end_ms=5)
    with pytest.raises(ValidationError):
        TimeWindow(start_ms=6, end_ms=5)
    outer = TimeWindow(start_ms=0, end_ms=10)
    assert outer.contains(TimeWindow(start_ms=0, end_ms=10))
    assert not outer.contains(TimeWindow(start_ms=5, end_ms=11))


def minimal_findings(**overrides: object) -> dict[str, object]:
    findings: dict[str, object] = {
        "event_id": "evt_1",
        "eligibility": {"status": "unknown", "supporting_claim_ids": [], "reason": "none"},
        "factual_claims": [
            {
                "claim_type": "availability",
                "claim_id": "c1",
                "metric": "downlink_mbps",
                "window": {"start_ms": 0, "end_ms": 1000},
                "basis": "native",
                "valid_count": 1,
                "missing_count": 0,
                "evidence_ids": ["ev_1"],
            }
        ],
        "observation_patterns": [],
        "hypotheses": [],
        "missing_evidence": [],
        "next_measurements": [],
        "decision_summary": "short",
        "cause_status": "unresolved",
    }
    findings.update(overrides)
    return findings


def test_findings_reject_resolved_cause_extra_fields_and_nan() -> None:
    InvestigationFindings.model_validate(minimal_findings())
    with pytest.raises(ValidationError):
        InvestigationFindings.model_validate(minimal_findings(cause_status="resolved"))
    with pytest.raises(ValidationError):
        InvestigationFindings.model_validate(minimal_findings(root_cause="congestion"))
    claim = dict(minimal_findings()["factual_claims"][0])  # type: ignore[index]
    claim.update(claim_type="metric_statistic", statistic="mean", value=math.nan, unit="Mbps")
    with pytest.raises(ValidationError):
        InvestigationFindings.model_validate(minimal_findings(factual_claims=[claim]))


def test_findings_reject_duplicate_and_unknown_claim_references() -> None:
    claims = minimal_findings()["factual_claims"]
    with pytest.raises(ValidationError, match="unique"):
        InvestigationFindings.model_validate(
            minimal_findings(factual_claims=[*claims, *claims])  # type: ignore[misc]
        )
    with pytest.raises(ValidationError, match="unknown claim ids"):
        InvestigationFindings.model_validate(
            minimal_findings(
                eligibility={"status": "eligible", "supporting_claim_ids": ["nope"], "reason": "r"}
            )
        )


def test_claim_without_evidence_is_rejected() -> None:
    claim = dict(minimal_findings()["factual_claims"][0])  # type: ignore[index]
    claim["evidence_ids"] = []
    with pytest.raises(ValidationError):
        InvestigationFindings.model_validate(minimal_findings(factual_claims=[claim]))


def test_review_decision_is_constrained() -> None:
    ReviewResult.model_validate({"decision": "accept", "reason_summary": "ok"})
    with pytest.raises(ValidationError):
        ReviewResult.model_validate({"decision": "approve", "reason_summary": "ok"})


def test_model_facing_schemas_show_the_limits_python_enforces() -> None:
    from celltrace.agents.roles import INVESTIGATOR, REVIEWER

    investigator = {tool["name"]: tool for tool in INVESTIGATOR.tool_definitions()}
    findings = investigator["submit_findings"]["input_schema"]["properties"]
    assert findings["factual_claims"]["maxItems"] == 16
    assert findings["decision_summary"]["maxLength"] == 1200
    assert investigator["summarize_window"]["input_schema"]["properties"]["metrics"]["maxItems"]
    review = {tool["name"]: tool for tool in REVIEWER.tool_definitions()}["submit_review"]
    assert review["input_schema"]["properties"]["reason_summary"]["maxLength"] == 1200
    assert "title" not in json.dumps(review["input_schema"])
