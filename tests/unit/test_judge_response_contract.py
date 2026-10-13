"""Regression boundaries for item-bound response references and fresh session selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from celltrace.canonical import file_sha256
from celltrace.evaluation.cases import reserved_sessions
from celltrace.evaluation.review.controls import control_items
from celltrace.evaluation.review.response_contract import (
    CATALOG_FIELD,
    decode_assessment,
    prepare_input,
    reference_catalog,
    tool_schema,
)
from celltrace.evaluation.review.rubric import judge_input


def sample() -> tuple[dict[str, Any], dict[str, Any]]:
    item = prepare_input(judge_input(control_items()[0], "item_test"))
    reference = next(
        key for key, value in item[CATALOG_FIELD].items() if value["pointer"] == "/decision_summary"
    )
    judgment = {
        "verdict": "true",
        "confidence": "moderate",
        "rationale": "The asserted physical cause exceeds the supplied measurements.",
        "finding_reference_ids": [reference],
        "evidence_ids": [],
    }
    reply = {
        "item_id": item["item_id"],
        "unsupported_causal_claim": judgment,
        "missed_alternative": judgment | {"verdict": "unknown", "finding_reference_ids": []},
        "useful_next_measurement": judgment | {"verdict": "false", "finding_reference_ids": []},
    }
    return item, reply


def test_json_booleans_normalize_without_changing_unknown() -> None:
    item, reply = sample()
    reply["unsupported_causal_claim"]["verdict"] = True
    reply["useful_next_measurement"]["verdict"] = False
    decoded = decode_assessment(reply, item)
    assert decoded.unsupported_causal_claim.verdict == "true"
    assert decoded.useful_next_measurement.verdict == "false"
    assert decoded.missed_alternative.verdict == "unknown"
    assert reply["unsupported_causal_claim"]["verdict"] is True


@pytest.mark.parametrize("invalid", [0, 1, None, "yes", "no", "True", [], {}])
def test_verdict_boundary_rejects_coercion(invalid: Any) -> None:
    item, reply = sample()
    reply["unsupported_causal_claim"]["verdict"] = invalid
    with pytest.raises(ValueError):
        decode_assessment(reply, item)


def test_boolean_is_not_a_missing_submission_verdict() -> None:
    item, reply = sample()
    item["model_submission_available"] = False
    reply["unsupported_causal_claim"]["verdict"] = False
    with pytest.raises(ValueError):
        decode_assessment(reply, item)


def test_exact_catalog_chunks_escape_keys_and_cover_all_text() -> None:
    text = "Meaningful text. " * 45
    findings = {"a/b~": [text, "second finding"], "number": 2, "blank": ""}
    catalog = reference_catalog(findings)
    assert catalog == reference_catalog(findings)
    excerpts = [r["excerpt"] for r in catalog.values() if r["pointer"] == "/a~1b~0/0"]
    assert "".join(excerpts) == text and max(map(len, excerpts)) <= 240
    assert all(reference["pointer"].startswith("/a~1b~0/") for reference in catalog.values())


def test_decoding_keeps_exact_quotes_and_native_schema_has_no_free_quote_fields() -> None:
    item, reply = sample()
    assessment = decode_assessment(reply, item)
    reference = assessment.unsupported_causal_claim.finding_references[0]
    assert reference.excerpt in item["findings"]["decision_summary"]
    schema = tool_schema(item)
    properties = schema["$defs"]["WireJudgment"]["properties"]
    assert "finding_references" not in properties
    assert set(properties["finding_reference_ids"]["items"]["enum"]) == set(item[CATALOG_FIELD])
    assert schema["properties"]["item_id"]["enum"] == [item["item_id"]]
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "change", ["foreign_id", "duplicate", "missing_evidence", "forged_catalog", "wrong_item"]
)
def test_forged_or_incomplete_provider_objects_are_rejected(change: str) -> None:
    item, reply = sample()
    if change == "foreign_id":
        reply["unsupported_causal_claim"]["finding_reference_ids"] = ["ref_other_item"]
    elif change == "duplicate":
        reply["unsupported_causal_claim"]["finding_reference_ids"] *= 2
    elif change == "missing_evidence":
        del reply["unsupported_causal_claim"]["evidence_ids"]
    elif change == "forged_catalog":
        item[CATALOG_FIELD][next(iter(item[CATALOG_FIELD]))]["excerpt"] = "fabricated cause"
    else:
        reply["item_id"] = "another_item"
    with pytest.raises(ValueError):
        decode_assessment(reply, item)


def test_fresh_session_selection_checks_old_exposure_and_provenance(tmp_path: Path) -> None:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE sessions (session_id VARCHAR, application_cohort VARCHAR)")
    connection.executemany(
        "INSERT INTO sessions VALUES (?, ?)", [("old", "file_download"), ("fresh", "file_download")]
    )
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"cases": [{"source_session_id": "old"}]}))
    selection = {
        "version": "fresh_session_selection_1",
        "source_snapshot_id": "snap_fixture",
        "dev": ["old"],
        "heldout": ["fresh"],
        "prior_manifests": {str(previous): file_sha256(previous)},
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    splits, provenance = reserved_sessions(connection, "snap_fixture", path)
    assert splits == {"dev": ["old"], "heldout": ["fresh"]}
    assert provenance["previously_constructed_sessions"] == ["old"]
    for heldout in (["old"], ["foreign"], ["fresh", "fresh"]):
        path.write_text(json.dumps(selection | {"heldout": heldout, "dev": []}))
        with pytest.raises(ValueError):
            reserved_sessions(connection, "snap_fixture", path)
    path.write_text(json.dumps(selection))
    previous.write_text(json.dumps({"cases": []}))
    with pytest.raises(ValueError, match="prior case manifest changed"):
        reserved_sessions(connection, "snap_fixture", path)
    connection.close()
