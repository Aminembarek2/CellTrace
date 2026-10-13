"""The review workflow end to end, offline, through an injected scripted judge."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from celltrace.clock import ManualClock
from celltrace.contracts.common import ExecutionMode
from celltrace.evaluation.review.controls import control_items
from celltrace.evaluation.review.judge import CallLimits
from celltrace.evaluation.review.ledger import load_results
from celltrace.evaluation.review.plan import (
    JudgeProfile,
    QualificationRule,
    ReviewSettings,
    load_plan,
    prepare_plan,
)
from celltrace.evaluation.review.response_contract import CATALOG_FIELD
from celltrace.evaluation.review.rubric import TOOL_NAME
from celltrace.evaluation.review.runner import run_review
from celltrace.evaluation.review.summary import summarize_review
from celltrace.providers.base import (
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    TokenEstimate,
    ToolCall,
)
from celltrace.providers.rate_card import RateCard

FIRST = "deepseek/deepseek-v4.1-flash"
SECOND = "z-ai/glm-5.3-flash"
EXPECTED_BY_SUMMARY = {
    control["findings"]["decision_summary"]: control["expected"] for control in control_items()
}
Behaviour = Callable[[dict[str, Any]], dict[str, Any]]


def rate_card(model_id: str) -> RateCard:
    return RateCard(
        model_id=model_id,
        provider_slug="fixture",
        currency="USD",
        unit="per_million_tokens",
        input=0.3,
        output=1.2,
        cache_read=0.0,
        cache_write_5m=0.0,
        cache_write_1h=0.0,
        tool_use_system_prompt_tokens_auto=0,
        source_url="fixture",
        checked_date="2026-10-09",
        cache_treatment="fixture",
        extra_charges="fixture",
    )


def judgment(
    verdict: str, pointer: str | None = None, excerpt: str | None = None
) -> dict[str, Any]:
    references = [] if pointer is None else [{"pointer": pointer, "excerpt": excerpt}]
    return {
        "verdict": verdict,
        "confidence": "moderate",
        "rationale": "Scripted judgment for the offline review workflow test.",
        "finding_references": references,
        "evidence_ids": [],
    }


def faithful(item: dict[str, Any]) -> dict[str, Any]:
    """Follows the rubric on controls, and calls every saved report qualified and useful."""
    findings = item["findings"]
    expected = EXPECTED_BY_SUMMARY.get(
        findings["decision_summary"],
        {"unsupported_causal_claim": "false", "useful_next_measurement": "true"},
    )
    causal = expected["unsupported_causal_claim"]
    useful = expected["useful_next_measurement"]
    useful_index = (
        next(
            index
            for index, text in enumerate(findings["next_measurements"])
            if text not in ("Collect more data.", "Explore the situation further.")
        )
        if useful == "true"
        else 0
    )
    return {
        "item_id": item["item_id"],
        "unsupported_causal_claim": judgment(
            causal, "/decision_summary", findings["decision_summary"][:20]
        )
        if causal == "true"
        else judgment(causal),
        "missed_alternative": (
            judgment("true", "/decision_summary", findings["decision_summary"][:20])
            | {
                "evidence_ids": [
                    next(
                        key
                        for key, value in item["cited_measurement_evidence"].items()
                        if value is not None
                    )
                ]
            }
        )
        if expected.get("missed_alternative") == "true"
        else judgment(expected.get("missed_alternative", "false")),
        "useful_next_measurement": judgment(
            useful,
            f"/next_measurements/{useful_index}",
            findings["next_measurements"][useful_index][:20],
        )
        if useful == "true"
        else judgment(useful),
    }


def contrarian(item: dict[str, Any]) -> dict[str, Any]:
    """Valid replies that contradict the rubric, so the judge fails its controls."""
    return {
        "item_id": item["item_id"],
        "unsupported_causal_claim": judgment("false"),
        "missed_alternative": judgment("false"),
        "useful_next_measurement": judgment("false"),
    }


def schema_breaking(item: dict[str, Any]) -> dict[str, Any]:
    """The Gemini failure from the live batch: a renamed verdict field."""
    broken = faithful(item)
    for criterion in ("unsupported_causal_claim", "missed_alternative", "useful_next_measurement"):
        broken[criterion]["judgment"] = broken[criterion].pop("verdict")
    return broken


def wire_reply(canonical: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    output = {"item_id": canonical["item_id"]}
    for criterion in ("unsupported_causal_claim", "missed_alternative", "useful_next_measurement"):
        judgment = dict(canonical[criterion])
        references = judgment.pop("finding_references", [])
        judgment["finding_reference_ids"] = [
            next(
                identifier
                for identifier, reference in item[CATALOG_FIELD].items()
                if reference["pointer"] == selected["pointer"]
                and selected["excerpt"] in reference["excerpt"]
            )
            for selected in references
        ]
        output[criterion] = judgment
    return output


class ScriptedJudge:
    execution_mode = ExecutionMode.scripted_fixture

    def __init__(self, model_id: str, behaviour: Behaviour):
        self.model_id = model_id
        self.behaviour = behaviour
        self.items_seen: list[str] = []

    def estimate_input_tokens(self, request: ProviderRequest) -> TokenEstimate:
        return TokenEstimate(input_tokens=600, method="fixture")

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        item = json.loads(request.messages[0]["content"])
        self.items_seen.append(item["item_id"])
        return ProviderResponse(
            assistant_content=[{"type": "text", "text": ""}],
            tool_calls=[
                ToolCall("call_fixture", TOOL_NAME, wire_reply(self.behaviour(item), item))
            ],
            text="",
            stop_reason="tool_calls",
            usage=ProviderUsage(True, input_tokens=600, output_tokens=300),
            reported_model=self.model_id,
            request_id="request_fixture",
            execution_mode=ExecutionMode.scripted_fixture,
        )

    def describe(self) -> dict[str, Any]:
        return {"model_id": self.model_id}


def report(item_id: str, phase: str, summary: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "phase": phase,
        "model_submission_available": True,
        "candidate": {"event_id": "evt_private_name"},
        "findings": {
            "decision_summary": summary,
            "cause_status": "unresolved",
            "hypotheses": [],
            "next_measurements": ["Measure offered demand to separate idle from limited delivery."],
        },
        "cited_measurement_evidence": {"ev_rate": {"tool_name": "compare_windows", "result": {}}},
    }


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    directory = tmp_path / "packet"
    (directory / "items").mkdir(parents=True)
    reports = [
        report("item_a_final", "final", "Application downlink fell. The cause is unresolved."),
        report("item_b_final", "final", "Delivered rate fell while traffic stayed active."),
        report("item_c_initial", "initial", "Rate fell. Radio impairment is not measured here."),
    ]
    for entry in reports:
        (directory / "items" / f"{entry['item_id']}.json").write_text(json.dumps(entry))
    mapping = [
        {"item_id": "item_a_final", "arm": "investigator_gated", "phase": "final"},
        {"item_id": "item_b_final", "arm": "investigator_reviewer_gated", "phase": "final"},
        {"item_id": "item_c_initial", "arm": "investigator_gated", "phase": "initial"},
    ]
    (directory / "auditor_mapping.json").write_text(json.dumps(mapping))
    (directory / "README.md").write_text("fixture packet\n")
    (directory / "annotations.csv").write_text("item_id,phase\n")
    return directory


def prepared(packet: Path, tmp_path: Path, ceiling: float = 1.0) -> Path:
    output = tmp_path / "review"
    judges = [JudgeProfile(FIRST, rate_card(FIRST)), JudgeProfile(SECOND, rate_card(SECOND))]
    prepare_plan(
        packet,
        judges,
        ReviewSettings(maximum_estimated_usd=ceiling, repeat_sample_size=2),
        output,
        ManualClock(),
    )
    return output


def judges(first: Behaviour, second: Behaviour) -> dict[str, ScriptedJudge]:
    return {FIRST: ScriptedJudge(FIRST, first), SECOND: ScriptedJudge(SECOND, second)}


def test_judges_never_see_expectations_arms_phases_or_names(packet: Path, tmp_path: Path) -> None:
    output = prepared(packet, tmp_path)
    for path in (output / "inputs").glob("*.json"):
        text = path.read_text()
        item = json.loads(text)
        assert set(item) == {
            "item_id",
            "model_submission_available",
            "candidate",
            "findings",
            "cited_measurement_evidence",
            CATALOG_FIELD,
        }
        assert item["item_id"] == path.stem and item["item_id"].startswith("item_")
        for hidden in ("expected", "investigator_gated", "evt_private_name", '"phase"'):
            assert hidden not in text


def test_schedule_checks_every_judge_on_controls_before_any_report(
    packet: Path, tmp_path: Path
) -> None:
    plan = load_plan(prepared(packet, tmp_path))
    strata = [entry["stratum"] for entry in plan["schedule"]]
    assert strata.index("report") == strata.count("control") == 20 * 3 * 2
    assert strata.count("report") == 3 * 2 + 2 * 2


def test_two_qualified_judges_form_a_panel_and_public_output_holds_no_report_text(
    packet: Path, tmp_path: Path
) -> None:
    output = prepared(packet, tmp_path)
    scripted = judges(faithful, faithful)
    progress = run_review(output, scripted.__getitem__, ManualClock())
    assert progress["recorded_items"] == progress["scheduled_items"] == 130
    summary = summarize_review(output, tmp_path / "public")
    assert summary["panel_status"] == "multiple_qualified_judges"
    assert all(record["qualified"] for record in summary["qualification"])
    groups = {(group["arm"], group["phase"]): group for group in summary["groups"]}
    assert groups[("investigator_gated", "final")]["panel"]["unsupported_causal_claim"] == {
        "false": 1
    }
    assert summary["known_or_reserved_usd"] > 0
    assert summary["gate"]["status"] == "passed"
    public = (tmp_path / "public" / "summary.json").read_text()
    assert "Application downlink fell" not in public and "decision_summary" not in public


def test_a_judge_that_fails_its_controls_is_never_paid_for_reports(
    packet: Path, tmp_path: Path
) -> None:
    output = prepared(packet, tmp_path)
    scripted = judges(faithful, contrarian)
    progress = run_review(output, scripted.__getitem__, ManualClock())
    assert len(scripted[SECOND].items_seen) == 60
    assert progress["not_dispatched"] == {"judge_not_qualified": 5}
    summary = summarize_review(output, tmp_path / "public")
    assert summary["panel_judges"] == [FIRST]
    assert summary["panel_status"] == "single_qualified_judge"
    assert summary["gate"]["status"] == "controls_failed"


def test_invalid_replies_open_the_circuit_after_three_items(packet: Path, tmp_path: Path) -> None:
    output = prepared(packet, tmp_path)
    scripted = judges(faithful, schema_breaking)
    progress = run_review(output, scripted.__getitem__, ManualClock())
    # Three items, each with its original attempt and one correction attempt.
    assert len(scripted[SECOND].items_seen) == 6
    assert progress["not_dispatched"] == {"judge_invalid_streak": 57 + 5}
    results = load_results(output, load_plan(output))
    assert sum(result is not None for result in results.values()) == 65 + 3


def test_edited_cost_or_input_is_refused(packet: Path, tmp_path: Path) -> None:
    output = prepared(packet, tmp_path)
    run_review(output, judges(faithful, faithful).__getitem__, ManualClock(), limit=2)
    saved = next((output / "assessments").glob("*.json"))
    record = json.loads(saved.read_text())
    record["usage"]["estimated_usd"] = 0.0
    saved.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="cost differs"):
        load_results(output, load_plan(output))
    prepared_input = next((output / "inputs").glob("*.json"))
    tampered = json.loads(prepared_input.read_text())
    tampered["findings"]["decision_summary"] += " Congestion is proven."
    prepared_input.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="judge input changed"):
        load_plan(output)


def test_stop_file_batch_limit_and_ceiling_end_the_run_between_items(
    packet: Path, tmp_path: Path
) -> None:
    output = prepared(packet, tmp_path)
    scripted = judges(faithful, faithful)
    assert (
        run_review(output, scripted.__getitem__, ManualClock(), limit=4)["dispatched_this_run"] == 4
    )
    (output / "STOP").touch()
    stopped = run_review(output, scripted.__getitem__, ManualClock())
    assert stopped["stopped_by"] == "stop_file" and stopped["dispatched_this_run"] == 0
    tight = prepared(packet, tmp_path / "tight", ceiling=0.00001)
    halted = run_review(tight, judges(faithful, faithful).__getitem__, ManualClock())
    assert halted["stopped_by"] == "budget_prevented"
    assert halted["known_or_reserved_usd"] == 0


def test_interrupted_result_publish_recovers_without_a_second_provider_call(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = prepared(packet, tmp_path)
    scripted = judges(faithful, faithful)
    replace = Path.replace

    def fail_publish(path: Path, target: Path) -> Path:
        if path.parent.name == "assessments":
            raise OSError("interrupted before atomic result publication")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_publish)
        with pytest.raises(OSError, match="interrupted"):
            run_review(output, scripted.__getitem__, ManualClock(), limit=1)
    assert not list((output / "assessments").glob("*.json"))
    assert sum(len(judge.items_seen) for judge in scripted.values()) == 1
    progress = run_review(output, scripted.__getitem__, ManualClock(), limit=1)
    assert progress["recorded_items"] == 1
    assert sum(len(judge.items_seen) for judge in scripted.values()) == 1
    assert progress["known_or_reserved_usd"] > 0
    with sqlite3.connect(output / "runs.sqlite3") as connection:
        assert connection.execute("SELECT execution_mode FROM runs").fetchone()[0] == (
            "scripted_fixture"
        )


@pytest.mark.parametrize("field", ["model_id", "assessment_method"])
def test_result_model_and_method_cannot_be_relabelled(
    packet: Path, tmp_path: Path, field: str
) -> None:
    output = prepared(packet, tmp_path)
    run_review(output, judges(faithful, faithful).__getitem__, ManualClock(), limit=1)
    saved = next((output / "assessments").glob("*.json"))
    result = json.loads(saved.read_text())
    result[field] = "different_identity"
    saved.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=r"saved model|saved assessment method"):
        load_results(output, load_plan(output))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_nonfinite_or_nonpositive_limits_cannot_enter_a_plan(value: float) -> None:
    with pytest.raises(ValueError):
        ReviewSettings(maximum_estimated_usd=value)
    with pytest.raises(ValueError):
        CallLimits(maximum_usd_per_item=value)
    with pytest.raises(ValueError):
        QualificationRule(minimum_canonical_agreement=value)


def test_controls_only_plan_cannot_dispatch_packet_reports(packet: Path, tmp_path: Path) -> None:
    output = tmp_path / "qualification"
    prepare_plan(
        packet,
        [JudgeProfile(FIRST, rate_card(FIRST))],
        ReviewSettings(controls_only=True),
        output,
        ManualClock(),
    )
    judge = ScriptedJudge(FIRST, faithful)
    progress = run_review(output, lambda model: judge, ManualClock())
    assert progress["recorded_items"] == progress["scheduled_items"] == 60
    summary = summarize_review(output, tmp_path / "qualification_summary")
    assert summary["scope"] == "judge_qualification_only"
    assert summary["gate"]["status"] == "passed"
    assert summary["gate"]["report_valid_fraction"] == {}
    assert summary["groups"] == []
    assert summary["qualification"][0]["criterion_denominators"]["missed_alternative"] == 10
