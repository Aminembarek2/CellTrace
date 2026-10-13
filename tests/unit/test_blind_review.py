"""Blind review packet: blinding, calibration pairs and the preregistered statistics."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[2] / "scripts" / "blind_review.py"
specification = importlib.util.spec_from_file_location("blind_review", script)
assert specification is not None and specification.loader is not None
blind_review = importlib.util.module_from_spec(specification)
sys.modules["blind_review"] = blind_review
specification.loader.exec_module(blind_review)


def test_exact_statistics_match_known_values() -> None:
    assert blind_review.sign_test(14, 18) == pytest.approx(0.0309, abs=1e-4)
    assert blind_review.sign_test(9, 18) == 1.0
    assert blind_review.clopper_pearson(23, 27) == pytest.approx([0.6627, 0.9581], abs=1e-4)
    assert blind_review.krippendorff_alpha([["a", "a"], ["b", "b"], ["a", "a"]], "nominal") == 1.0
    disagreeing = [["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]]
    assert blind_review.krippendorff_alpha(disagreeing, "nominal") < 0


def _report(event: str, summary: str) -> dict[str, object]:
    return {
        "event_id": event,
        "arm": "secret_arm",
        "run_id": "secret_run",
        "model_id": "secret_model",
        "final_findings": {"decision_summary": summary},
        "verified_observation_patterns": ["active_rate_drop"],
        "verified_claims": [],
        "qualified_hypotheses": [],
        "missing_evidence": ["No cell load counters."],
        "next_measurements": ["Cell load."],
    }


def test_packet_hides_identity_and_scores_a_consistent_rater(tmp_path: Path) -> None:
    agent, template = tmp_path / "agent", tmp_path / "template"
    agent.mkdir()
    template.mkdir()
    for index in range(8):
        event = f"evt_{index}"
        (agent / f"{event}.json").write_text(json.dumps(_report(event, "Agent summary.")))
        (template / f"{event}.json").write_text(json.dumps(_report(event, "Template summary.")))
    packet = tmp_path / "packet"
    blind_review.prepare(agent, template, packet, 3)
    text = (packet / "packet.md").read_text()
    assert "secret_arm" not in text and "secret_run" not in text and "secret_model" not in text
    mapping = json.loads((packet / "private_unblinding" / "mapping.json").read_text())
    assert sum(entry["kind"] == "calibration" for entry in mapping["pairs"]) == 6
    rows = []
    for entry in mapping["pairs"]:
        good = "A" if entry["A"] in ("agent", "original") else "B"
        bad = "B" if good == "A" else "A"
        row = {"pair_id": entry["pair_id"], "preference": good, "comment": ""}
        for name in blind_review.CRITERIA:
            row[f"{good}_{name}"], row[f"{bad}_{name}"] = "5", "2"
        rows.append(row)
    sheet = tmp_path / "rater.csv"
    with sheet.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    blind_review.score(packet, [sheet, sheet], tmp_path / "score.json")
    result = json.loads((tmp_path / "score.json").read_text())
    rater = result["per_rater"][0]
    assert rater["agent_preferred"] == 8 and rater["template_preferred"] == 0
    assert rater["calibration_detected"] == "6 of 6"
    assert result["agreement"]["preference_percent"] == 1.0


class _PreferSummary:
    """Scripted rater that prefers whichever side carries a given summary sentence."""

    def __init__(self, preferred_summary: str | None) -> None:
        self.preferred_summary = preferred_summary

    def next_turn(self, request: object) -> object:
        from celltrace.providers.scripted import ScriptedTurn

        message = request.messages[0]["content"]  # type: ignore[attr-defined]
        first = message.split("## Report B")[0]
        if self.preferred_summary is None:
            winner = "A"
        else:
            winner = "A" if self.preferred_summary in first else "B"
        loser = "B" if winner == "A" else "A"
        arguments = {
            f"report_{winner}": {name: 5 for name in blind_review.CRITERIA},
            f"report_{loser}": {name: 2 for name in blind_review.CRITERIA},
            "preference": winner,
            "comment": "scripted",
        }
        return ScriptedTurn("", blind_review.RATER_TOOL, arguments)


def _scripted_rating(tmp_path: Path, policy: object, name: str) -> tuple[Path, dict[str, object]]:
    from celltrace.providers.scripted import ScriptedModelAdapter

    packet = tmp_path / "packet"
    if not packet.exists():
        agent, template = tmp_path / "agent", tmp_path / "template"
        agent.mkdir()
        template.mkdir()
        for index in range(8):
            event = f"evt_{index}"
            (agent / f"{event}.json").write_text(json.dumps(_report(event, "Agent summary.")))
            (template / f"{event}.json").write_text(json.dumps(_report(event, "Template one.")))
        blind_review.prepare(agent, template, packet, 3)
    rates = blind_review.RateCard(
        model_id="scripted-fixture",
        currency="USD",
        unit="per_million_tokens",
        input=0,
        output=0,
        cache_write_5m=0,
        cache_write_1h=0,
        cache_read=0,
        tool_use_system_prompt_tokens_auto=0,
        source_url="none",
        checked_date="none",
        cache_treatment="none",
        extra_charges="none",
    )
    settings = blind_review.RaterSettings(model_id="scripted-fixture", maximum_usd=0.01)
    adapter = ScriptedModelAdapter({blind_review.RATER_ROLE: policy})
    summary = blind_review.rate(packet, tmp_path / name, adapter, rates, settings, "f" * 64)
    return packet, summary


def test_automated_rater_fills_the_sheet_in_both_orders(tmp_path: Path) -> None:
    packet, summary = _scripted_rating(tmp_path, _PreferSummary("Agent summary."), "rater")
    assert summary["pairs_rated_in_both_orders"] == 14
    assert summary["order_consistent_preferences"] == 14
    assert len(list((tmp_path / "rater" / "presentations").glob("*.json"))) == 28
    blind_review.score(packet, [tmp_path / "rater" / "ratings.csv"], tmp_path / "score.json")
    rater = json.loads((tmp_path / "score.json").read_text())["per_rater"][0]
    assert rater["agent_preferred"] == 8 and rater["template_preferred"] == 0
    assert rater["calibration_detected"] == "6 of 6" and rater["qualified"]


def test_position_bias_becomes_a_tie_and_fails_qualification(tmp_path: Path) -> None:
    packet, summary = _scripted_rating(tmp_path, _PreferSummary(None), "always_a")
    assert summary["order_consistent_preferences"] == 0
    rows = list(csv.DictReader((tmp_path / "always_a" / "ratings.csv").open()))
    assert {row["preference"] for row in rows} == {"tie"}
    assert {row["A_grounded"] for row in rows} == {"3.5"}
    blind_review.score(packet, [tmp_path / "always_a" / "ratings.csv"], tmp_path / "score.json")
    rater = json.loads((tmp_path / "score.json").read_text())["per_rater"][0]
    assert rater["ties"] == 8 and not rater["qualified"]


def test_rerun_resumes_without_new_provider_calls(tmp_path: Path) -> None:
    _scripted_rating(tmp_path, _PreferSummary("Agent summary."), "rater")
    first = (tmp_path / "rater" / "ratings.csv").read_text()
    _, summary = _scripted_rating(tmp_path, _PreferSummary(None), "rater")
    assert (tmp_path / "rater" / "ratings.csv").read_text() == first
    assert summary["order_consistent_preferences"] == 14


def test_rating_validation_rejects_coercions() -> None:
    scores = {name: 4 for name in blind_review.CRITERIA}
    valid = {"report_A": scores, "report_B": scores, "preference": "tie", "comment": ""}
    assert blind_review.validate_rating(valid) is valid
    for broken in (
        {**valid, "report_A": {**scores, "clarity": True}},
        {**valid, "report_B": {**scores, "clarity": 6}},
        {**valid, "preference": "both"},
        {key: value for key, value in valid.items() if key != "comment"},
    ):
        with pytest.raises(ValueError):
            blind_review.validate_rating(broken)
    swapped = blind_review.in_packet_frame({**valid, "preference": "A"}, "swapped")
    assert swapped["preference"] == "B"


def test_second_rendering_shows_the_basis_of_each_verdict() -> None:
    report = _report("evt_0", "Summary.")
    report["verified_claims"] = [
        {
            "claim_id": "c1",
            "claim_type": "activity",
            "state": "active",
            "fraction": 1.0,
            "denominator": 32,
            "window": {"start_ms": 0, "end_ms": 32000},
        }
    ]
    report["qualified_hypotheses"] = [
        {
            "hypothesis_id": "application_demand",
            "status": "contradicted",
            "contradicting_claim_ids": ["c1"],
            "supporting_claim_ids": [],
            "assumptions": ["Active state means the application wanted data."],
            "discriminating_measurement": "Application request log.",
        }
    ]
    first = blind_review.render(report)
    assert "active share 1 of 32 samples" in first and "Contradicted by" not in first
    second = blind_review.render(report, rendering=2)
    assert "active in 100% of 32 samples" in second
    assert "Contradicted by: active in 100% of 32 samples (0 to 32000 ms)." in second
    assert "Assumes: Active state means the application wanted data." in second
