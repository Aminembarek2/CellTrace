"""Selection refuses missing, uncertain or incomparable development evidence."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from celltrace.clock import ManualClock
from celltrace.evaluation import joint_resources as study
from celltrace.storage.run_store import ActionKind, ActionState, Reservation, RunLimits, RunStore


def observations(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dev"
    (root / "labels").mkdir(parents=True)
    (root / "cases").mkdir()
    (root / "cases" / "input.json").write_text('{"unchanged": true}')
    manifest = root / "labels/manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "dev",
                "frozen": True,
                "cases": [{"case_id": str(i), "source_session_id": str(i // 9)} for i in range(18)],
            }
        )
    )
    config = tmp_path / "config.yaml"
    rate_card = tmp_path / "rates.yaml"
    rate_card.write_text("offline placeholder; no provider constructed in these tests")
    config.write_text(f"runtime:\n  provider: openrouter\n  verified_rate_card: {rate_card}\n")
    return manifest, config


def test_policy_freeze_preserves_cases_and_refuses_retargeted_link(tmp_path: Path) -> None:
    manifest, config = observations(tmp_path)
    output = tmp_path / "study"
    plan = study.prepare(manifest, config, output)
    assert study.load(output) == plan
    for name, (query, joint) in study.POLICIES.items():
        settings = json.loads(Path(plan["policies"][name]["config"]).read_text())["runtime"]
        assert settings["require_review_measurement"] == query
        assert settings["maximum_tool_requests"] == 27
        assert settings["review_round_output_reserve"] == (16000 if joint else 0)
    other = tmp_path / "other"
    other.mkdir()
    link = output / "policies/shared/cases"
    link.unlink()
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="case link changed"):
        study.load(output)


def test_changed_case_refuses_dispatch_before_provider_factory(tmp_path: Path) -> None:
    manifest, config = observations(tmp_path)
    output = tmp_path / "study"
    study.prepare(manifest, config, output)
    (manifest.parent.parent / "cases/input.json").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        study.run(output, lambda _: pytest.fail("no network dispatch after input mutation"))


@pytest.mark.parametrize("unknown", [False, True])
def test_corrected_freeze_carries_prior_charges_and_refuses_unknown_spending(
    tmp_path: Path, unknown: bool
) -> None:
    manifest, config = observations(tmp_path)
    prior = tmp_path / "prior"
    previous = study.prepare(manifest, config, prior)
    database = study.index_path(previous, "shared").with_name("runs.sqlite3")
    store = RunStore(database, ManualClock())
    store.create_run(
        "prior_run",
        "case",
        "event",
        "investigator_reviewer_gated",
        "scripted_fixture",
        None,
        "snapshot",
        {},
        {},
        RunLimits(3, 3, 1000, 1000, 1.0, 60, 2, 1),
    )
    action = store.reserve_action(
        "prior_run",
        "prior_call",
        "prior_call",
        "investigator",
        ActionKind.provider_attempt,
        {},
        Reservation(usd=0.1),
    )
    store.mark_started(action)
    if unknown:
        store.mark_uncertain(action, "lost response")
    else:
        store.complete_action(
            action, ActionState.completed, 0, usage={"usage_known": True, "usd": 0.1}
        )
    store.close()
    if unknown:
        with pytest.raises(ValueError, match="spending is unknown"):
            study.prepare(manifest, config, tmp_path / "corrected", prior)
        assert not (tmp_path / "corrected").exists()
        return
    result = study.prepare(manifest, config, tmp_path / "corrected", prior)
    assert result["carried_comparison_usd"] == pytest.approx(0.1)
    assert len({p["ceiling_usd"] for p in result["policies"].values()}) == 1
    assert 0.1 + sum(p["ceiling_usd"] for p in result["policies"].values()) == pytest.approx(2.2)
    database.write_bytes(database.read_bytes() + b"changed after freeze")
    with pytest.raises(ValueError, match="input changed"):
        study.load(tmp_path / "corrected")


def comparison() -> tuple[dict, dict]:
    record = {
        "runs": 18,
        "completed": 16,
        "final_independent_review": 16,
        "unsupported_delivered_claims": 0,
        "prohibited_checked_labels": 0,
        "missing_outcomes": 0,
        "required_facts": 72,
        "required_facts_recovered": 64,
        "pattern_cases": 18,
        "patterns_recovered": 16,
    }
    metrics = {name: deepcopy(record) for name in study.POLICIES}
    paid = {name: {"unknown_attempts": 0, "known_or_reserved_usd": 0.3} for name in study.POLICIES}
    return metrics, paid


def test_better_completion_does_not_waive_final_review_or_unknown_usage() -> None:
    metrics, paid = comparison()
    metrics["combined"]["completed"] = 17
    assert all(study.gates(metrics, paid).values())
    metrics["combined"].update(completed=18, final_independent_review=15)
    assert not study.gates(metrics, paid)["final_independent_review_not_decreased"]
    paid["query"]["unknown_attempts"] = 1
    assert not study.gates(metrics, paid)["all_outcomes_known"]
    paid.pop("query")
    assert not study.gates(metrics, paid)["all_outcomes_known"]


@pytest.mark.parametrize(
    "field,value,gate",
    [
        ("required_facts_recovered", 63, "required_fact_recovery_not_decreased"),
        ("patterns_recovered", 15, "pattern_recovery_not_decreased"),
        ("required_facts", 71, "comparable_quality_denominators"),
        ("pattern_cases", 17, "comparable_quality_denominators"),
    ],
)
def test_more_completed_runs_cannot_hide_less_complete_or_incomparable_reports(
    field: str, value: int, gate: str
) -> None:
    metrics, paid = comparison()
    metrics["combined"].update(completed=18, final_independent_review=18)
    metrics["combined"][field] = value
    assert study.gates(metrics, paid)["observed_quality_or_cost_benefit"]
    assert not study.gates(metrics, paid)[gate]


def test_equal_quality_requires_lower_ledger_cost_and_positive_denominators() -> None:
    metrics, paid = comparison()
    assert not study.gates(metrics, paid)["observed_quality_or_cost_benefit"]
    paid["combined"]["known_or_reserved_usd"] = 0.31
    assert not study.gates(metrics, paid)["observed_quality_or_cost_benefit"]
    paid["combined"]["known_or_reserved_usd"] = 0.29
    assert all(study.gates(metrics, paid).values())
    for record in metrics.values():
        record["required_facts"] = 0
    assert not study.gates(metrics, paid)["comparable_quality_denominators"]


@pytest.mark.parametrize(
    "status,phases,accepted",
    [
        ("finished", list(study.POLICIES), True),
        ("blocked_unknown_usage", list(study.POLICIES), True),
        ("blocked_unknown_usage", ["shared", "query"], False),
        ("incomplete", list(study.POLICIES), False),
        ("stopped_between_phases", ["shared"], False),
    ],
)
def test_unknown_usage_after_every_phase_reaches_the_gate_instead_of_refusal(
    status: str, phases: list[str], accepted: bool
) -> None:
    state = {"status": status, "phases": {name: f"{name}/index.json" for name in phases}}
    if accepted:
        study.require_summarizable(state)
        return
    with pytest.raises(ValueError, match="must finish"):
        study.require_summarizable(state)
