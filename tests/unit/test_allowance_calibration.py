"""Calibration retains censored demand and refuses drift before adapter creation."""

import json
from pathlib import Path
from typing import Any

import pytest

from celltrace.canonical import content_hash
from celltrace.clock import ManualClock
from celltrace.config import load_config
from celltrace.evaluation.allowances import (
    derive_candidate,
    load_study,
    nearest_rank,
    phase_inputs,
    prepare_study,
    run_study,
    study_expenses,
)
from celltrace.evaluation.benchmark import FreezeMismatch
from celltrace.evaluation.commands import freeze_hashes
from celltrace.storage.run_store import ActionKind, Reservation, RunLimits, RunStore


def store_at(path: Path) -> RunStore:
    store = RunStore(path, ManualClock())
    store.create_run(
        "run",
        "case",
        "event",
        "investigator_reviewer_gated",
        "scripted_fixture",
        None,
        "snapshot",
        {},
        {},
        RunLimits(6, 3, 1000, 1000, None, 1000, 2, 1, 2),
    )
    return store


def profile_rows() -> list[dict[str, Any]]:
    return [
        {
            "investigator": n,
            "reviewer": 2,
            "censored": False,
            "unknown_attempts": 0,
            "source_session_id": "session_a" if n <= 9 else "session_b",
            "outcome": "completed",
        }
        for n in range(1, 19)
    ]


def frozen_study(tmp_path: Path) -> Path:
    config = load_config(
        environment={},
        cli_overrides={
            ("runtime", "verified_rate_card"): Path(
                "configs/rate_cards/deepseek_deepseek-v4.1-flash.yaml"
            ),
            ("runtime", "maximum_tool_requests"): 48,
            ("runtime", "show_remaining_budget"): True,
            ("runtime", "maximum_estimated_usd_per_experiment"): 0.6,
        },
    )
    config_path = tmp_path / "profile.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")))
    case_root = tmp_path / "original/cases"
    case_root.mkdir(parents=True)
    (case_root / "immutable.json").write_text('{"input": 1}')
    manifest = tmp_path / "original/labels/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "split": "dev",
                "frozen": True,
                "freeze": freeze_hashes(config),
                "cases": [
                    {"case_id": str(n), "source_session_id": "a" if n < 9 else "b"}
                    for n in range(18)
                ],
            }
        )
    )
    output = tmp_path / "study"
    prepare_study(manifest, config_path, output)
    return output


def test_candidate_is_descriptive_and_failed_trajectory_is_not_zero_demand() -> None:
    rows = profile_rows()
    rows[-1] |= {"censored": True, "outcome": "budget_exhausted", "investigator": 40}
    candidate = derive_candidate(rows)
    assert candidate["status"] == "conditional_censored"
    assert candidate["investigator_tools"] == 16
    assert candidate["reviewer_tool_reserve"] == 3
    assert candidate["total_tools"] == 19
    assert candidate["censored_rows"][0]["investigator"] == 40
    assert candidate["percentile_interval"] is None
    assert candidate["session_percentile_sensitivity"]["session_b"]["investigator"] == 17
    rows[0]["unknown_attempts"] = 1
    assert derive_candidate(rows)["completed"] == 16
    assert derive_candidate([rows[-1]])["status"] == "blocked_no_completed_profile"


def test_nearest_rank_boundary_is_predeclared() -> None:
    assert nearest_rank(list(range(1, 19))) == 17
    assert nearest_rank([1], 1) == 1
    for values, fraction in (([], 0.9), ([1], 0), ([1], 1.1)):
        with pytest.raises(ValueError):
            nearest_rank(values, fraction)


def test_changed_input_or_source_is_refused_before_provider_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = frozen_study(tmp_path)
    plan = load_study(output)
    from celltrace.evaluation import allowances

    monkeypatch.setattr(allowances, "source_tree_hash", lambda: "changed")
    with pytest.raises(ValueError, match="source changed"):
        run_study(output, lambda _: pytest.fail("adapter created before guard"))
    monkeypatch.undo()
    Path(plan["configuration"]).write_text("{}")
    with pytest.raises(ValueError, match="input changed"):
        run_study(output, lambda _: pytest.fail("adapter created before guard"))


def test_stop_prevents_first_dispatch_and_phase_inputs_cannot_drift(tmp_path: Path) -> None:
    output = frozen_study(tmp_path)
    (output / "STOP").touch()
    result = run_study(output, lambda _: pytest.fail("STOP must prevent all dispatch"))
    assert result["status"] == "stopped_between_phases"
    assert json.loads((output / "progress.json").read_text()) == result
    plan = load_study(output)
    manifest, config = phase_inputs(output, plan, "reserved", 20, 3)
    original = manifest.read_bytes()
    assert config.runtime.reviewer_tool_reserve == 3
    assert phase_inputs(output, plan, "reserved", 20, 3)[0].read_bytes() == original
    with pytest.raises(FreezeMismatch):
        phase_inputs(output, plan, "reserved", 21, 3)
    assert (manifest.parent.parent / "cases").is_symlink()


def test_plan_tampering_and_unknown_unindexed_cost_cannot_disappear(tmp_path: Path) -> None:
    output = frozen_study(tmp_path)
    plan = load_study(output)
    database = (
        output
        / "policies/profile/experiments"
        / f"exp_allowance_profile_{plan['plan_hash'][:12]}"
        / "runs.sqlite3"
    )
    store = store_at(database)
    store.reserve_action(
        "run",
        "unindexed",
        "unindexed",
        "investigator",
        ActionKind.provider_attempt,
        {},
        Reservation(usd=0.02),
    )
    store.close()
    assert study_expenses(output, plan)["profile"] == {
        "known_or_reserved_usd": 0.02,
        "unknown_attempts": 1,
    }
    plan["rule"]["percentile"] = 0.5
    (output / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="content or method changed"):
        load_study(output)
    plan.pop("plan_hash")
    assert content_hash(plan)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_negative_or_noninteger_reservations_cannot_release_budget(value: Any) -> None:
    with pytest.raises(ValueError):
        Reservation(tool_units=value)


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), True])
def test_reserved_cost_must_be_finite_nonnegative(value: float) -> None:
    with pytest.raises(ValueError):
        Reservation(usd=value)


def test_recommendation_refuses_missing_cost_and_wrong_gated_labels() -> None:
    from copy import deepcopy

    from celltrace.contracts.common import Arm
    from celltrace.evaluation.allowances import _quality_gates

    record = {
        "completed": 18,
        "unsupported_delivered_claims": 0,
        "prohibited_active_or_radio_label_runs": 0,
        "unknown_attempts": 0,
        "missing_outcomes": 0,
    }
    metrics = {
        policy: {arm.value: deepcopy(record) for arm in Arm}
        for policy in ("baseline", "shared", "reserved")
    }
    expenses = {"profile": {"unknown_attempts": 0}}
    assert all(_quality_gates(metrics, "shared", expenses).values())
    metrics["baseline"][Arm.investigator_gated.value]["prohibited_active_or_radio_label_runs"] = 1
    assert not _quality_gates(metrics, "shared", expenses)["no_prohibited_checked_labels"]
    expenses["profile"]["unknown_attempts"] = 1
    assert not _quality_gates(metrics, "shared", expenses)["no_unknown_usage"]
    metrics["reserved"][Arm.investigator_reviewer_gated.value]["completed"] = 17
    assert not _quality_gates(metrics, "shared", expenses)["completion_not_decreased"]


def test_fallback_submission_does_not_count_as_delivered_wrong_claims() -> None:
    from celltrace.evaluation.allowances import _unsupported_delivered_claims

    assert (
        _unsupported_delivered_claims(
            {"report_basis": "deterministic_fallback", "checkable_claims": 3, "supported_claims": 1}
        )
        == 0
    )
    assert (
        _unsupported_delivered_claims(
            {"report_basis": "model_findings", "checkable_claims": 3, "supported_claims": 1}
        )
        == 2
    )
    assert _unsupported_delivered_claims({"checkable_claims": 1, "supported_claims": 3}) == 0
