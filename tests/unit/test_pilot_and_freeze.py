"""Pilot bounds and evaluation identity include settings and implementation content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from celltrace.config import load_config
from celltrace.evaluation.benchmark import FreezeMismatch, verify_freeze
from celltrace.evaluation.commands import freeze_hashes
from celltrace.evaluation.pilot import PILOT_FAMILIES, prepare_pilot, run_pilot
from celltrace.providers.profiles import COMPARISON_MODEL_IDS, MODEL_PROFILES


def test_pilot_preselects_six_gated_and_three_reviewed_cases_per_profile(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "dev",
                "frozen": True,
                "cases": [
                    {"case_id": f"case_{position}", "family": family.value}
                    for position, family in enumerate(PILOT_FAMILIES)
                ],
            }
        )
    )
    plan = prepare_pilot(manifest, Path("configs"), tmp_path / "plan.json")
    assert len(plan["schedule"]) == 27
    for model_id in MODEL_PROFILES:
        scheduled = [entry for entry in plan["schedule"] if entry["model_id"] == model_id]
        assert sum(entry["arm"] == "investigator_gated" for entry in scheduled) == 6
        assert sum(entry["arm"] == "investigator_reviewer_gated" for entry in scheduled) == 3
    with pytest.raises(FileExistsError):
        prepare_pilot(manifest, Path("configs"), tmp_path / "plan.json")
    payload = json.loads(manifest.read_text())
    payload["split"] = "heldout"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="development"):
        prepare_pilot(manifest, Path("configs"), tmp_path / "another_plan.json")


def test_heldout_freeze_refuses_a_model_or_budget_change() -> None:
    configuration = load_config(environment={})
    frozen = freeze_hashes(configuration)
    assert frozen["source_tree_sha256"]
    verify_freeze(frozen, configuration)
    changed_model = load_config(
        environment={},
        cli_overrides={
            ("runtime", "model_id"): COMPARISON_MODEL_IDS[0],
        },
    )
    with pytest.raises(FreezeMismatch, match="runtime_limits"):
        verify_freeze(frozen, changed_model)
    stricter = load_config(
        environment={},
        cli_overrides={
            ("runtime", "maximum_input_tokens"): 20000,
        },
    )
    with pytest.raises(FreezeMismatch, match="runtime_limits"):
        verify_freeze(frozen, stricter)


def test_operator_stop_file_prevents_any_dispatch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "dev",
                "frozen": True,
                "cases": [
                    {"case_id": f"case_{position}", "family": family.value}
                    for position, family in enumerate(PILOT_FAMILIES)
                ],
            }
        )
    )
    plan = prepare_pilot(manifest, Path("configs"), tmp_path / "plan.json")
    experiment_directory = tmp_path / "pilot" / f"exp_pilot_{plan['plan_hash'][:16]}"
    experiment_directory.mkdir(parents=True)
    (experiment_directory / "STOP").touch()
    result = run_pilot(tmp_path / "plan.json", tmp_path / "pilot", live=False)
    assert result["stopped_by_operator"] is True
    assert result["runs"] == [] and result["attempted"] == 0
    assert result["scheduled"] == 27


def test_scoring_an_explicit_index_requires_the_same_experiment(tmp_path: Path) -> None:
    import argparse

    from celltrace.evaluation.commands import command_score

    index = tmp_path / "index.json"
    index.write_text(json.dumps({"experiment_id": "exp_pilot_a", "runs": []}))
    arguments = argparse.Namespace(
        experiment="exp_pilot_b",
        manifest=str(tmp_path / "manifest.json"),
        out=str(tmp_path / "scores"),
        index=str(index),
    )
    with pytest.raises(ValueError, match="another experiment"):
        command_score(arguments)
