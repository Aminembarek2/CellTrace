"""Complete frozen reviewer experiment with positive, negative and crash controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from celltrace.agents.scripted_policies import ScriptedReviewer
from celltrace.clock import SystemClock
from celltrace.config import load_config
from celltrace.data.snapshot import run_ingest
from celltrace.evaluation.cases import build_cases
from celltrace.evaluation.planted_commands import load_plan, prepare, run_study, study_config
from celltrace.providers.base import FailureKind, ProviderRequest
from celltrace.providers.scripted import ScriptedModelAdapter, ScriptedTurn, tool_results
from celltrace.storage.run_store import RunStore
from tests.integration.test_evaluation import write_long_archive


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    base = tmp_path_factory.mktemp("planted")
    write_long_archive(base / "raw")
    snapshot = run_ingest(base / "raw", base / "processed")
    manifest = build_cases(
        snapshot,
        base / "raw",
        base / "cases",
        "dev",
        load_config(environment={}),
        maximum_base_windows=2,
        variants_per_base=8,
    )
    directory = base / "plan"
    return directory, prepare(manifest, directory)


class LocatedReviewer:
    """Preprogrammed positive protocol control, not a semantic inference algorithm."""

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        results = [r for r in tool_results(request.messages) if r.get("evidence_id")]
        if not results:
            return ScriptedReviewer().next_turn(request)
        return ScriptedTurn(
            "Locating one section for the positive harness control.",
            "submit_review",
            {
                "decision": "unresolved",
                "disputed_claim_ids": [],
                "disputed_finding_paths": ["/decision_summary", "/hypotheses/0/status"],
                "independent_evidence_ids": [results[-1]["evidence_id"]],
                "alternative_checks": [],
                "requested_queries": [],
                "reason_summary": "An unsupported interpretation needs unavailable measurements.",
            },
        )


def test_full_freeze_negative_and_positive_controls_and_no_dispatch_on_replay(
    study: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    directory, plan = study
    assert len(plan["items"]) == 84 and len(plan["verifier_controls"]) == 24
    assert len({item["finding_id"] for item in plan["items"]}) == 42
    # Radio exclusion from RSRP alone is now rejected by code before any model sees it.
    assert {c["kind"] for c in plan["verifier_controls"]} >= {"unobservable_exclusion"}
    assert all(c["rejected"] for c in plan["verifier_controls"])
    # Every base records which rejected observation labels its control dropped, if any.
    assert len(plan["control_repairs"]) == 6
    assert all(
        set(repair["removed_unsupported_patterns"])
        <= {
            "reported_radio_degradation",
            "stable_reported_radio",
            "active_rate_drop",
            "qualified_cell_transition",
        }
        or repair["removed_unsupported_patterns"] == []
        for repair in plan["control_repairs"]
    )
    negative = ScriptedModelAdapter({"reviewer": ScriptedReviewer()})
    execution = tmp_path / "negative"
    config = study_config(execution)
    result = run_study(directory, execution, config, negative, None)
    assert result["status"] == "finished" and result["known_or_reserved_usd"] == 0
    assert result["families"]["physical_cause_new"]["localized_challenges"] == 0
    assert result["families"]["correct_control"]["decision_objections"] == 0
    attempts = negative.attempts
    assert run_study(directory, execution, config, negative, None) == result
    assert negative.attempts == attempts
    positive_path = tmp_path / "positive"
    positive = run_study(
        directory,
        positive_path,
        study_config(positive_path),
        ScriptedModelAdapter({"reviewer": LocatedReviewer()}),
        None,
    )
    assert positive["families"]["physical_cause_original"]["localized_challenges"] == 12
    assert positive["families"]["window_scope_new"]["localized_challenges"] == 12
    assert positive["families"]["window_scope_new"]["draw_consistency"] == {
        "0_of_2": 0,
        "1_of_2": 0,
        "2_of_2": 6,
    }
    assert positive["families"]["correct_control"]["decision_objections"] == 12
    assert positive["families"]["correct_control"]["disputed_target_paths"] == {
        "/decision_summary": 12,
    }


def test_unknown_dispatch_stops_whole_study_and_cannot_repeat(
    study: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    directory, _ = study
    execution = tmp_path / "unknown"
    adapter = ScriptedModelAdapter(
        {"reviewer": ScriptedReviewer()}, failures={1: FailureKind.uncertain}
    )
    result = run_study(directory, execution, study_config(execution), adapter, None)
    assert result["status"] == "blocked_unknown_usage"
    assert not result["all_outcomes_known"] and result["scheduled"] == 84
    attempts = adapter.attempts
    run_study(directory, execution, study_config(execution), adapter, None)
    assert adapter.attempts == attempts
    store = RunStore(execution / "runs.sqlite3", SystemClock())
    try:
        assert (
            sum(store.usage(run["run_id"]).unknown_usage_attempts for run in store.list_runs()) == 1
        )
    finally:
        store.close()


def test_modified_plan_and_heldout_selection_refused(
    study: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    _, plan = study
    changed = tmp_path / "changed"
    changed.mkdir()
    edited = dict(plan)
    edited["study_ceiling_usd"] = 999
    (changed / "plan.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="changed"):
        load_plan(changed)
    heldout = tmp_path / "manifest.json"
    heldout.write_text(json.dumps({"split": "heldout"}))
    with pytest.raises(ValueError, match="development"):
        prepare(heldout, tmp_path / "should_not_exist")
    assert not (tmp_path / "should_not_exist").exists()


def test_version_4_uses_distinct_windows_one_wording_and_its_own_ceiling(
    tmp_path: Path,
) -> None:
    # Each base on its own window, a chosen wording and draw count.
    write_long_archive(tmp_path / "raw")
    snapshot = run_ingest(tmp_path / "raw", tmp_path / "processed")
    manifest_path = build_cases(
        snapshot,
        tmp_path / "raw",
        tmp_path / "cases",
        "dev",
        load_config(environment={}),
        maximum_base_windows=2,
        variants_per_base=8,
    )
    directory = tmp_path / "plan_v4"
    plan = prepare(
        manifest_path, directory, bases_per_family=1, phrasings=("new",), draws=1, ceiling_usd=1.5
    )
    assert plan["method"] == "planted_interpretations_4"
    cases = {case["case_id"]: case for case in json.loads(manifest_path.read_text())["cases"]}
    bases = sorted({item["base_id"] for item in plan["items"]})
    windows = {
        (cases[base]["source_session_id"], cases[base]["source_confirmation_ms"]) for base in bases
    }
    assert len(windows) == len(bases) == 2  # two windows cannot carry three distinct bases
    families = {item["label"]["family"] for item in plan["items"]}
    assert families == {
        "correct_control",
        "physical_cause_new",
        "capacity_conflation_new",
        "window_scope_new",
    }
    assert len(plan["items"]) == 4 * len(bases)
    assert load_plan(directory)["study_ceiling_usd"] == 1.5
    execution = directory / "offline"
    summary = run_study(
        directory,
        execution,
        study_config(execution, ceiling_usd=1.5),
        ScriptedModelAdapter({"reviewer": ScriptedReviewer()}),
        None,
    )
    assert summary["status"] == "finished"
    assert summary["method"] == "planted_interpretations_4"
    assert summary["draws_per_finding"] == 1
    assert set(summary["families"]) == families
