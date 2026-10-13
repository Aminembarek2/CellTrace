"""Exploration outputs on the fixture snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from celltrace.analysis.explore import run_explore
from celltrace.data import fixture

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def explore_outputs(
    fixture_snapshot_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, Any], Path]:
    out_dir = tmp_path_factory.mktemp("fixture_explore")
    return run_explore(fixture_snapshot_dir, out_dir), out_dir


def test_files_are_written(explore_outputs: tuple[dict[str, Any], Path]) -> None:
    summary, out_dir = explore_outputs
    for name in (
        "rate_activity_timeline.png",
        "radio_rate_small_multiples.png",
        "transition_aligned_windows.png",
    ):
        content = (out_dir / name).read_bytes()
        assert content.startswith(PNG_SIGNATURE) and len(content) > 10_000
    stored = json.loads((out_dir / "explore_summary.json").read_text())
    assert stored["snapshot_id"] == summary["snapshot_id"]
    assert stored["session_count"] == 10


def test_summaries_and_transition_counts(
    explore_outputs: tuple[dict[str, Any], Path], fixture_session_ids: dict[str, str]
) -> None:
    summary, _ = explore_outputs
    sessions = {item["session_id"]: item for item in summary["sessions"]}
    missing = sessions[fixture_session_ids[fixture.MISSING_VALUES_MEMBER]]
    assert missing["interval_counts_ms"] == {"0": 1, "1000": 54, "2000": 1, "3000": 1}
    assert missing["missing_fraction"]["downlink_mbps"] == pytest.approx(1 / 58, abs=1e-6)
    assert summary["transitions_by_application_cohort"]["file_download"] == {
        "serving_cell_change": 2,
        "technology_change": 4,
        "reported_identity_change": 1,
    }
    assert summary["value_rejections_by_reason"] == {
        "android_unavailable_sentinel": 2,
        "suspected_fill_value": 3,
    }


def test_figure_choices_are_deterministic(
    explore_outputs: tuple[dict[str, Any], Path], fixture_session_ids: dict[str, str]
) -> None:
    summary, _ = explore_outputs
    timeline = summary["figures"]["rate_activity_timeline"]
    assert timeline["session_id"] == fixture_session_ids[fixture.GAP_MEMBER]
    assert timeline["segment_gaps_shown"] == 1
    aligned = summary["figures"]["transition_aligned_windows"]
    assert aligned["aligned_serving_cell_changes"] == 2
    assert aligned["counterexample"]["session_id"] == fixture_session_ids[fixture.STEADY_MEMBER]
    assert aligned["counterexample"]["later_time_ms"] == 30000
    assert aligned["counterexample"]["relative_change"] == 0.0


def test_capability_matrix_matches_snapshot(explore_outputs: tuple[dict[str, Any], Path]) -> None:
    summary, out_dir = explore_outputs
    matrix = json.loads((out_dir / "capability_matrix.json").read_text())
    assert matrix["snapshot_id"] == summary["snapshot_id"]
    assert matrix["capabilities"]["neighbour_metrics"]["status"] == "unavailable"
    assert all(entry["reason"] for entry in matrix["capabilities"].values())
