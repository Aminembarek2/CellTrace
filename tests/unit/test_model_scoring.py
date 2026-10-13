"""Architecture pairs stay within one model and exclude unavailable schedules."""

from typing import Any

import pytest

from celltrace.evaluation.scoring import paired_differences


def measurement_row(model_id: str, arm: str, completeness: float) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "case_id": "shared_case",
        "source_session_id": "shared_session",
        "arm": arm,
        "workflow_outcome": "completed",
        "fact_completeness": completeness,
        "pattern_recovered": completeness == 1,
        "factual_support": completeness,
    }


def test_architecture_pair_cannot_mix_models_on_the_same_case() -> None:
    rows = [
        measurement_row("first_model", "investigator_gated", 0.2),
        measurement_row("second_model", "investigator_reviewer_gated", 1.0),
    ]
    with pytest.raises(ValueError, match="single model"):
        paired_differences(rows, 42)


def test_unavailable_review_is_not_a_measured_paired_result() -> None:
    first = measurement_row("first_model", "investigator_gated", 0.2)
    second = measurement_row("first_model", "investigator_reviewer_gated", 1.0)
    metric = "fact_completeness:investigator_reviewer_gated_minus_investigator_gated"
    result = paired_differences([first, second], 42)[metric]
    assert result["pairs"] == 1
    assert result["mean_difference"] == pytest.approx(0.8)
    second["workflow_outcome"] = "unavailable"
    result = paired_differences([first, second], 42)[metric]
    assert result["pairs"] == 0
    assert result["mean_difference"] is None
