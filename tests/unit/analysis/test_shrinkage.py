"""Empirical Bayes cell effects: formulas, selection and the verification checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from celltrace.analysis import shrinkage
from celltrace.analysis.cells import Visit


def cell(name: str, deviations: list[float], days: int = 2) -> shrinkage.CellData:
    return shrinkage.CellData(name, "LTE", days, tuple(deviations))


def test_within_variance_pools_by_degrees_of_freedom() -> None:
    cells = [cell("a", [0.0, 2.0]), cell("b", [1.0, 1.0, 4.0]), cell("c", [5.0])]
    # Squares 2 and 6 over 1 + 2 degrees of freedom; single visit cells add nothing.
    assert shrinkage.pooled_within_variance(cells) == pytest.approx(8 / 3)


def test_between_variance_is_zero_without_spread_and_positive_with_it() -> None:
    errors = np.full(4, 0.1)
    assert shrinkage.between_variance(np.zeros(4), errors) == 0.0
    spread = np.array([-1.0, -0.5, 0.5, 1.0])
    expected = (np.sum(spread**2 / 0.01) - 4) / np.sum(1 / errors**2)
    assert shrinkage.between_variance(spread, errors) == pytest.approx(expected)


def test_posterior_shrinks_by_reliability() -> None:
    posterior = shrinkage.normal_posterior(
        np.array([-0.6, -0.6]), np.array([0.1, 0.4]), 0.04, shrinkage.SLOW_THRESHOLD
    )
    assert posterior.reliability == pytest.approx([0.04 / 0.05, 0.04 / 0.2])
    assert posterior.mean == pytest.approx(posterior.reliability * -0.6)
    # The precise cell is more surely slow than the noisy one with the same raw mean.
    assert posterior.tail_probability[0] > posterior.tail_probability[1]


def test_selection_keeps_the_expected_false_share_within_the_limit() -> None:
    tail = np.array([0.5, 0.99, 0.1, 0.95, 0.9])
    selected = shrinkage.select_by_false_discovery(tail, 0.10)
    assert selected.tolist() == [False, True, False, True, True]
    assert not shrinkage.select_by_false_discovery(np.array([0.5, 0.4]), 0.10).any()


def test_rank_intervals_are_tight_for_separated_cells() -> None:
    posterior = shrinkage.Posterior(
        np.array([-2.0, 0.0, 2.0]), np.full(3, 0.01), np.zeros(3), np.ones(3)
    )
    bounds = shrinkage.rank_intervals(posterior, np.random.default_rng(1), draws=500)
    assert bounds.tolist() == [[1, 1], [2, 2], [3, 3]]


def test_nonparametric_prior_is_a_distribution_and_tails_are_probabilities() -> None:
    generator = np.random.default_rng(2)
    means = np.concatenate([generator.normal(-0.6, 0.1, 10), generator.normal(0, 0.1, 30)])
    posterior = shrinkage.npmle_posterior(means, np.full(40, 0.1), shrinkage.SLOW_THRESHOLD)
    assert ((posterior.tail_probability >= 0) & (posterior.tail_probability <= 1)).all()
    assert posterior.tail_probability[:10].mean() > posterior.tail_probability[10:].mean()
    _, weights = shrinkage.npmle_prior(means, np.full(40, 0.1))
    assert weights.sum() == pytest.approx(1.0)


def visit(day: str, name: str, log_rate: float) -> Visit:
    return Visit(f"{day}-{name}", day, name, "LTE", 20, 0, 10**log_rate)


def test_raw_cell_means_overstate_effects_and_shrinkage_corrects_them() -> None:
    generator = np.random.default_rng(3)
    truth = generator.normal(0, 0.2, 40)
    visits = [
        visit(day, f"c{index}", 1 + effect + generator.normal(0, 0.3))
        for index, effect in enumerate(truth)
        for day in ("d1", "d2", "d3")
    ]
    result = shrinkage.held_out_forecast(visits)
    assert result["slope_raw"] < 0.8
    assert abs(result["slope_shrunken"] - 1) < abs(result["slope_raw"] - 1)
    assert result["mae_shrunken"] < result["mae_raw"]


def test_synthetic_recovery_controls_false_discoveries_and_calibrates_slopes() -> None:
    result = shrinkage.synthetic_recovery([4] * 60, within=0.02, between=0.09, replications=60)
    assert result["mean_false_discovery_proportion"] <= 0.15
    assert result["mean_power"] > 0.5
    assert result["mean_held_out_slope"] == pytest.approx(1.0, abs=0.15)


def test_split_sample_reliability_is_high_when_cells_differ_and_low_when_they_do_not() -> None:
    generator = np.random.default_rng(4)
    distinct = [
        cell(f"c{index}", list(effect + generator.normal(0, 0.05, 4)))
        for index, effect in enumerate(generator.normal(0, 1, 30))
    ]
    alike = [cell(f"c{index}", list(generator.normal(0, 1, 4))) for index in range(30)]
    assert shrinkage.split_sample_reliability(distinct, splits=50)["reliability"] > 0.9
    assert shrinkage.split_sample_reliability(alike, splits=50)["reliability"] < 0.4


def test_slow_threshold_means_a_factor_of_two() -> None:
    threshold = shrinkage.SLOW_THRESHOLD
    factor = 10**threshold
    assert factor == pytest.approx(0.5)
    assert threshold == pytest.approx(-math.log10(2))
