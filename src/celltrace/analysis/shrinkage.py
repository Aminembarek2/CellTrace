"""Empirical Bayes cell effects: reliability, shrinkage and selection of slow cells.

A cell effect is how far a cell's visits sit from the mean of their technology, on log10
of the median rate. Cells seen a few times are noisy, so each estimate is pulled toward
zero by its reliability, and slow cells are selected by posterior tail probability under a
false discovery limit (Gu and Koenker 2023). Method: docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from celltrace.analysis.cells import Visit

SLOW_FACTOR = 2.0
SLOW_THRESHOLD = -math.log10(SLOW_FACTOR)
FALSE_DISCOVERY_LIMIT = 0.10
RANK_DRAWS = 10_000
RANK_INTERVAL = (0.05, 0.95)
GRID_POINTS = 400


@dataclass(frozen=True)
class CellData:
    cell: str
    technology: str
    days: int
    deviations: tuple[float, ...]

    @property
    def visits(self) -> int:
        return len(self.deviations)

    @property
    def mean(self) -> float:
        return float(np.mean(self.deviations))


def technology_means(visits: Sequence[Visit]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for visit in visits:
        groups[visit.technology].append(visit.log_rate)
    return {technology: float(np.mean(values)) for technology, values in groups.items()}


def cell_data(
    visits: Sequence[Visit], means: Mapping[str, float], minimum_days: int = 2
) -> list[CellData]:
    """Cells seen on enough days, with visit deviations from their technology mean."""
    groups: dict[str, list[Visit]] = defaultdict(list)
    for visit in visits:
        if visit.technology in means:
            groups[visit.cell].append(visit)
    cells = []
    for cell, members in sorted(groups.items()):
        days = {visit.day for visit in members}
        if len(days) < minimum_days:
            continue
        technologies = sorted({visit.technology for visit in members})
        main = max(technologies, key=lambda name: sum(v.technology == name for v in members))
        cells.append(
            CellData(
                cell=cell,
                technology=main,
                days=len(days),
                deviations=tuple(visit.log_rate - means[visit.technology] for visit in members),
            )
        )
    return cells


def pooled_within_variance(cells: Sequence[CellData]) -> float:
    """Within cell variance pooled over cells with at least two visits, by degrees of freedom."""
    squares = 0.0
    freedom = 0
    for cell in cells:
        if cell.visits < 2:
            continue
        values = np.asarray(cell.deviations)
        squares += float(np.sum((values - values.mean()) ** 2))
        freedom += cell.visits - 1
    if freedom == 0:
        raise ValueError("within cell variance needs a cell with at least two visits")
    return squares / freedom


def between_variance(means: np.ndarray, standard_errors: np.ndarray) -> float:
    """Moment estimator of the variance of true effects around zero, truncated at zero.

    With known zero mean, the weighted sum of squares has expectation k plus the between
    variance times the sum of weights (DerSimonian and Laird, adapted to a fixed mean).
    """
    weights = 1 / standard_errors**2
    statistic = float(np.sum(weights * means**2))
    return max(0.0, (statistic - len(means)) / float(np.sum(weights)))


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    result: np.ndarray = 0.5 * (1 + np.vectorize(math.erf)(values / math.sqrt(2)))
    return result


@dataclass(frozen=True)
class Posterior:
    mean: np.ndarray
    standard_deviation: np.ndarray
    tail_probability: np.ndarray
    reliability: np.ndarray


def normal_posterior(
    means: np.ndarray, standard_errors: np.ndarray, between: float, threshold: float
) -> Posterior:
    reliability = between / (between + standard_errors**2)
    mean = reliability * means
    deviation = np.sqrt(reliability * standard_errors**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = np.where(
            deviation > 0,
            _normal_cdf((threshold - mean) / np.where(deviation > 0, deviation, 1)),
            (mean < threshold).astype(float),
        )
    return Posterior(mean, deviation, tail, reliability)


def npmle_prior(
    means: np.ndarray, standard_errors: np.ndarray, iterations: int = 5_000
) -> tuple[np.ndarray, np.ndarray]:
    """Kiefer and Wolfowitz mixing distribution on a grid, fitted by EM."""
    spread = float(np.max(standard_errors)) * 3
    grid = np.linspace(float(means.min()) - spread, float(means.max()) + spread, GRID_POINTS)
    likelihood = np.exp(-0.5 * ((means[:, None] - grid[None, :]) / standard_errors[:, None]) ** 2)
    likelihood /= standard_errors[:, None]
    weights = np.full(GRID_POINTS, 1 / GRID_POINTS)
    for _ in range(iterations):
        joint = likelihood * weights
        responsibility = joint / joint.sum(axis=1, keepdims=True)
        updated = responsibility.mean(axis=0)
        if float(np.max(np.abs(updated - weights))) < 1e-10:
            weights = updated
            break
        weights = updated
    return grid, weights


def npmle_posterior(means: np.ndarray, standard_errors: np.ndarray, threshold: float) -> Posterior:
    grid, weights = npmle_prior(means, standard_errors)
    likelihood = np.exp(-0.5 * ((means[:, None] - grid[None, :]) / standard_errors[:, None]) ** 2)
    joint = likelihood * weights
    posterior = joint / joint.sum(axis=1, keepdims=True)
    mean = posterior @ grid
    deviation = np.sqrt(np.maximum(posterior @ grid**2 - mean**2, 0))
    tail = posterior[:, grid < threshold].sum(axis=1)
    reliability = np.full(len(means), np.nan)
    return Posterior(mean, deviation, tail, reliability)


def select_by_false_discovery(tail_probability: np.ndarray, limit: float) -> np.ndarray:
    """Largest set, by descending tail probability, whose expected false share is within limit."""
    order = np.argsort(-tail_probability, kind="stable")
    expected_false = np.cumsum(1 - tail_probability[order]) / np.arange(1, len(order) + 1)
    within = np.flatnonzero(expected_false <= limit)
    selected = np.zeros(len(tail_probability), dtype=bool)
    if len(within):
        selected[order[: within[-1] + 1]] = True
    return selected


def rank_intervals(
    posterior: Posterior, generator: np.random.Generator, draws: int = RANK_DRAWS
) -> np.ndarray:
    """Posterior ranks, 1 for the slowest cell, as lower and upper interval bounds."""
    samples = posterior.mean[None, :] + posterior.standard_deviation[None, :] * (
        generator.standard_normal((draws, len(posterior.mean)))
    )
    ranks = np.argsort(np.argsort(samples, axis=1), axis=1) + 1
    bounds: np.ndarray = np.quantile(ranks, RANK_INTERVAL, axis=0).T
    return bounds


@dataclass(frozen=True)
class Fit:
    within: float
    between: float
    cells: list[CellData]
    standard_errors: np.ndarray
    normal: Posterior
    npmle: Posterior | None


def fit(cells: Sequence[CellData], with_npmle: bool = True) -> Fit:
    within = pooled_within_variance(cells)
    means = np.array([cell.mean for cell in cells])
    errors = np.sqrt(within / np.array([cell.visits for cell in cells]))
    between = between_variance(means, errors)
    return Fit(
        within=within,
        between=between,
        cells=list(cells),
        standard_errors=errors,
        normal=normal_posterior(means, errors, between, SLOW_THRESHOLD),
        npmle=npmle_posterior(means, errors, SLOW_THRESHOLD) if with_npmle else None,
    )


def held_out_pairs(visits: Sequence[Visit]) -> list[tuple[str, float, float, float]]:
    """Leave one day out: cell, held out deviation, raw prediction, shrunken prediction.

    Technology means, variances and cell effects come from the other days only.
    """
    pairs: list[tuple[str, float, float, float]] = []
    for day in sorted({visit.day for visit in visits}):
        reference = [visit for visit in visits if visit.day != day]
        means = technology_means(reference)
        cells = cell_data(reference, means, minimum_days=1)
        if not any(cell.visits >= 2 for cell in cells):
            continue
        within = pooled_within_variance(cells)
        raw = np.array([cell.mean for cell in cells])
        errors = np.sqrt(within / np.array([cell.visits for cell in cells]))
        between = between_variance(raw, errors)
        reliability = between / (between + errors**2)
        lookup = {
            cell.cell: (float(value), float(weight))
            for cell, value, weight in zip(cells, raw, reliability, strict=True)
        }
        for visit in visits:
            if visit.day != day or visit.cell not in lookup or visit.technology not in means:
                continue
            raw_value, weight = lookup[visit.cell]
            actual = visit.log_rate - means[visit.technology]
            pairs.append((visit.cell, actual, raw_value, weight * raw_value))
    return pairs


def held_out_forecast(visits: Sequence[Visit]) -> dict[str, Any]:
    """Do raw or shrunken cell effects predict held out deviations with slope one?"""
    return _forecast_summary(held_out_pairs(visits))


def _slope(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    spread = float(np.var(predicted))
    if spread == 0:
        return None
    return float(np.cov(predicted, actual, bias=True)[0, 1] / spread)


def _forecast_summary(
    pairs: Sequence[tuple[str, float, float, float]], bootstrap: int = 2_000, seed: int = 42
) -> dict[str, Any]:
    if not pairs:
        return {"pairs": 0}
    cells = np.array([cell for cell, *_ in pairs])
    actual = np.array([item[1] for item in pairs])
    raw = np.array([item[2] for item in pairs])
    shrunk = np.array([item[3] for item in pairs])
    generator = np.random.default_rng([seed, 11])
    unique = np.unique(cells)
    members = {cell: np.flatnonzero(cells == cell) for cell in unique}
    draws: dict[str, list[float]] = {"raw": [], "shrunken": []}
    for _ in range(bootstrap):
        chosen = generator.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[cell] for cell in chosen])
        for name, predicted in (("raw", raw), ("shrunken", shrunk)):
            value = _slope(actual[index], predicted[index])
            if value is not None:
                draws[name].append(value)
    return {
        "pairs": len(pairs),
        "cells": len(unique),
        "slope_raw": _slope(actual, raw),
        "slope_shrunken": _slope(actual, shrunk),
        "slope_raw_interval_95": [float(v) for v in np.quantile(draws["raw"], [0.025, 0.975])],
        "slope_shrunken_interval_95": [
            float(v) for v in np.quantile(draws["shrunken"], [0.025, 0.975])
        ],
        "mae_raw": float(np.mean(np.abs(actual - raw))),
        "mae_shrunken": float(np.mean(np.abs(actual - shrunk))),
        "mae_technology_only": float(np.mean(np.abs(actual))),
    }


def split_sample_reliability(
    cells: Sequence[CellData], splits: int = 200, seed: int = 42
) -> dict[str, Any]:
    """Random half splits within cells, half mean correlation, Spearman and Brown corrected."""
    usable = [cell for cell in cells if cell.visits >= 2]
    if len(usable) < 3:
        return {"cells": len(usable), "reliability": None}
    generator = np.random.default_rng([seed, 13])
    values = []
    for _ in range(splits):
        first, second = [], []
        for cell in usable:
            order = generator.permutation(cell.visits)
            half = cell.visits // 2
            first.append(float(np.mean([cell.deviations[i] for i in order[:half]])))
            second.append(float(np.mean([cell.deviations[i] for i in order[half:]])))
        correlation = float(np.corrcoef(first, second)[0, 1])
        values.append(2 * correlation / (1 + correlation))
    return {
        "cells": len(usable),
        "splits": splits,
        "reliability": float(np.mean(values)),
        "interval_90": [float(v) for v in np.quantile(values, [0.05, 0.95])],
    }


def synthetic_recovery(
    visit_counts: Sequence[int],
    within: float,
    between: float,
    replications: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate archives from the fitted model, rerun the selection, measure what it gets right."""
    generator = np.random.default_rng([seed, 17])
    counts = np.asarray(visit_counts)
    proportions, powers, slopes = [], [], []
    for _ in range(replications):
        truth = generator.normal(0, math.sqrt(between), len(counts))
        simulated = [
            CellData(
                f"c{index}", "t", 2, tuple(truth[index] + generator.normal(0, math.sqrt(within), n))
            )
            for index, n in enumerate(counts)
        ]
        estimate = fit(simulated, with_npmle=False)
        selected = select_by_false_discovery(
            estimate.normal.tail_probability, FALSE_DISCOVERY_LIMIT
        )
        slow = truth < SLOW_THRESHOLD
        if selected.any():
            proportions.append(float(np.mean(~slow[selected])))
        if slow.any():
            powers.append(float(np.mean(selected[slow])))
        held_out = truth + generator.normal(0, math.sqrt(within), len(counts))
        value = _slope(held_out, estimate.normal.mean)
        if value is not None:
            slopes.append(value)
    return {
        "replications": replications,
        "mean_false_discovery_proportion": float(np.mean(proportions)) if proportions else None,
        "replications_with_selection": len(proportions),
        "mean_power": float(np.mean(powers)) if powers else None,
        "mean_held_out_slope": float(np.mean(slopes)) if slopes else None,
    }


def analyse(visits: Sequence[Visit], seed: int = 42) -> dict[str, Any]:
    """Step 1 for one archive: fitted model, selection, ranks and the three verifications."""
    means = technology_means(visits)
    cells = cell_data(visits, means)
    estimate = fit(cells)
    if estimate.npmle is None:
        raise RuntimeError("the nonparametric prior was not fitted")
    selected = select_by_false_discovery(estimate.normal.tail_probability, FALSE_DISCOVERY_LIMIT)
    npmle_selected = select_by_false_discovery(
        estimate.npmle.tail_probability, FALSE_DISCOVERY_LIMIT
    )
    intervals = rank_intervals(estimate.normal, np.random.default_rng([seed, 19]))
    table = [
        {
            "cell": cell.cell,
            "technology": cell.technology,
            "days": cell.days,
            "visits": cell.visits,
            "raw_effect": cell.mean,
            "standard_error": float(error),
            "reliability": float(reliability),
            "shrunken_effect": float(mean),
            "tail_probability": float(tail),
            "selected": bool(chosen),
            "npmle_tail_probability": float(npmle_tail),
            "npmle_selected": bool(npmle_chosen),
            "rank_interval_90": [float(bound) for bound in interval],
        }
        for cell, error, reliability, mean, tail, chosen, npmle_tail, npmle_chosen, interval in zip(
            cells,
            estimate.standard_errors,
            estimate.normal.reliability,
            estimate.normal.mean,
            estimate.normal.tail_probability,
            selected,
            estimate.npmle.tail_probability,
            npmle_selected,
            intervals,
            strict=True,
        )
    ]
    table.sort(key=lambda row: (row["shrunken_effect"], row["cell"]))
    return {
        "slow_factor": SLOW_FACTOR,
        "false_discovery_limit": FALSE_DISCOVERY_LIMIT,
        "cells": len(cells),
        "within_variance": estimate.within,
        "between_variance": estimate.between,
        "mean_reliability": float(np.mean(estimate.normal.reliability)),
        "share_reliability_below_0_7": float(np.mean(estimate.normal.reliability < 0.7)),
        "selected": int(selected.sum()),
        "npmle_selected": int(npmle_selected.sum()),
        "expected_false_among_selected": float(
            np.sum(1 - estimate.normal.tail_probability[selected])
        ),
        "table": table,
        "held_out_forecast": held_out_forecast(visits),
        "split_sample": split_sample_reliability(cells, seed=seed),
        "synthetic_recovery": synthetic_recovery(
            [cell.visits for cell in cells], estimate.within, estimate.between, seed=seed
        ),
    }
