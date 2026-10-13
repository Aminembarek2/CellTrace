"""Session bootstrap shared by the population and forecasting analyses.

The session is the unit of independence: seconds and rows inside a session overlap
and are correlated, so intervals resample whole sessions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 2_000
# Percentile intervals from five of this archive's sessions covered 87.1% of the time for a
# nominal 95%, and 95.5% from eight (docs/TECHNICAL_REPORT.md). Agent evaluation
# scoring keeps its own frozen rule.
MINIMUM_SESSIONS_FOR_INTERVAL = 8


def session_bootstrap(
    rows: Sequence[tuple[float, ...]],
    statistic: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> list[float] | None:
    """Percentile interval of a ratio of sums statistic, resampling whole sessions.

    Each row holds one session's additive counts. Resamples whose statistic is
    undefined (an empty denominator) are dropped, and no interval is returned
    when more than five percent are undefined or too few sessions contribute.
    """
    if len(rows) < MINIMUM_SESSIONS_FOR_INTERVAL:
        return None
    data = np.asarray(rows, dtype=float)
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(rows), size=(BOOTSTRAP_RESAMPLES, len(rows)))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = statistic(data[picks].sum(axis=1))
    finite = values[np.isfinite(values)]
    if finite.size < 0.95 * BOOTSTRAP_RESAMPLES:
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
