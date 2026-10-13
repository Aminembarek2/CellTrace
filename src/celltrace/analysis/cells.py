"""Does a serving cell keep its delivered rate across days? Leave one day out prediction.

A visit is one driving session on one serving cell with enough active download samples.
Each held out day's visits are predicted from other days only, by the same cell, by the
same technology or by all visits. Permuting cell labels within day and technology gives
the null. This measures a cell's area as driven on these routes, not why it is slow.
Method: docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import duckdb
import numpy as np
import pandas as pd

MINIMUM_VISIT_SAMPLES = 10
SENSITIVITY_VISIT_SAMPLES = 20
RATE_FLOOR_MBPS = 0.1
LOW_RATE_MBPS = 1.5
PERMUTATIONS = 999
RANKED_CELLS_SHOWN = 10
DRIVING = ("driving",)
TECHNOLOGY_GROUPS = {
    "5G": "5G",
    "LTE": "LTE",
    "HSPA+": "3G",
    "HSDPA": "3G",
    "HSUPA": "3G",
    "UMTS": "3G",
}


@dataclasses.dataclass(frozen=True)
class Visit:
    session_id: str
    day: str
    cell: str
    technology: str
    samples: int
    low_samples: int
    median_mbps: float
    # Reported RSRP is undefined on 3G rows, so 3G visits carry None.
    median_rsrp_dbm: float | None = None

    @property
    def log_rate(self) -> float:
        return float(np.log10(max(self.median_mbps, RATE_FLOOR_MBPS)))

    @property
    def low_share(self) -> float:
        return self.low_samples / self.samples


SAMPLES_SQL = """
WITH session_days AS (
    SELECT session_id, CAST(CAST(min(local_timestamp) AS DATE) AS VARCHAR) AS session_day
    FROM samples GROUP BY session_id
)
SELECT samples.session_id, session_day, serving_cell_id, network_mode, downlink_mbps,
       reported_rsrp_dbm
FROM samples
JOIN sessions USING (session_id)
JOIN session_days USING (session_id)
WHERE application_cohort = ? AND list_contains(?, mobility_cohort)
  AND download_state = 'active' AND downlink_mbps IS NOT NULL
  AND serving_cell_id IS NOT NULL AND (? OR cell_identity_basis <> 'unknown')
  AND serving_cell_id NOT IN (
      SELECT serving_cell_id FROM samples
      WHERE ? AND serving_cell_id IS NOT NULL
        AND split_part(serving_cell_id, ':', 2) || ':' || split_part(serving_cell_id, ':', 3) IN (
            SELECT split_part(serving_cell_id, ':', 2) || ':' || split_part(serving_cell_id, ':', 3)
            FROM samples WHERE serving_cell_id IS NOT NULL GROUP BY 1
            HAVING count(DISTINCT split_part(serving_cell_id, ':', 1)) > 1
        )
  )
"""


def load_visits(
    connection: duckdb.DuckDBPyConnection,
    cohort: str,
    minimum_samples: int,
    mobilities: Sequence[str] = DRIVING,
    accept_proxy_identity: bool = False,
) -> list[Visit]:
    """Visits per session and cell. A proxy identity, technology family, site and local number,
    is accepted only when asked, and a site and number seen under two families is dropped."""
    frame = connection.execute(
        SAMPLES_SQL, [cohort, list(mobilities), accept_proxy_identity, accept_proxy_identity]
    ).df()
    visits = []
    for (session_id, day, cell), group in frame.groupby(
        ["session_id", "session_day", "serving_cell_id"], sort=True
    ):
        if len(group) < minimum_samples:
            continue
        counts = group["network_mode"].value_counts()
        # Most common mode, ties broken by name so the grouping is deterministic.
        main_mode = sorted(counts[counts == counts.max()].index)[0]
        if main_mode not in TECHNOLOGY_GROUPS:
            continue
        rates = group["downlink_mbps"]
        technology = TECHNOLOGY_GROUPS[main_mode]
        rsrp = group["reported_rsrp_dbm"].dropna()
        visits.append(
            Visit(
                session_id=str(session_id),
                day=str(day),
                cell=str(cell),
                technology=technology,
                samples=len(group),
                low_samples=int((rates < LOW_RATE_MBPS).sum()),
                median_mbps=float(rates.median()),
                median_rsrp_dbm=float(rsrp.median()) if technology != "3G" and len(rsrp) else None,
            )
        )
    return visits


def _pooled_low_share(visits: Sequence[Visit]) -> float:
    return sum(visit.low_samples for visit in visits) / sum(visit.samples for visit in visits)


def predictions(visits: Sequence[Visit], fold_of: Callable[[Visit], str]) -> list[dict[str, Any]]:
    """Every scorable held out visit with predictions from the other folds only."""
    rows = []
    for fold in sorted({fold_of(visit) for visit in visits}):
        reference = [visit for visit in visits if fold_of(visit) != fold]
        if not reference:
            continue
        by_cell: dict[str, list[Visit]] = defaultdict(list)
        by_technology: dict[str, list[Visit]] = defaultdict(list)
        for visit in reference:
            by_cell[visit.cell].append(visit)
            by_technology[visit.technology].append(visit)
        overall = float(np.median([visit.log_rate for visit in reference]))
        for visit in visits:
            if fold_of(visit) != fold or visit.cell not in by_cell:
                continue
            same_technology = by_technology.get(visit.technology, [])
            rows.append(
                {
                    "fold": fold,
                    "cell": visit.cell,
                    "technology": visit.technology,
                    "actual": visit.log_rate,
                    "cell_prediction": float(
                        np.median([item.log_rate for item in by_cell[visit.cell]])
                    ),
                    # A technology absent from the other folds falls back to all visits.
                    "technology_prediction": float(
                        np.median([item.log_rate for item in same_technology])
                    )
                    if same_technology
                    else overall,
                    "technology_fallback": not same_technology,
                    "global_prediction": overall,
                    "low_share": visit.low_share,
                    "cell_low_share": _pooled_low_share(by_cell[visit.cell]),
                    "technology_low_share": _pooled_low_share(same_technology or reference),
                }
            )
    return rows


def _mae(rows: Sequence[dict[str, Any]], prediction: str, actual: str = "actual") -> float:
    return float(np.mean([abs(row[actual] - row[prediction]) for row in rows]))


def skill(rows: Sequence[dict[str, Any]], baseline: str = "technology_prediction") -> float | None:
    if not rows:
        return None
    reference = _mae(rows, baseline)
    return None if reference == 0 else 1 - _mae(rows, "cell_prediction") / reference


def permuted_cells(visits: Sequence[Visit], generator: np.random.Generator) -> list[Visit]:
    """Cell labels shuffled among visits of the same day and technology."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, visit in enumerate(visits):
        groups[(visit.day, visit.technology)].append(index)
    labels = [visit.cell for visit in visits]
    shuffled = list(labels)
    for key in sorted(groups):
        members = groups[key]
        for target, source in zip(members, generator.permutation(len(members)), strict=True):
            shuffled[target] = labels[members[source]]
    return [
        dataclasses.replace(visit, cell=cell) for visit, cell in zip(visits, shuffled, strict=True)
    ]


def replication_test(
    visits: Sequence[Visit], fold_of: Callable[[Visit], str], permutations: int, seed: int
) -> dict[str, Any]:
    rows = predictions(visits, fold_of)
    observed = skill(rows)
    generator = np.random.default_rng([seed, 7])
    null = [
        skill(predictions(permuted_cells(visits, generator), fold_of)) for _ in range(permutations)
    ]
    finite = [value for value in null if value is not None]
    p_value = (
        None
        if observed is None or len(finite) != len(null)
        else (1 + sum(value >= observed for value in finite)) / (len(finite) + 1)
    )
    return {
        "scored_visits": len(rows),
        "scored_cells": len({row["cell"] for row in rows}),
        "folds": len({row["fold"] for row in rows}),
        "technology_fallbacks": sum(row["technology_fallback"] for row in rows),
        "mae_cell": _mae(rows, "cell_prediction") if rows else None,
        "mae_technology": _mae(rows, "technology_prediction") if rows else None,
        "mae_global": _mae(rows, "global_prediction") if rows else None,
        "skill_against_technology": observed,
        "skill_against_global": skill(rows, "global_prediction"),
        "permutations": permutations,
        "null_draws": null,
        "null_mean": float(np.mean(finite)) if finite else None,
        "null_interval_95": [float(value) for value in np.quantile(finite, [0.025, 0.975])]
        if finite
        else None,
        "p_value": p_value,
        "rows": rows,
    }


def _per_fold(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    folds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        folds[row["fold"]].append(row)
    return [
        {
            "fold": fold,
            "visits": len(members),
            "mae_cell": _mae(members, "cell_prediction"),
            "mae_technology": _mae(members, "technology_prediction"),
            "skill": skill(members),
        }
        for fold, members in sorted(folds.items())
    ]


def _rank_correlation(rows: Sequence[dict[str, Any]], technology: str) -> dict[str, Any]:
    members = [row for row in rows if row["technology"] == technology]
    if len(members) < 3:
        return {"visits": len(members), "spearman": None}
    frame = pd.DataFrame(members)
    value = frame["cell_prediction"].corr(frame["actual"], method="spearman")
    return {"visits": len(members), "spearman": None if pd.isna(value) else float(value)}


def ranked_cells(visits: Sequence[Visit]) -> dict[str, Any]:
    """Cells seen on at least two days, slowest first. Descriptive, uses every day."""
    by_cell: dict[str, list[Visit]] = defaultdict(list)
    for visit in visits:
        by_cell[visit.cell].append(visit)
    cells = []
    for cell, members in by_cell.items():
        days = {visit.day for visit in members}
        if len(days) < 2:
            continue
        technologies = pd.Series([visit.technology for visit in members]).value_counts()
        rsrp = [visit.median_rsrp_dbm for visit in members if visit.median_rsrp_dbm is not None]
        cells.append(
            {
                "cell": cell,
                "technology": sorted(technologies[technologies == technologies.max()].index)[0],
                "days": len(days),
                "visits": len(members),
                "samples": sum(visit.samples for visit in members),
                "median_visit_mbps": float(np.median([visit.median_mbps for visit in members])),
                "visit_median_range_mbps": [
                    min(visit.median_mbps for visit in members),
                    max(visit.median_mbps for visit in members),
                ],
                "median_rsrp_dbm": float(np.median(rsrp)) if rsrp else None,
                "low_share": _pooled_low_share(members),
                "low_samples": sum(visit.low_samples for visit in members),
            }
        )
    cells.sort(key=lambda item: (item["median_visit_mbps"], item["cell"]))
    all_low = sum(visit.low_samples for visit in visits)
    shown = cells[:RANKED_CELLS_SHOWN]
    return {
        "cells_on_two_or_more_days": len(cells),
        "slowest": shown,
        "low_samples_in_slowest": sum(item["low_samples"] for item in shown),
        "low_samples_all_visits": all_low,
        "samples_in_slowest": sum(item["samples"] for item in shown),
        "samples_all_visits": sum(visit.samples for visit in visits),
    }


def cell_replication(
    connection: duckdb.DuckDBPyConnection,
    cohort: str,
    seed: int,
    permutations: int = PERMUTATIONS,
    mobilities: Sequence[str] = DRIVING,
    accept_proxy_identity: bool = False,
) -> dict[str, Any]:
    visits = load_visits(
        connection, cohort, MINIMUM_VISIT_SAMPLES, mobilities, accept_proxy_identity
    )

    def by_day(visit: Visit) -> str:
        return visit.day

    def by_session(visit: Visit) -> str:
        return visit.session_id

    primary = replication_test(visits, by_day, permutations, seed)
    rows = primary.pop("rows")
    larger = load_visits(
        connection, cohort, SENSITIVITY_VISIT_SAMPLES, mobilities, accept_proxy_identity
    )
    sensitivity_larger = replication_test(larger, by_day, permutations, seed)
    sensitivity_larger.pop("rows")
    sensitivity_session = replication_test(visits, by_session, permutations, seed)
    sensitivity_session.pop("rows")
    low_share = (
        {
            "mae_cell": _mae(rows, "cell_low_share", "low_share"),
            "mae_technology": _mae(rows, "technology_low_share", "low_share"),
        }
        if rows
        else {"mae_cell": None, "mae_technology": None}
    )
    return {
        "definition": (
            "leave one day out prediction of a driving visit's median delivered rate, "
            "log10 with a 0.1 Mbps floor, by the same cell, the same technology or all visits"
        ),
        "minimum_visit_samples": MINIMUM_VISIT_SAMPLES,
        "visits": len(visits),
        "cells": len({visit.cell for visit in visits}),
        "sessions": len({visit.session_id for visit in visits}),
        "days": len({visit.day for visit in visits}),
        "primary": primary,
        "per_day": _per_fold(rows),
        "rank_correlation": {
            technology: _rank_correlation(rows, technology) for technology in ("5G", "LTE")
        },
        "low_share_error": low_share,
        "sensitivity": {
            "minimum_visit_samples_20": sensitivity_larger,
            "leave_one_session_out": sensitivity_session,
        },
        "ranking": ranked_cells(visits),
        "scored_rows": rows,
        "decision": (
            "replicates beyond technology"
            if primary["skill_against_technology"] is not None
            and primary["skill_against_technology"] > 0
            and primary["p_value"] is not None
            and primary["p_value"] < 0.05
            else "does not replicate beyond technology"
        ),
    }
