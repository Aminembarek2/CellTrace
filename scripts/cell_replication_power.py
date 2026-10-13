"""Power of the Q7 cell test on a new archive, before its rates are read.

Takes the visit structure of a snapshot (which sessions met which cells on which days, in
which technology) without reading any rate, simulates rates from the cell model fitted on
the Raca archive, and reruns the unchanged leave one day out test on each simulated archive.
Method: docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from celltrace.analysis import cells

STRUCTURE_SQL = """
WITH session_days AS (
    SELECT session_id, CAST(CAST(min(local_timestamp) AS DATE) AS VARCHAR) AS day
    FROM samples GROUP BY session_id
),
shared_keys AS (
    SELECT split_part(serving_cell_id, ':', 2) || ':' || split_part(serving_cell_id, ':', 3) AS key
    FROM samples WHERE serving_cell_id IS NOT NULL GROUP BY 1
    HAVING count(DISTINCT split_part(serving_cell_id, ':', 1)) > 1
),
eligible AS (
    SELECT samples.session_id, day, serving_cell_id, network_mode
    FROM samples JOIN sessions USING (session_id) JOIN session_days USING (session_id)
    WHERE application_cohort = 'file_download' AND list_contains(?, mobility_cohort)
      AND download_state = 'active' AND downlink_mbps IS NOT NULL
      AND serving_cell_id IS NOT NULL
      AND split_part(serving_cell_id, ':', 2) || ':' || split_part(serving_cell_id, ':', 3)
          NOT IN (SELECT key FROM shared_keys)
)
SELECT session_id, day, serving_cell_id, network_mode, count(*) AS samples
FROM eligible GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
"""

TECHNOLOGY_LEVELS = {"5G": 1.3, "LTE": 1.0, "3G": 0.5, "2G": 0.0}


def visit_structure(snapshot: Path, mobilities: list[str]) -> list[tuple[str, str, str, str]]:
    """(session, day, cell, technology) for visits with enough samples, no rate read."""
    connection = duckdb.connect(str(snapshot / "measurements.duckdb"), read_only=True)
    try:
        rows = connection.execute(STRUCTURE_SQL, [mobilities]).fetchall()
    finally:
        connection.close()
    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for session, day, cell, mode, samples in rows:
        counts.setdefault((session, day, cell), {})[mode] = samples
    structure = []
    for (session, day, cell), modes in sorted(counts.items()):
        if sum(modes.values()) < cells.MINIMUM_VISIT_SAMPLES:
            continue
        main = sorted(mode for mode, count in modes.items() if count == max(modes.values()))[0]
        if main in cells.TECHNOLOGY_GROUPS:
            structure.append((session, day, cell, cells.TECHNOLOGY_GROUPS[main]))
    return structure


def simulate_once(task: tuple[list[tuple[str, str, str, str]], float, float, int, int]) -> bool:
    structure, within, between, permutations, seed = task
    generator = np.random.default_rng([seed, 23])
    effects: dict[str, float] = {}
    visits = []
    for session, day, cell, technology in structure:
        if cell not in effects:
            effects[cell] = float(generator.normal(0, math.sqrt(between)))
        level = TECHNOLOGY_LEVELS[technology] + effects[cell]
        level += float(generator.normal(0, math.sqrt(within)))
        visits.append(cells.Visit(session, day, cell, technology, 20, 0, 10**level))
    result = cells.replication_test(visits, lambda visit: visit.day, permutations, seed)
    skill = result["skill_against_technology"]
    p_value = result["p_value"]
    return skill is not None and skill > 0 and p_value is not None and p_value < 0.05


def wilson(successes: int, trials: int) -> list[float]:
    z = 1.96
    share = successes / trials
    centre = (share + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials**2))
    half /= 1 + z * z / trials
    return [centre - half, centre + half]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fitted", type=Path, required=True, help="Raca cell study summary")
    parser.add_argument("--mobility", action="append", required=True)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    fitted = json.loads(arguments.fitted.read_text())["shrinkage"]
    structure = visit_structure(arguments.snapshot, arguments.mobility)
    tasks = [
        (
            structure,
            fitted["within_variance"],
            fitted["between_variance"],
            arguments.permutations,
            seed,
        )
        for seed in range(arguments.simulations)
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        outcomes = list(pool.map(simulate_once, tasks, chunksize=4))
    successes = sum(outcomes)
    record: dict[str, Any] = {
        "snapshot": str(arguments.snapshot),
        "mobilities": arguments.mobility,
        "visits": len(structure),
        "cells_on_two_or_more_days": sum(
            1
            for cell in {item[2] for item in structure}
            if len({item[1] for item in structure if item[2] == cell}) >= 2
        ),
        "within_variance": fitted["within_variance"],
        "between_variance": fitted["between_variance"],
        "simulations": arguments.simulations,
        "permutations": arguments.permutations,
        "power": successes / arguments.simulations,
        "power_interval_95": wilson(successes, arguments.simulations),
        "underpowered": successes / arguments.simulations < 0.5,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
