"""Cell study report: sections, verification table and figures render from a synthetic archive."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from celltrace.analysis import cell_study, cells, shrinkage


def synthetic_visits() -> list[cells.Visit]:
    generator = np.random.default_rng(8)
    effects = generator.normal(0, 0.3, 15)
    return [
        cells.Visit(
            f"{day}-s",
            day,
            f"c{index}",
            "LTE",
            20,
            0,
            10 ** (1 + effect + generator.normal(0, 0.2)),
        )
        for index, effect in enumerate(effects)
        for day in ("d1", "d2", "d3", "d4")
    ]


def summary_for(visits: list[cells.Visit]) -> dict[str, Any]:
    def by_day(visit: cells.Visit) -> str:
        return visit.day

    primary = cells.replication_test(visits, by_day, 19, 1)
    rows = primary.pop("rows")
    sensitivity = {
        key: dict(primary) for key in ("leave_one_session_out", "minimum_visit_samples_20")
    }
    return {
        "label": "synthetic",
        "method_version": cell_study.METHOD_VERSION,
        "snapshot_id": "snap_test",
        "cohort": "file_download",
        "mobilities": ["car"],
        "cell_identity": "reported serving cell with a known identity basis",
        "replication": {
            "visits": len(visits),
            "cells": 15,
            "sessions": 4,
            "days": 4,
            "primary": primary,
            "per_day": cells._per_fold(rows),
            "sensitivity": sensitivity,
            "decision": "replicates beyond technology",
        },
        "shrinkage": shrinkage.analyse(visits, seed=3),
    }


def test_report_has_every_section_and_the_verification_table(tmp_path: Path) -> None:
    visits = synthetic_visits()
    summary = summary_for(visits)
    text = cell_study.render(summary)
    for heading in (
        "# Cell study: synthetic",
        "## Does a cell keep its rate across days?",
        "## Which cells are reliably slow for their technology?",
        "## Verification",
        "## Limits",
    ):
        assert heading in text
    assert "Skill by held out day: d1" in text and "Sensitivity: leaving one session out" in text
    assert "| Held out slope, shrunken estimates |" in text
    cell_study.figure_funnel(summary["shrinkage"], tmp_path / "funnel.png")
    cell_study.figure_forecast(visits, summary["shrinkage"], tmp_path / "forecast.png")
    assert (tmp_path / "funnel.png").stat().st_size > 0
    assert (tmp_path / "forecast.png").stat().st_size > 0


def test_report_says_when_too_few_cells_repeat() -> None:
    summary = summary_for(synthetic_visits())
    summary["shrinkage"] = None
    assert "Too few cells were seen on two or more days" in cell_study.render(summary)
