"""Recount the planted error studies per distinct finding, with exact binomial intervals.

Each base carries one finding per error type and wording, and every finding was reviewed
in two or more conversations. Conversations about the same finding are repeated measures,
so the unit of inference here is the distinct finding. Reads the frozen plans and run
stores in SQLite read only mode. It makes no provider call and writes one JSON file.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import beta

from celltrace.evaluation.planted_errors import score_item

STUDIES = {
    "study_2": Path("artifacts/private/final_evaluation_20261010/planted"),
    "study_3": Path("artifacts/private/final_evaluation_20261011/planted_v3"),
}


def clopper_pearson(successes: int, trials: int) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    if successes == trials:
        return [round(lower, 4), 1.0]
    upper = float(beta.ppf(0.975, successes + 1, trials - successes))
    return [round(lower, 4), round(upper, 4)]


def recorded_outcomes(store_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT run_id, payload_json FROM run_events WHERE event_type = 'planted_outcome'"
        ).fetchall()
    finally:
        connection.close()
    return {run_id: json.loads(payload) for run_id, payload in rows}


def finding_outcomes() -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    by_finding: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for study, directory in STUDIES.items():
        plan = json.loads((directory / "plan.json").read_text())
        outcomes = recorded_outcomes(directory / "live" / "runs.sqlite3")
        for item in plan["items"]:
            scored = score_item(
                item["label"],
                item["findings"],
                outcomes.get(item["context"]["run_id"], {"status": "not_run"}),
            )
            family = item["label"]["family"]
            wording = "new" if family.endswith("_new") else "original"
            error = family.removesuffix("_new").removesuffix("_original")
            by_finding[(error, wording, item["base_id"])].append(
                {
                    "study": study,
                    "located": scored["localized_challenge"],
                    "objection": scored["decision_objection"],
                }
            )
    return by_finding


def summarize() -> dict[str, Any]:
    by_finding = finding_outcomes()
    errors: dict[str, dict[str, Any]] = {}
    for error in sorted({key[0] for key in by_finding}):
        keys = [key for key in by_finding if key[0] == error]
        draws = [by_finding[key] for key in keys]
        if error == "correct_control":
            clean = sum(not any(draw["objection"] for draw in group) for group in draws)
            errors[error] = {
                "distinct_reports": len(keys),
                "conversations": sum(len(group) for group in draws),
                "reports_never_challenged": clean,
                "false_alarm_rate_interval_95": [
                    round(1 - bound, 4) for bound in reversed(clopper_pearson(clean, len(keys)))
                ],
            }
            continue
        every = sum(all(draw["located"] for draw in group) for group in draws)
        errors[error] = {
            "distinct_findings": len(keys),
            "wordings": sorted({key[1] for key in keys}),
            "conversations": sum(len(group) for group in draws),
            "conversations_located": sum(draw["located"] for group in draws for draw in group),
            "findings_located_in_every_draw": every,
            "findings_located_in_any_draw": sum(
                any(draw["located"] for draw in group) for group in draws
            ),
            "every_draw_rate_interval_95": clopper_pearson(every, len(keys)),
        }
    return {
        "method": "distinct finding as unit, exact Clopper-Pearson 95% intervals",
        "studies": {name: str(path) for name, path in STUDIES.items()},
        "unit_note": (
            "Six bases from two reused development sessions. Study 2 and study 3 reuse the "
            "original wordings on the same bases, so their conversations are repeated measures."
        ),
        "ninety_percent_bar": (
            "With every finding located, a lower bound above 0.90 needs at least 36 "
            "distinct findings."
        ),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(summarize(), indent=2, sort_keys=True) + "\n")
    print(arguments.out)


if __name__ == "__main__":
    main()
