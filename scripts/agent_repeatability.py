"""Repeated trials of the agent on the same frozen cases: pass^k, consistency and cost.

Reads the scored runs of k isolated trials, grades each trial in code against the private
construction labels and reports pass^1, pass^k (all k trials succeed), pass@k, consistency
of the reported pattern sets, results by case family and cost. Intervals over cases are exact
(Clopper and Pearson). Method: docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ALPHA = 0.05


def binomial_tail_at_least(successes: int, trials: int, share: float) -> float:
    return sum(
        math.comb(trials, count) * share**count * (1 - share) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def _solve_increasing(function: Any, target: float) -> float:
    """Share at which an increasing function of the share reaches the target, by bisection."""
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if function(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def clopper_pearson(successes: int, trials: int, alpha: float = ALPHA) -> list[float]:
    """Exact interval: tails of the binomial equal to alpha / 2 at each bound."""
    lower = (
        0.0
        if successes == 0
        else _solve_increasing(
            lambda share: binomial_tail_at_least(successes, trials, share), alpha / 2
        )
    )
    upper = (
        1.0
        if successes == trials
        else _solve_increasing(
            lambda share: binomial_tail_at_least(successes + 1, trials, share), 1 - alpha / 2
        )
    )
    return [lower, upper]


def patterns(text: str) -> frozenset[str]:
    return frozenset(json.loads(text)) if text else frozenset()


def trial_success(run: dict[str, str], expected: dict[str, list[str]]) -> dict[str, bool]:
    reported = patterns(run["reported_patterns"])
    checks = {
        "completed": run["workflow_outcome"] == "completed",
        "claims_verified": run["final_verification_passed"] == "True"
        and run["fabricated_evidence_claims"] in ("0", ""),
        "required_patterns": set(expected["required"]) <= reported,
        "no_forbidden_pattern": not (set(expected["forbidden"]) & reported),
    }
    checks["success"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True, help="runs.csv")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text())
    expected_by_family = manifest["expected_patterns"]
    families = {case["case_id"]: case["family"] for case in manifest["cases"]}
    sessions = {
        case["case_id"]: case["base_window"]["source_session_id"] for case in manifest["cases"]
    }
    trials: list[dict[str, dict[str, str]]] = []
    for path in arguments.runs:
        with path.open(newline="", encoding="utf-8") as handle:
            trials.append({row["case_id"]: row for row in csv.DictReader(handle)})
    case_ids = sorted(families)
    per_case = []
    usd = [sum(float(row["estimated_usd"] or 0) for row in trial.values()) for trial in trials]
    for case_id in case_ids:
        expected = expected_by_family[families[case_id]]
        outcomes = [trial_success(trial[case_id], expected) for trial in trials if case_id in trial]
        reported = [
            patterns(trial[case_id]["reported_patterns"]) for trial in trials if case_id in trial
        ]
        successes = sum(outcome["success"] for outcome in outcomes)
        per_case.append(
            {
                "case_id": case_id,
                "family": families[case_id],
                "source_session_id": sessions[case_id],
                "trials": len(outcomes),
                "successes": successes,
                "pass_all": successes == len(outcomes) == len(trials),
                "pass_any": successes > 0,
                "consistent_patterns": len(set(reported)) == 1 and len(reported) == len(trials),
                "failed_checks": sorted(
                    {name for outcome in outcomes for name, value in outcome.items() if not value}
                    - {"success"}
                ),
            }
        )
    cases = len(per_case)
    all_pass = sum(item["pass_all"] for item in per_case)
    any_pass = sum(item["pass_any"] for item in per_case)
    consistent = sum(item["consistent_patterns"] for item in per_case)
    trial_successes = sum(item["successes"] for item in per_case)
    trial_count = sum(item["trials"] for item in per_case)
    by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "pass_all": 0, "trials": 0, "successes": 0}
    )
    for item in per_case:
        entry = by_family[item["family"]]
        entry["cases"] += 1
        entry["pass_all"] += item["pass_all"]
        entry["trials"] += item["trials"]
        entry["successes"] += item["successes"]
    record: dict[str, Any] = {
        "trials": len(trials),
        "cases": cases,
        "runs": trial_count,
        "pass_1": trial_successes / trial_count,
        "pass_1_trial_level_interval_95": clopper_pearson(trial_successes, trial_count),
        f"pass_hat_{len(trials)}": all_pass / cases,
        f"pass_hat_{len(trials)}_interval_95": clopper_pearson(all_pass, cases),
        f"pass_at_{len(trials)}": any_pass / cases,
        "consistent_pattern_sets": consistent / cases,
        "consistent_pattern_sets_interval_95": clopper_pearson(consistent, cases),
        "usd_per_trial": usd,
        "usd_total": sum(usd),
        "usd_per_successful_run": sum(usd) / trial_successes if trial_successes else None,
        "by_family": dict(sorted(by_family.items())),
        "failed_check_counts": {
            name: sum(name in item["failed_checks"] for item in per_case)
            for name in (
                "completed",
                "claims_verified",
                "required_patterns",
                "no_forbidden_pattern",
            )
        },
        "per_case": per_case,
        "interval_note": "Clopper and Pearson over cases. pass^1 over runs treats repeated runs "
        "of a case as independent and is optimistic. Cases come from three source sessions.",
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
