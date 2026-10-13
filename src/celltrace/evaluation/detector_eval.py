"""Detector evaluation on constructed cases, kept apart from agent evaluation.

Agents receive fixed windows, so detector quality never mixes with agent
quality. Here the core detector runs on each case snapshot. A constructed drop
counts as recovered when a flag confirms between the constructed onset and
the end of the case. Clean and idle cases measure flags under the constructed
definition, which is not real network fault precision.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from celltrace.analysis.detector import detect_candidates
from celltrace.config import DetectorSection
from celltrace.contracts.evidence import CandidateEvent
from celltrace.data.snapshot import open_snapshot

DROP_FAMILIES = {
    "rate_drop_with_radio_change",
    "cell_change_with_rate_drop",
    "stable_radio_rate_drop",
    "missing_metrics",
    "tool_failure",
    "misleading_string",
}
NO_FLAG_FAMILIES = {"clean_control", "application_idle"}


def evaluate_detector(
    manifest_path: Path, detector: DetectorSection, out_dir: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    cases_dir = manifest_path.parent.parent / "cases"
    rows = []
    for case in manifest["cases"]:
        case_dir = cases_dir / case["case_id"]
        candidate = CandidateEvent.model_validate_json((case_dir / "candidate.json").read_text())
        onset = candidate.event_window.start_ms
        connection, _ = open_snapshot(case_dir / "snapshot")
        try:
            result = detect_candidates(connection, detector)
        finally:
            connection.close()
        confirmations = sorted(
            int(flag["confirmation_ms"])
            for flag in result.flags
            if int(flag["confirmation_ms"]) > onset
        )
        before_onset = sum(1 for flag in result.flags if int(flag["confirmation_ms"]) <= onset)
        rows.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "expected": "flag" if case["family"] in DROP_FAMILIES else "no_flag",
                "flagged_after_onset": bool(confirmations),
                "first_delay_ms": confirmations[0] - onset if confirmations else None,
                "flags_after_onset": len(confirmations),
                "flags_before_onset": before_onset,
            }
        )
    by_family: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cases": 0, "flagged": 0, "delays": []}
    )
    for row in rows:
        entry = by_family[row["family"]]
        entry["cases"] += 1
        entry["flagged"] += int(row["flagged_after_onset"])
        if row["first_delay_ms"] is not None:
            entry["delays"].append(row["first_delay_ms"])
    summary_by_family = {
        family: {
            "cases": entry["cases"],
            "flagged_after_onset": entry["flagged"],
            "flag_rate": entry["flagged"] / entry["cases"],
            "delays_ms": sorted(entry["delays"]),
        }
        for family, entry in sorted(by_family.items())
    }
    drop_rows = [row for row in rows if row["expected"] == "flag"]
    no_flag_rows = [row for row in rows if row["expected"] == "no_flag"]
    summary = {
        "label": "Constructed pattern recovery by the core detector. Not real fault precision.",
        "detector": detector.model_dump(mode="json"),
        "cases": len(rows),
        "constructed_drop_recovery": None
        if not drop_rows
        else sum(row["flagged_after_onset"] for row in drop_rows) / len(drop_rows),
        "flags_on_clean_or_idle_cases": sum(row["flagged_after_onset"] for row in no_flag_rows),
        "clean_or_idle_cases": len(no_flag_rows),
        "flags_before_onset_total": sum(row["flags_before_onset"] for row in rows),
        "by_family": summary_by_family,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detector_evaluation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
