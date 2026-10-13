"""Refit selected outer folds and compare them with saved cross validation predictions.

A full rerun of nested cross validation takes about half an hour, so this check refits a
chosen subset of folds from both schemes with the same code and compares every saved
numeric column. It reads the snapshot and the saved outputs and writes one JSON record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltrace.data.snapshot import open_snapshot
from celltrace.prediction import cross_validation as validation
from celltrace.prediction.features import build_feature_frame
from celltrace.release import source_tree_hash


def compare_fold(
    frame: pd.DataFrame,
    groups: dict[str, Any],
    saved: pd.DataFrame,
    session: str,
    training_sessions: list[str],
) -> dict[str, Any]:
    refitted = validation.cross_validate(frame, groups, [(session, training_sessions)])
    original = saved.loc[saved["session_id"] == session].reset_index(drop=True)
    mismatched = []
    largest = 0.0
    for column in refitted.columns:
        if column not in original:
            mismatched.append(f"{column} missing from saved predictions")
            continue
        if pd.api.types.is_numeric_dtype(refitted[column]):
            difference = np.abs(
                refitted[column].to_numpy(dtype=float) - original[column].to_numpy(dtype=float)
            )
            largest = max(largest, float(np.nanmax(difference)) if len(difference) else 0.0)
            if not np.allclose(refitted[column], original[column], rtol=0, atol=1e-9):
                mismatched.append(column)
        elif not refitted[column].astype(str).equals(original[column].astype(str)):
            mismatched.append(column)
    return {
        "session": session,
        "rows": len(refitted),
        "columns": len(refitted.columns),
        "mismatched_columns": mismatched,
        "largest_absolute_difference": largest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True, help="saved prediction directory")
    parser.add_argument("--folds-per-scheme", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.private / validation.FOLDS_NAME).read_text())
    connection, _ = open_snapshot(arguments.snapshot)
    try:
        build = build_feature_frame(connection)
    finally:
        connection.close()
    groups = {name: build.feature_groups[name] for name in validation.FEATURE_SETS}
    saved = {
        validation.LEAVE_ONE_OUT: pd.read_csv(arguments.private / validation.PREDICTIONS_NAME),
        validation.FORWARD: pd.read_csv(arguments.private / validation.FORWARD_PREDICTIONS_NAME),
    }
    results = []
    for scheme, folds in manifest["folds"].items():
        # The first and the last fold of each scheme, which differ most in training data.
        chosen = [folds[0], folds[-1]][: arguments.folds_per_scheme]
        for fold in chosen:
            result = compare_fold(
                build.frame,
                groups,
                saved[scheme],
                fold["test_sessions"][0],
                fold["training_sessions"],
            )
            results.append({"scheme": scheme, **result})
    # Refitting on a later source and matching every saved column shows that the forecast
    # code path behaves identically, even when other files of the package changed.
    record = {
        "saved_source_sha256": manifest["source_sha256"],
        "source_sha256": source_tree_hash(),
        "same_source": manifest["source_sha256"] == source_tree_hash(),
        "tolerance": "absolute 1e-9 on every numeric column, exact text otherwise",
        "folds": results,
        "identical": all(not item["mismatched_columns"] for item in results),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"identical": record["identical"], "out": str(arguments.out)}))


if __name__ == "__main__":
    main()
