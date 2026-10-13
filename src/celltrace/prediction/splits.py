"""Whole session and forward time splits for forecasting.

The whole session split assigns every file download session to exactly one
of train, dev and test, separately inside each mobility cohort, with a fixed
seed. It answers how a model transfers to sessions it has never seen.

The forward time split works inside each session on its own eligible issuance
times. The first 60 percent train, the next 20 percent are a forward dev
period and the final 20 percent test. A training row is kept only when its
target ends at least maximum_history_ms before the next period starts, which
is the earliest feature support of any row in that period. This purge is
derived from interval support, not from a guessed row count. Sessions are
never concatenated into one timeline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from celltrace.canonical import content_hash
from celltrace.prediction.features import APPLICATION_COHORT, FEATURE_VERSION, FeatureBuild

SPLIT_VERSION = "prediction_splits_v1"
SPLIT_FILE_NAME = "split_manifest.json"
SPLIT_SEED = 42
SESSION_FRACTIONS = {"train": 0.6, "dev": 0.2, "test": 0.2}
FORWARD_TRAIN_FRACTION = 0.6
FORWARD_TEST_FRACTION = 0.2
GROUPS = ("train", "dev", "test")


class SplitManifestError(ValueError):
    pass


def session_split(
    session_cohorts: Mapping[str, str], seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    """Deterministic train, dev and test session lists, stratified by mobility cohort.

    Each cohort with at least three sessions gives at least one session to dev
    and to test. Smaller cohorts go entirely to train and are reported as such.
    """
    generator = np.random.default_rng(seed)
    assignment: dict[str, list[str]] = {group: [] for group in GROUPS}
    for cohort in sorted(set(session_cohorts.values())):
        members = sorted(session for session, value in session_cohorts.items() if value == cohort)
        order = [members[index] for index in generator.permutation(len(members))]
        if len(order) < 3:
            assignment["train"].extend(order)
            continue
        test_count = max(1, round(SESSION_FRACTIONS["test"] * len(order)))
        dev_count = max(1, round(SESSION_FRACTIONS["dev"] * len(order)))
        assignment["test"].extend(order[:test_count])
        assignment["dev"].extend(order[test_count : test_count + dev_count])
        assignment["train"].extend(order[test_count + dev_count :])
    return {group: sorted(members) for group, members in assignment.items()}


def forward_time_boundaries(
    frame: pd.DataFrame,
    train_fraction: float = FORWARD_TRAIN_FRACTION,
    test_fraction: float = FORWARD_TEST_FRACTION,
) -> dict[str, dict[str, int]]:
    """Per session dev and test start times from that session's own eligible as_of times."""
    boundaries: dict[str, dict[str, int]] = {}
    for session_id, rows in frame.groupby("session_id", sort=True):
        times = np.sort(rows["as_of_ms"].to_numpy())
        count = len(times)
        dev_index = math.floor(train_fraction * count)
        test_index = math.floor((1.0 - test_fraction) * count)
        if count < 3 or dev_index >= test_index or test_index >= count:
            continue
        boundaries[str(session_id)] = {
            "eligible_rows": count,
            "dev_start_ms": int(times[dev_index]),
            "test_start_ms": int(times[test_index]),
        }
    return boundaries


def assign_forward_time(
    frame: pd.DataFrame,
    boundaries: Mapping[str, Mapping[str, int]],
    maximum_history_ms: int,
) -> pd.Series:
    """Label each row train, dev, test, purged or unassigned using the frozen boundaries."""
    labels = pd.Series("unassigned", index=frame.index, dtype=object)
    for session_id, bounds in boundaries.items():
        in_session = frame["session_id"] == session_id
        as_of = frame["as_of_ms"]
        target_end = frame["target_end_ms"]
        dev_start = int(bounds["dev_start_ms"])
        test_start = int(bounds["test_start_ms"])
        before_dev = in_session & (as_of < dev_start)
        in_dev = in_session & (as_of >= dev_start) & (as_of < test_start)
        in_test = in_session & (as_of >= test_start)
        labels[before_dev] = "purged"
        labels[before_dev & (target_end <= dev_start - maximum_history_ms)] = "train"
        labels[in_dev] = "purged"
        labels[in_dev & (target_end <= test_start - maximum_history_ms)] = "dev"
        labels[in_test] = "test"
    return labels


def check_forward_purge(
    frame: pd.DataFrame,
    labels: pd.Series,
    boundaries: Mapping[str, Mapping[str, int]],
    maximum_history_ms: int,
) -> None:
    """Raise if any training interval reaches the feature support of a later period."""
    for session_id, bounds in boundaries.items():
        in_session = frame["session_id"] == session_id
        train = frame[in_session & (labels == "train")]
        dev = frame[in_session & (labels == "dev")]
        test = frame[in_session & (labels == "test")]
        for earlier, later in ((train, dev), (train, test), (dev, test)):
            if earlier.empty or later.empty:
                continue
            if int(earlier["target_end_ms"].max()) > int(later["earliest_support_ms"].min()):
                raise SplitManifestError(f"{session_id}: purge violated between periods")
        if not test.empty and int(test["as_of_ms"].min()) < int(bounds["test_start_ms"]):
            raise SplitManifestError(f"{session_id}: test row before its test start")
        if not train.empty and int(train["target_end_ms"].max()) > (
            int(bounds["dev_start_ms"]) - maximum_history_ms
        ):
            raise SplitManifestError(f"{session_id}: training target inside the embargo")


def build_split_manifest(
    connection: duckdb.DuckDBPyConnection,
    snapshot_manifest: Mapping[str, Any],
    features: FeatureBuild,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT session_id, mobility_cohort FROM sessions WHERE application_cohort = ? "
        "ORDER BY session_id",
        [APPLICATION_COHORT],
    ).fetchall()
    cohorts = {str(row[0]): str(row[1]) for row in rows}
    sessions = session_split(cohorts, seed)
    boundaries = forward_time_boundaries(features.frame)
    labels = assign_forward_time(features.frame, boundaries, features.maximum_history_ms)
    check_forward_purge(features.frame, labels, boundaries, features.maximum_history_ms)
    label_counts = {str(key): int(value) for key, value in labels.value_counts().items()}
    payload: dict[str, Any] = {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "snapshot_logical_hash": snapshot_manifest["logical_hash"],
        "application_cohort": APPLICATION_COHORT,
        "feature_version": FEATURE_VERSION,
        "feature_specification_hash": features.specification_hash,
        "horizon_ms": features.horizon_ms,
        "maximum_history_ms": features.maximum_history_ms,
        "session_cohorts": cohorts,
        "session_split": {
            "strategy": "whole sessions, shuffled with the seed inside each mobility cohort, "
            "rounded 60 20 20 with at least one dev and one test session per cohort of three "
            "or more sessions",
            "fractions": SESSION_FRACTIONS,
            "sessions": sessions,
            "session_counts": {group: len(members) for group, members in sessions.items()},
            "cohort_counts": {
                group: {
                    cohort: sum(1 for member in members if cohorts[member] == cohort)
                    for cohort in sorted(set(cohorts.values()))
                }
                for group, members in sessions.items()
            },
        },
        "forward_time_split": {
            "strategy": "inside each session, first 60 percent of eligible as_of times train, "
            "next 20 percent dev, last 20 percent test. Rows of an earlier period are kept "
            "only when target_end_ms <= next period start minus maximum_history_ms",
            "train_fraction": FORWARD_TRAIN_FRACTION,
            "test_fraction": FORWARD_TEST_FRACTION,
            "embargo_ms": features.maximum_history_ms,
            "boundaries": boundaries,
            "row_counts": label_counts,
        },
        "feature_rows": features.exclusions,
    }
    payload["manifest_hash"] = content_hash(payload)
    return payload


def write_split_manifest(payload: Mapping[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SPLIT_FILE_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_split_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest and verify that its content still matches its recorded hash."""
    if not path.is_file():
        raise SplitManifestError(f"split manifest not found: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_hash", None)
    if recorded != content_hash(payload):
        raise SplitManifestError(f"{path} content does not match its manifest_hash")
    payload["manifest_hash"] = recorded
    return payload


def session_labels(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.Series:
    """train, dev, test or unassigned for each row by its session."""
    membership: dict[str, str] = {}
    for group, members in manifest["session_split"]["sessions"].items():
        for member in members:
            membership[member] = group
    return frame["session_id"].map(membership).fillna("unassigned")


def check_session_groups_disjoint(manifest: Mapping[str, Any]) -> None:
    groups = manifest["session_split"]["sessions"]
    seen: dict[str, str] = {}
    for group, members in groups.items():
        for member in members:
            if member in seen:
                raise SplitManifestError(f"{member} is in both {seen[member]} and {group}")
            seen[member] = group


def labelled_rows(
    snapshot_dir: Path, manifest: Mapping[str, Any]
) -> tuple[FeatureBuild, pd.DataFrame, dict[str, Any]]:
    """Rebuild feature rows from the snapshot and attach the frozen split labels.

    Refuses a snapshot or feature specification that differs from the one the
    manifest was built from.
    """
    from celltrace.data.snapshot import open_snapshot
    from celltrace.prediction.features import assert_past_only, build_feature_frame

    connection, snapshot_manifest = open_snapshot(snapshot_dir)
    try:
        features = build_feature_frame(
            connection,
            horizon_ms=int(manifest["horizon_ms"]),
            maximum_history_ms=int(manifest["maximum_history_ms"]),
        )
    finally:
        connection.close()
    if snapshot_manifest["snapshot_id"] != manifest["snapshot_id"]:
        raise SplitManifestError(
            f"snapshot {snapshot_manifest['snapshot_id']} differs from split manifest "
            f"{manifest['snapshot_id']}"
        )
    if snapshot_manifest["logical_hash"] != manifest["snapshot_logical_hash"]:
        raise SplitManifestError("snapshot logical hash differs from the split manifest")
    if features.specification_hash != manifest["feature_specification_hash"]:
        raise SplitManifestError("feature specification differs from the split manifest")
    if features.frame.empty:
        raise SplitManifestError("snapshot has no eligible forecast rows")
    check_session_groups_disjoint(manifest)
    frame = features.frame.copy()
    assert_past_only(frame, features.horizon_ms)
    boundaries = manifest["forward_time_split"]["boundaries"]
    frame["session_group"] = session_labels(frame, manifest)
    frame["forward_group"] = assign_forward_time(frame, boundaries, features.maximum_history_ms)
    check_forward_purge(frame, frame["forward_group"], boundaries, features.maximum_history_ms)
    return features, frame, snapshot_manifest
