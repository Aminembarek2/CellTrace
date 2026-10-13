"""The predict command group: splits, train, evaluate and calibrate-residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXIT_OK = 0

DEFAULT_MODEL_DIR = Path("artifacts/private/models")
DEFAULT_PUBLIC_DIR = Path("artifacts/public/prediction")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _require_file(path: Path, label: str) -> None:
    from celltrace.cli import CommandError

    if not path.is_file():
        raise CommandError(f"{label} not found: {path}")


def _require_snapshot(path: Path) -> None:
    from celltrace.cli import EXIT_BLOCKED, CommandError

    if not (path / "snapshot_manifest.json").is_file():
        raise CommandError(f"no snapshot at {path}", EXIT_BLOCKED)


def command_splits(arguments: argparse.Namespace) -> int:
    from celltrace.data.snapshot import open_snapshot
    from celltrace.prediction.features import build_feature_frame
    from celltrace.prediction.splits import build_split_manifest, write_split_manifest

    snapshot = Path(arguments.snapshot)
    _require_snapshot(snapshot)
    connection, snapshot_manifest = open_snapshot(snapshot)
    try:
        features = build_feature_frame(connection)
        manifest = build_split_manifest(connection, snapshot_manifest, features)
    finally:
        connection.close()
    path = write_split_manifest(manifest, Path(arguments.out))
    _print_json(
        {
            "split_manifest": str(path),
            "manifest_hash": manifest["manifest_hash"],
            "session_counts": manifest["session_split"]["session_counts"],
            "cohort_counts": manifest["session_split"]["cohort_counts"],
            "forward_time_row_counts": manifest["forward_time_split"]["row_counts"],
            "feature_rows": manifest["feature_rows"],
        }
    )
    return EXIT_OK


def command_train(arguments: argparse.Namespace) -> int:
    from celltrace.prediction.models import train_models

    snapshot = Path(arguments.snapshot)
    splits = Path(arguments.splits)
    _require_snapshot(snapshot)
    _require_file(splits, "split manifest")
    summary = train_models(snapshot, splits, Path(arguments.out))
    _print_json(
        {
            "main_artifact": str(Path(arguments.out) / summary["main_artifact"]),
            "failed_fits": len(summary["failed_fits"]),
            "artifacts": {
                name: {
                    "status": entry["status"],
                    "chosen_configuration": entry.get("chosen_configuration"),
                    "train_rows": entry["train_rows"],
                    "dev_rows": entry["dev_rows"],
                    "best_dev_mae_mbps": min(
                        (score["dev_mae_mbps"] for score in entry["dev_scores"]), default=None
                    ),
                }
                for name, entry in summary["artifacts"].items()
            },
        }
    )
    return EXIT_OK


def command_evaluate(arguments: argparse.Namespace) -> int:
    from celltrace.prediction.evaluate import METRICS_JSON, run_evaluate

    model = Path(arguments.model)
    splits = Path(arguments.splits)
    snapshot = Path(arguments.snapshot)
    _require_snapshot(snapshot)
    _require_file(model, "model artifact")
    _require_file(splits, "split manifest")
    predictions_dir = Path(arguments.predictions_out) if arguments.predictions_out else None
    payload = run_evaluate(model, splits, snapshot, Path(arguments.out), predictions_dir)
    _print_json(
        {
            "metrics": str(Path(arguments.out) / METRICS_JSON),
            "figures": payload["figures"],
            "failed_fits": len(payload["failed_fits"]),
            "session_holdout_mae_mbps": {
                name: summary["mae_mbps"]
                for name, summary in payload["session_holdout"]["overall"].items()
            },
            "forward_time_mae_mbps": {
                name: summary["mae_mbps"]
                for name, summary in payload["forward_time"]["overall"].items()
            },
        }
    )
    return EXIT_OK


def command_calibrate_residual(arguments: argparse.Namespace) -> int:
    from celltrace.prediction.residual import run_calibrate, threshold_path

    model = Path(arguments.model)
    splits = Path(arguments.splits)
    snapshot = Path(arguments.snapshot)
    _require_snapshot(snapshot)
    _require_file(model, "model artifact")
    _require_file(splits, "split manifest")
    frozen = run_calibrate(model, snapshot, splits, Path(arguments.out))
    calibration = frozen["calibration"]
    _print_json(
        {
            "frozen_threshold": str(threshold_path(model)),
            "threshold_hash": frozen["threshold_hash"],
            "forecaster": calibration["forecaster"],
            "threshold_mbps": calibration["threshold_mbps"],
            "quantile": calibration["quantile"],
            "dev_grouped_flags": calibration["dev_grouped_flags"],
            "dev_eligible_rows": calibration["dev_eligible_rows"],
            "dev_mae_mbps": calibration["dev_mae_mbps"],
        }
    )
    return EXIT_OK


def command_error_analysis(arguments: argparse.Namespace) -> int:
    from celltrace.prediction.error_analysis import ANALYSIS_JSON, run_error_analysis

    manifest = Path(arguments.predictions)
    _require_file(manifest, "prediction manifest")
    payload = run_error_analysis(manifest, Path(arguments.out))
    _print_json(
        {
            "analysis": str(Path(arguments.out) / ANALYSIS_JSON),
            "figures": payload["figures"],
            "regime_row_counts": {
                split_kind: split["regime_row_counts"]
                for split_kind, split in payload["splits"].items()
            },
        }
    )
    return EXIT_OK


def command_cross_validate(arguments: argparse.Namespace) -> int:
    from celltrace.prediction.cross_validation import run_cross_validation

    summary = run_cross_validation(
        Path(arguments.snapshot), Path(arguments.out), Path(arguments.private_out)
    )
    point = summary["point"]
    print(
        json.dumps(
            {
                "sessions": summary["sessions"],
                "rows": summary["rows"],
                "combined_mae_skill_vs_persistence": point["combined_mean"]["skill_vs_persistence"][
                    "value"
                ],
                "forward_chaining_combined_mae_skill_vs_persistence": summary["forward_chaining"][
                    "point"
                ]["combined_mean"]["skill_vs_persistence"]["value"],
                "report": str(Path(arguments.out) / "cross_validation_report.md"),
            },
            indent=2,
        )
    )
    return 0


def command_external_validate(arguments: argparse.Namespace) -> int:
    from celltrace.prediction.external import run_external_validation

    summary = run_external_validation(
        Path(arguments.snapshot),
        Path(arguments.pam),
        Path(arguments.plan_freeze),
        Path(arguments.out),
        Path(arguments.private_out),
    )
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "decisions": summary["scores"]["decisions"],
                "report": str(Path(arguments.out) / "external_validation_report.md"),
            },
            indent=2,
        )
    )
    return 0


def register(subparsers: Any) -> None:
    predict = subparsers.add_parser("predict", help="forecasting splits, models and residuals")
    actions = predict.add_subparsers(dest="predict_command", required=True)

    splits = actions.add_parser("splits", help="whole session and forward time split manifest")
    splits.add_argument("--snapshot", required=True)
    splits.add_argument("--out", required=True)
    splits.set_defaults(handler=command_splits)

    train = actions.add_parser("train", help="select on dev and fit the boosting models")
    train.add_argument("--snapshot", required=True)
    train.add_argument("--splits", required=True)
    train.add_argument("--out", default=str(DEFAULT_MODEL_DIR))
    train.set_defaults(handler=command_train)

    evaluate = actions.add_parser("evaluate", help="test errors, ablation and figures")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--splits", required=True)
    evaluate.add_argument("--snapshot", required=True)
    evaluate.add_argument("--out", default=str(DEFAULT_PUBLIC_DIR))
    evaluate.add_argument(
        "--predictions-out", default=None, help="new private prediction directory"
    )
    evaluate.set_defaults(handler=command_evaluate)

    calibrate = actions.add_parser(
        "calibrate-residual", help="freeze a dev only residual threshold for the model"
    )
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--snapshot", required=True)
    calibrate.add_argument("--splits", required=True)
    calibrate.add_argument("--out", default=str(DEFAULT_PUBLIC_DIR))
    calibrate.set_defaults(handler=command_calibrate_residual)

    cross_validate = actions.add_parser(
        "cross-validate",
        help="nested session cross validation: leave one session out and forward chaining",
    )
    cross_validate.add_argument("--snapshot", required=True)
    cross_validate.add_argument("--out", required=True, help="aggregate report and figures")
    cross_validate.add_argument("--private-out", required=True, help="per row predictions")
    cross_validate.set_defaults(handler=command_cross_validate)

    external = actions.add_parser(
        "external-validate",
        help="frozen, run once validation of the history only forecast on PAM 2025 drives",
    )
    external.add_argument("--snapshot", required=True, help="Raca training snapshot")
    external.add_argument("--pam", required=True, help="admitted PAM 2025 directory")
    external.add_argument("--plan-freeze", required=True, help="hash record of the frozen plan")
    external.add_argument("--out", required=True, help="aggregate report and figure")
    external.add_argument("--private-out", required=True, help="per row predictions")
    external.set_defaults(handler=command_external_validate)

    error_analysis = actions.add_parser(
        "error-analysis", help="saved forecast errors by session and regime, with intervals"
    )
    error_analysis.add_argument(
        "--predictions", required=True, help="manifest.json written beside saved predictions"
    )
    error_analysis.add_argument("--out", required=True)
    error_analysis.set_defaults(handler=command_error_analysis)
