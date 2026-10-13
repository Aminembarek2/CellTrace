"""Thin CLI for explicitly paid, frozen allowance calibration."""

import argparse
import json
from pathlib import Path
from typing import Any

from celltrace.evaluation.allowances import (
    load_study,
    prepare_study,
    run_study,
    study_expenses,
    summarize_study,
)


def command_prepare(arguments: argparse.Namespace) -> int:
    plan = prepare_study(Path(arguments.manifest), Path(arguments.config), Path(arguments.out))
    print(json.dumps({key: plan[key] for key in ("plan_hash", "model_id", "ceilings", "scope")}))
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    from celltrace.cli import CommandError, build_adapter
    from celltrace.config import api_key_present
    from celltrace.evaluation.review.persistence import write_json_atomic

    if not arguments.live or not api_key_present():
        raise CommandError("allowance calibration needs --live and configured credentials")
    output = Path(arguments.out)
    plan = load_study(output)
    try:
        result = run_study(output, lambda config: build_adapter(config, True))
        print(json.dumps(result, indent=2))
    finally:
        expenses = study_expenses(output, plan)
        write_json_atomic(output / "expenses.json", expenses)
        print(json.dumps({"expenses": expenses}), flush=True)
    return 0 if result.get("status") == "finished" else 3


def command_summarize(arguments: argparse.Namespace) -> int:
    summary = summarize_study(Path(arguments.out), Path(arguments.public_out))
    print(
        json.dumps({key: summary[key] for key in ("recommendation", "gates", "expenses")}, indent=2)
    )
    return 0


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("allowances", help="development tool allowances for each model")
    actions = parser.add_subparsers(dest="allowance_action", required=True)
    prepare = actions.add_parser("prepare", help="freeze development cases, model and ceilings")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--out", required=True)
    prepare.set_defaults(handler=command_prepare)
    run = actions.add_parser("run", help="profile once and compare capacity versus allocation")
    run.add_argument("--out", required=True)
    run.add_argument("--live", action="store_true")
    run.set_defaults(handler=command_run)
    summarize = actions.add_parser(
        "summarize", help="replay outcomes and draw cost/completion plot"
    )
    summarize.add_argument("--out", required=True)
    summarize.add_argument("--public-out", required=True)
    summarize.set_defaults(handler=command_summarize)
