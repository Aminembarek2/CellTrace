"""Thin entry point for frozen multi-resource development studies."""

import argparse
import json
from pathlib import Path
from typing import Any

from celltrace.evaluation import joint_resources
from celltrace.evaluation.review.persistence import write_json_atomic


def command(arguments: argparse.Namespace) -> int:
    output = Path(arguments.out)
    if arguments.action == "prepare":
        result = joint_resources.prepare(
            Path(arguments.manifest),
            Path(arguments.config),
            output,
            Path(arguments.prior_study) if arguments.prior_study else None,
        )
        result = {key: result[key] for key in ("plan_hash", "model_id", "scope")}
    elif arguments.action == "summarize":
        result = joint_resources.summarize(output, Path(arguments.public_out))
    else:
        from celltrace.cli import CommandError, build_adapter
        from celltrace.config import api_key_present

        if not arguments.live or not api_key_present():
            raise CommandError("joint resources need --live and configured credentials")
        plan = joint_resources.load(output)
        try:
            result = joint_resources.run(output, lambda config: build_adapter(config, True))
        finally:
            write_json_atomic(output / "expenses.json", joint_resources.expenses(plan))
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "finished") == "finished" else 3


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "joint-resources", help="frozen development contrast at equal budget"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "run", "summarize"):
        action = actions.add_parser(name)
        action.add_argument("--out", required=True)
        action.set_defaults(handler=command)
        if name == "prepare":
            action.add_argument("--manifest", required=True)
            action.add_argument("--config", required=True)
            action.add_argument(
                "--prior-study", help="carry prior charges within the original envelope"
            )
        elif name == "run":
            action.add_argument("--live", action="store_true")
        else:
            action.add_argument("--public-out", required=True)
