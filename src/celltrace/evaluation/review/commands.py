"""The review command group: prepare a frozen plan, run it live, summarize it.

Configuration files are resolved into typed judge profiles here, so the layers below never read
configuration or credentials themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from celltrace.canonical import file_sha256
from celltrace.clock import SystemClock
from celltrace.evaluation.review.judge import CallLimits
from celltrace.evaluation.review.plan import JudgeProfile, ReviewSettings, load_plan, prepare_plan

EXIT_OK = 0


def judge_profile(configuration_path: Path) -> JudgeProfile:
    """A judge from a fixed profile configuration whose endpoint and prices are verified now."""
    from celltrace.config import load_config
    from celltrace.providers.readiness import verified_rate_card

    configuration = load_config(configuration_path, environment={})
    rates = verified_rate_card(configuration.runtime)
    provenance = {str(configuration_path): file_sha256(configuration_path)}
    for dependency in (
        configuration.runtime.verified_endpoint_record,
        configuration.runtime.verified_rate_card,
    ):
        if dependency is None:
            raise ValueError(f"{configuration_path} lacks provider verification")
        provenance[str(dependency)] = file_sha256(dependency)
    return JudgeProfile(
        model_id=rates.model_id,
        rate_card=rates,
        configuration_path=str(configuration_path),
        provenance_hashes=provenance,
    )


def _reverify_judges(plan: dict[str, Any]) -> None:
    """Endpoint records expire, so prices are verified again immediately before paid calls."""
    for judge in plan["judges"]:
        current = judge_profile(Path(judge["configuration_path"]))
        if current.rate_card.model_dump(mode="json") != judge["rate_card"]:
            raise ValueError(f"verified prices for {judge['model_id']} changed since the plan")


def command_prepare(arguments: argparse.Namespace) -> int:
    judges = [judge_profile(Path(path)) for path in arguments.judge_config]
    settings = ReviewSettings(
        maximum_estimated_usd=arguments.ceiling_usd,
        repeat_sample_size=arguments.repeat_sample,
        call_limits=CallLimits(),
        controls_only=arguments.controls_only,
    )
    result = prepare_plan(
        Path(arguments.packet), judges, settings, Path(arguments.out), SystemClock()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_OK


def command_run(arguments: argparse.Namespace) -> int:
    from celltrace.cli import EXIT_BLOCKED, CommandError
    from celltrace.config import api_key_present
    from celltrace.evaluation.review.runner import run_review
    from celltrace.providers.openrouter_adapter import OpenRouterAdapter

    if not arguments.live:
        raise CommandError("review run sends report text to paid judges. Add --live to proceed")
    output = Path(arguments.out)
    plan = load_plan(output)
    try:
        _reverify_judges(plan)
    except (ValueError, OSError) as failure:
        raise CommandError(f"blocked judge readiness: {failure}", EXIT_BLOCKED) from failure
    if not api_key_present():
        raise CommandError("blocked: OPENROUTER_API_KEY is not set", EXIT_BLOCKED)
    progress = run_review(
        output,
        OpenRouterAdapter,
        SystemClock(),
        limit=arguments.limit,
        emit=lambda event: print(json.dumps(event, sort_keys=True), flush=True),
    )
    print(json.dumps(progress, indent=2, sort_keys=True))
    return EXIT_OK


def command_summarize(arguments: argparse.Namespace) -> int:
    from celltrace.evaluation.review.summary import summarize_review

    summary = summarize_review(Path(arguments.out), Path(arguments.public_out))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "plan_hash",
                    "panel_status",
                    "panel_judges",
                    "status_counts",
                    "known_or_reserved_usd",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def register(subparsers: Any) -> None:
    review = subparsers.add_parser(
        "review", help="automated semantic review of saved reports by fixed model judges"
    )
    actions = review.add_subparsers(dest="review_command", required=True)

    prepare = actions.add_parser("prepare", help="freeze judges, rubric, controls and schedule")
    prepare.add_argument("--packet", required=True, help="review packet written by scoring")
    prepare.add_argument(
        "--judge-config",
        required=True,
        action="append",
        help="fixed profile configuration, repeat for each judge",
    )
    prepare.add_argument("--out", required=True, help="new private plan directory")
    prepare.add_argument("--ceiling-usd", type=float, default=ReviewSettings.maximum_estimated_usd)
    prepare.add_argument("--repeat-sample", type=int, default=ReviewSettings.repeat_sample_size)
    prepare.add_argument(
        "--controls-only", action="store_true", help="qualify judges without dispatching reports"
    )
    prepare.set_defaults(handler=command_prepare)

    run = actions.add_parser("run", help="judge pending items within the frozen ceiling")
    run.add_argument("--out", required=True, help="plan directory written by prepare")
    run.add_argument("--live", action="store_true", help="confirm paid calls to the judges")
    run.add_argument("--limit", type=int, default=None, help="stop after this many items")
    run.set_defaults(handler=command_run)

    summarize = actions.add_parser("summarize", help="qualification, panel verdicts and spend")
    summarize.add_argument("--out", required=True, help="plan directory written by prepare")
    summarize.add_argument("--public-out", required=True, help="new directory for aggregates")
    summarize.set_defaults(handler=command_summarize)
