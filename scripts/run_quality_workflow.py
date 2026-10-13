"""Run the whole evidence workflow as logged stages, the same way locally and in CI.

Default stages are offline and need no credentials: locked sync, lint, format, types, offline
tests, the scripted demo, its run viewer and its trace export. Saved
forecast error analysis runs when a prediction manifest is given. The paid semantic review runs
only when a frozen review plan is given with --review, and its frozen gate decides the result.

Each execution gets its own directory with one log per stage, an atomically updated
workflow.json and a workflow.md report. A failed stage stops the run and is kept on record.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from celltrace.canonical import file_sha256
from celltrace.release import source_tree_hash

ROOT = Path(__file__).resolve().parents[1]
DEMO_SCENARIOS = 6
CELLTRACE = ("uv", "run", "celltrace")


@dataclass
class StageResult:
    name: str
    command: list[str]
    status: Literal["running", "passed", "failed"]
    exit_code: int | None
    elapsed_seconds: float
    log: str
    detail: str = ""


@dataclass
class WorkflowResult:
    execution_id: str
    mode: Literal["offline", "with_review"]
    source_sha256: str
    lock_sha256: str
    status: Literal["running", "passed", "failed"] = "running"
    stages: list[StageResult] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    review_gate: str = "not_run"
    review_usd: float = 0.0


class WorkflowFailure(RuntimeError):
    pass


def save_result(execution: Path, result: WorkflowResult) -> None:
    """Atomic snapshots, so an interruption never leaves a truncated status file."""
    temporary = execution / "workflow.json.tmp"
    temporary.write_text(json.dumps(asdict(result), indent=2) + "\n")
    temporary.replace(execution / "workflow.json")


def run_stage(execution: Path, result: WorkflowResult, name: str, command: list[str]) -> None:
    log = execution / f"{len(result.stages):02d}_{name}.log"
    stage = StageResult(name, command, "running", None, 0.0, str(log))
    result.stages.append(stage)
    save_result(execution, result)
    started = time.monotonic()
    try:
        with log.open("w") as handle:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        stage.exit_code = completed.returncode
        stage.status = "passed" if completed.returncode == 0 else "failed"
    except OSError as failure:
        stage.status = "failed"
        stage.detail = f"{type(failure).__name__}: {failure}"
    finally:
        stage.elapsed_seconds = round(time.monotonic() - started, 3)
        save_result(execution, result)
    print(json.dumps(asdict(stage)), flush=True)
    if stage.status != "passed":
        raise WorkflowFailure(f"{name} failed, see {log}")


def demo_artifacts(execution: Path) -> tuple[Path, Path]:
    """The single demo summary and its private run store, after checking every scenario."""
    summaries = list((execution / "demo").rglob("demo_summary.json"))
    if len(summaries) != 1:
        raise WorkflowFailure(f"expected one demo summary, found {len(summaries)}")
    path = summaries[0]
    summary = json.loads(path.read_text())
    scenarios = summary.get("scenarios", [])
    if (
        summary.get("status") != "passed"
        or len(scenarios) != DEMO_SCENARIOS
        or any(scenario.get("status") != "passed" for scenario in scenarios)
    ):
        raise WorkflowFailure(f"demo did not pass all {DEMO_SCENARIOS} declared scenarios")
    # The demo uses the same execution identifier for its public and private folders.
    store = ROOT / "artifacts/private/demo" / path.parent.name / "runs.sqlite3"
    if not store.is_file():
        raise WorkflowFailure("demo run store is missing")
    return path, store


def offline_commands(execution: Path) -> list[tuple[str, list[str]]]:
    return [
        ("sync", ["uv", "sync", "--locked", "--group", "dev"]),
        ("lint", ["uv", "run", "ruff", "check", "."]),
        ("format", ["uv", "run", "ruff", "format", "--check", "."]),
        ("types", ["uv", "run", "mypy", "src/celltrace"]),
        ("tests", ["uv", "run", "pytest", "-m", "not live and not external_data"]),
        ("demo", [*CELLTRACE, "demo", "--offline", "--out", str(execution / "demo")]),
    ]


def review_commands(
    plan: Path, summary: Path, env_file: Path | None
) -> list[tuple[str, list[str]]]:
    credentials = [] if env_file is None else ["--env-file", str(env_file)]
    return [
        (
            "review_run",
            [*CELLTRACE, *credentials, "review", "run", "--out", str(plan), "--live"],
        ),
        (
            "review_summary",
            [*CELLTRACE, "review", "summarize", "--out", str(plan), "--public-out", str(summary)],
        ),
    ]


def write_report(execution: Path, result: WorkflowResult) -> None:
    lines = [
        "# Evidence workflow",
        "",
        f"Status: **{result.status}**. Mode: `{result.mode}`. "
        f"Source `{result.source_sha256[:16]}`.",
        "",
        "| Stage | Result | Exit | Seconds | Log |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for stage in result.stages:
        name = Path(stage.log).name
        lines.append(
            f"| {stage.name} | {stage.status} | {stage.exit_code} | {stage.elapsed_seconds} "
            f"| [{name}]({name}) |"
        )
    lines += [
        "",
        f"Semantic review gate: `{result.review_gate}`. Review cost known or reserved: "
        f"USD {result.review_usd:.6f}.",
        "",
        "The demo is a scripted synthetic fixture. Automated review verdicts are model opinions "
        "under a frozen rubric. They establish neither physical cause nor human agreement.",
        "",
        "Artifacts:",
        "",
    ]
    for name, path in result.artifacts.items():
        lines.append(f"- [{name}]({os.path.relpath(Path(path).resolve(), execution)})")
    (execution / "workflow.md").write_text("\n".join(lines) + "\n")


def workflow(
    output: Path,
    review_plan: Path | None = None,
    predictions: Path | None = None,
    env_file: Path | None = None,
    population_snapshot: Path | None = None,
    planted_plan: Path | None = None,
) -> tuple[Path, WorkflowResult]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = output / ("execution_" + uuid4().hex[:16])
        execution.mkdir()
        result = WorkflowResult(
            execution.name,
            "with_review" if review_plan else "offline",
            source_tree_hash(),
            file_sha256(ROOT / "uv.lock"),
        )
        save_result(execution, result)
        try:
            for name, command in offline_commands(execution):
                run_stage(execution, result, name, command)
            summary_path, store = demo_artifacts(execution)
            result.artifacts["offline_demo"] = str(summary_path)
            for name, action, target in (
                ("viewer", "view", execution / "demo_viewer.html"),
                ("traces", "export-otel", execution / "demo_traces.json"),
            ):
                run_stage(
                    execution,
                    result,
                    name,
                    [*CELLTRACE, "runs", action, "--store", str(store), "--out", str(target)],
                )
                result.artifacts[name] = str(target)
            if predictions is not None:
                target = execution / "forecast_errors"
                run_stage(
                    execution,
                    result,
                    "forecast_errors",
                    [
                        *CELLTRACE,
                        "predict",
                        "error-analysis",
                        "--predictions",
                        str(predictions.resolve()),
                        "--out",
                        str(target),
                    ],
                )
                result.artifacts["forecast_errors"] = str(target)
            if population_snapshot is not None:
                target = execution / "population"
                private = execution / "population_private"
                run_stage(
                    execution,
                    result,
                    "population",
                    [
                        *CELLTRACE,
                        "population",
                        "--snapshot",
                        str(population_snapshot.resolve()),
                        "--out",
                        str(target),
                        "--private-out",
                        str(private),
                    ],
                )
                result.artifacts["population_report"] = str(target / "population_report.md")
                result.artifacts["population_queue_private"] = str(
                    private / "investigation_queue.json"
                )
            if planted_plan is not None:
                target = execution / "planted_errors"
                run_stage(
                    execution,
                    result,
                    "planted_errors_offline",
                    [
                        *CELLTRACE,
                        "planted-errors",
                        "run",
                        "--plan",
                        str(planted_plan.resolve()),
                        "--execution",
                        str(target),
                        "--offline",
                    ],
                )
                result.artifacts["planted_errors_offline"] = str(target / "summary.json")
            if review_plan is not None:
                summary_directory = execution / "review_summary"
                for name, command in review_commands(
                    review_plan.resolve(), summary_directory, env_file
                ):
                    run_stage(execution, result, name, command)
                summary: dict[str, Any] = json.loads(
                    (summary_directory / "summary.json").read_text()
                )
                result.artifacts["review_summary"] = str(summary_directory / "summary.json")
                result.review_usd = summary["known_or_reserved_usd"]
                result.review_gate = summary["gate"]["status"]
                if result.review_gate != "passed":
                    raise WorkflowFailure(f"semantic review gate: {result.review_gate}")
            result.status = "passed"
        except (WorkflowFailure, ValueError, OSError) as failure:
            result.status = "failed"
            (execution / "failure.txt").write_text(str(failure) + "\n")
            result.artifacts["failure"] = str(execution / "failure.txt")
        finally:
            if review_plan is not None and (review_plan / "plan.json").is_file():
                # Preserve paid/unknown charges even when the subprocess failed before summary.
                from celltrace.evaluation.review.ledger import experiment_usage
                from celltrace.evaluation.review.plan import recorded_plan

                try:
                    result.review_usd = experiment_usage(review_plan, recorded_plan(review_plan))
                    if any(
                        stage.name == "review_run" and stage.status == "failed"
                        for stage in result.stages
                    ):
                        result.review_gate = "execution_failed"
                    for name in ("runs.sqlite3", "progress.json"):
                        if (review_plan / name).is_file():
                            result.artifacts["review_" + name] = str(review_plan / name)
                except (ValueError, OSError) as failure:
                    result.status = "failed"
                    result.review_gate = "ledger_replay_failed"
                    (execution / "ledger_failure.txt").write_text(str(failure) + "\n")
                    result.artifacts["ledger_failure"] = str(execution / "ledger_failure.txt")
            save_result(execution, result)
            write_report(execution, result)
        return execution, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="workflow executions directory")
    parser.add_argument(
        "--review", type=Path, help="frozen review plan to run live, which spends money"
    )
    parser.add_argument("--predictions", type=Path, help="saved prediction manifest, no fitting")
    parser.add_argument(
        "--population-snapshot", type=Path, help="readonly population analysis, no model calls"
    )
    parser.add_argument("--env-file", type=Path, help="credential file for the paid review")
    parser.add_argument(
        "--planted-plan", type=Path, help="frozen planted-error controls, run offline only"
    )
    arguments = parser.parse_args()
    execution, result = workflow(
        arguments.out,
        arguments.review,
        arguments.predictions,
        arguments.env_file,
        arguments.population_snapshot,
        arguments.planted_plan,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "execution": str(execution),
                "review_gate": result.review_gate,
                "review_usd": result.review_usd,
            },
            indent=2,
        )
    )
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
