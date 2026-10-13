"""Workflow orchestration with a fake subprocess. Real stages are exercised by running it."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from celltrace.canonical import content_hash
from celltrace.evaluation.review.plan import PROTOCOL_VERSION, experiment_id

script = Path(__file__).parents[2] / "scripts/run_quality_workflow.py"
specification = importlib.util.spec_from_file_location("quality_workflow", script)
assert specification is not None and specification.loader is not None
workflow_module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = workflow_module
specification.loader.exec_module(workflow_module)

Outcome = Callable[[list[str]], int]


def demo_summary(scenarios: int = 6) -> dict[str, Any]:
    return {
        "status": "passed",
        "scenarios": [{"scenario": f"s{index}", "status": "passed"} for index in range(scenarios)],
    }


class FakeCommands:
    """Records every command and imitates the files the real commands would write."""

    def __init__(self, root: Path, exit_code: Outcome, gate: str = "passed"):
        self.root = root
        self.exit_code = exit_code
        self.gate = gate
        self.commands: list[list[str]] = []
        self.demo_copies = 1

    def __call__(self, command: list[str], **keywords: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if "demo" in command:
            out = Path(command[command.index("--out") + 1])
            for copy in range(self.demo_copies):
                execution = out / f"execution_{copy:016d}"
                execution.mkdir(parents=True)
                (execution / "demo_summary.json").write_text(json.dumps(demo_summary()))
                private = self.root / "artifacts/private/demo" / execution.name
                private.mkdir(parents=True, exist_ok=True)
                (private / "runs.sqlite3").write_text("")
        if "summarize" in command:
            public = Path(command[command.index("--public-out") + 1])
            public.mkdir(parents=True)
            summary = {"gate": {"status": self.gate}, "known_or_reserved_usd": 0.42}
            (public / "summary.json").write_text(json.dumps(summary))
        return subprocess.CompletedProcess(command, self.exit_code(command))


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "uv.lock").write_text("lock fixture\n")
    monkeypatch.setattr(workflow_module, "ROOT", tmp_path)
    return tmp_path


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeCommands) -> FakeCommands:
    monkeypatch.setattr(workflow_module.subprocess, "run", fake)
    return fake


def test_offline_workflow_runs_every_stage_and_never_pays_for_review(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, FakeCommands(root, lambda command: 0))
    execution, result = workflow_module.workflow(root / "runs")
    assert result.status == "passed" and result.review_gate == "not_run"
    assert [stage.name for stage in result.stages] == [
        "sync",
        "lint",
        "format",
        "types",
        "tests",
        "demo",
        "viewer",
        "traces",
    ]
    assert not any("review" in command for command in fake.commands)
    assert (execution / "workflow.md").is_file()
    assert json.loads((execution / "workflow.json").read_text())["status"] == "passed"


def test_a_failed_stage_stops_the_workflow_and_stays_on_record(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeCommands(root, lambda command: 1 if "ruff" in command else 0))
    execution, result = workflow_module.workflow(root / "runs")
    assert result.status == "failed"
    assert [(stage.name, stage.exit_code) for stage in result.stages] == [("sync", 0), ("lint", 1)]
    recorded = json.loads((execution / "workflow.json").read_text())
    assert recorded["stages"][-1]["status"] == "failed"
    assert "lint failed" in (execution / "failure.txt").read_text()


def test_ambiguous_demo_output_is_rejected(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCommands(root, lambda command: 0)
    fake.demo_copies = 2
    install(monkeypatch, fake)
    _, result = workflow_module.workflow(root / "runs")
    assert result.status == "failed"
    assert result.stages[-1].name == "demo"


@pytest.mark.parametrize(("gate", "status"), [("passed", "passed"), ("controls_failed", "failed")])
def test_the_frozen_review_gate_decides_the_paid_stage(
    root: Path, monkeypatch: pytest.MonkeyPatch, gate: str, status: str
) -> None:
    fake = install(monkeypatch, FakeCommands(root, lambda command: 0, gate=gate))
    _, result = workflow_module.workflow(
        root / "runs", review_plan=root / "plan", env_file=root / ".env"
    )
    assert result.status == status and result.review_gate == gate
    assert result.review_usd == 0.42
    run_command = next(
        command for command in fake.commands if "run" in command and "review" in command
    )
    assert run_command[-1] == "--live" and "--env-file" in run_command


def test_a_failed_paid_stage_keeps_known_and_uncertain_charges(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = root / "plan"
    review.mkdir()
    plan = {"protocol_version": PROTOCOL_VERSION, "settings": {"maximum_estimated_usd": 2.5}}
    plan["plan_hash"] = content_hash(plan)
    (review / "plan.json").write_text(json.dumps(plan))
    # Minimal persisted expense records: one receipt and one uncertain paid attempt.
    with sqlite3.connect(review / "runs.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE experiments (experiment_id, maximum_estimated_usd, payload_json)"
        )
        connection.execute("CREATE TABLE runs (run_id, experiment_id)")
        connection.execute("CREATE TABLE actions (run_id, usage_known, actual_usd, reserved_usd)")
        identifier = experiment_id(plan["plan_hash"])
        connection.execute(
            "INSERT INTO experiments VALUES (?, ?, ?)", (identifier, 2.5, json.dumps(plan))
        )
        connection.execute("INSERT INTO runs VALUES (?, ?)", ("paid_run", identifier))
        connection.executemany(
            "INSERT INTO actions VALUES (?, ?, ?, ?)",
            [("paid_run", 1, 0.05, 0.1), ("paid_run", 0, None, 0.2)],
        )
    install(
        monkeypatch,
        FakeCommands(root, lambda command: 1 if "review" in command and "run" in command else 0),
    )
    execution, result = workflow_module.workflow(root / "runs", review_plan=review)
    assert result.status == "failed" and result.review_gate == "execution_failed"
    assert result.review_usd == pytest.approx(0.25)
    assert json.loads((execution / "workflow.json").read_text())["review_usd"] == pytest.approx(
        0.25
    )


def test_population_stage_failure_stops_the_same_workflow(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install(
        monkeypatch, FakeCommands(root, lambda command: 3 if "population" in command else 0)
    )
    _, result = workflow_module.workflow(root / "runs", population_snapshot=root / "snapshot")
    assert result.status == "failed" and result.stages[-1].name == "population"
    assert "--private-out" in fake.commands[-1]
    assert not any("--live" in command for command in fake.commands)
