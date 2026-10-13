"""A role reserve remains charged across failures, cache hits and reopening the ledger."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from celltrace.clock import ManualClock
from celltrace.config import load_config
from celltrace.contracts.common import Arm
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    BudgetExhausted,
    Reservation,
    RunLimits,
    RunStore,
)
from celltrace.workflow.runner import run_limits


def store_at(path: Path, reserve: int = 2) -> RunStore:
    store = RunStore(path, ManualClock())
    limits = RunLimits(6, 3, 1000, 1000, None, 1000, 2, 1, reserve)
    store.create_run(
        "run",
        "case",
        "event",
        "investigator_reviewer_gated",
        "scripted_fixture",
        None,
        "snapshot",
        {},
        {},
        limits,
    )
    return store


def request(store: RunStore, role: str, key: str) -> int:
    return store.reserve_action(
        "run", key, key, role, ActionKind.tool_request, {}, Reservation(tool_units=1)
    )


def test_failed_and_cached_investigator_requests_cannot_take_the_reserve(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    store = store_at(path)
    for index in range(4):
        action = request(store, "investigator", str(index))
        store.complete_action(
            action,
            ActionState.failed if index == 0 else ActionState.completed,
            0,
            cache_hit=index == 1,
        )
    assert store.tool_allowance("run", "investigator")["role_remaining"] == 0
    with pytest.raises(BudgetExhausted, match="tool_requests:investigator"):
        request(store, "investigator", "revision")
    store.close()
    reopened = RunStore(path, ManualClock())
    assert reopened.tool_allowance("run", "reviewer")["role_remaining"] == 2
    request(reopened, "reviewer", "review1")
    request(reopened, "reviewer", "review2")
    with pytest.raises(BudgetExhausted, match="tool_requests"):
        request(reopened, "reviewer", "review3")
    assert reopened.usage("run").tool_requests == 6
    reopened.close()


def test_reviewer_can_borrow_unused_capacity_and_unknown_roles_cannot_bypass(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path / "runs.sqlite3")
    with pytest.raises(ValueError, match="investigator/reviewer"):
        request(store, "another_role", "bypass")
    request(store, "investigator", "one")
    for index in range(5):
        request(store, "reviewer", f"review{index}")
    assert store.tool_allowance("run", "investigator")["role_remaining"] == 0
    assert store.usage("run").tool_requests == 6
    store.close()


def test_reserve_zero_retains_shared_pool_and_old_json_reads(tmp_path: Path) -> None:
    store = store_at(tmp_path / "runs.sqlite3", reserve=0)
    for index in range(6):
        request(store, "investigator", str(index))
    assert store.usage("run").tool_requests == 6
    old = asdict(store.run_limits("run"))
    old.pop("reviewer_tool_reserve")
    assert RunLimits(**old).reviewer_tool_reserve == 0
    store.close()


def test_reserve_is_applied_only_to_the_reviewed_arm() -> None:
    configuration = load_config(
        environment={},
        cli_overrides={
            ("runtime", "maximum_tool_requests"): 6,
            ("runtime", "reviewer_tool_reserve"): 2,
        },
    )
    assert run_limits(configuration, None, Arm.investigator_gated).reviewer_tool_reserve == 0
    assert (
        run_limits(configuration, None, Arm.investigator_reviewer_gated).reviewer_tool_reserve == 2
    )
    with pytest.raises(ValueError):
        load_config(
            environment={},
            cli_overrides={
                ("runtime", "maximum_tool_requests"): 2,
                ("runtime", "reviewer_tool_reserve"): 2,
            },
        )
    with pytest.raises(ValueError):
        replace(run_limits(configuration, None), reviewer_tool_reserve=-1)
