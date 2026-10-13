"""Future review work stays protected in the same durable multi-resource ledger."""

from dataclasses import replace
from pathlib import Path

import pytest

from celltrace.clock import ManualClock
from celltrace.config import load_config
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    BudgetExhausted,
    Reservation,
    RunLimits,
    RunStore,
)


def joint_store(path: Path) -> RunStore:
    store = RunStore(path, ManualClock())
    limits = RunLimits(
        6,
        6,
        600,
        600,
        None,
        1000,
        2,
        1,
        review_round_tool_reserve=1,
        review_round_provider_reserve=1,
        review_round_input_reserve=100,
        review_round_output_reserve=100,
    )
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


def provider(store: RunStore, role: str, key: str, reservation: Reservation) -> int:
    return store.reserve_action("run", key, key, role, ActionKind.provider_attempt, {}, reservation)


@pytest.mark.parametrize(
    "resource,reservation",
    [
        ("input_tokens", Reservation(input_tokens=401)),
        ("output_tokens", Reservation(output_tokens=401)),
    ],
)
def test_investigator_cannot_use_future_round_token_protection(
    tmp_path: Path, resource: str, reservation: Reservation
) -> None:
    store = joint_store(tmp_path / "runs.sqlite3")
    with pytest.raises(BudgetExhausted, match=resource + ":future_review"):
        provider(store, "investigator", "too-large", reservation)
    assert store.usage("run").provider_attempts == 0
    store.close()


def test_cached_input_and_uncertain_output_are_not_refunded_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    store = joint_store(path)
    action = provider(
        store, "investigator", "cached", Reservation(input_tokens=400, output_tokens=100)
    )
    store.complete_action(
        action,
        ActionState.completed,
        0,
        usage={
            "usage_known": True,
            "input_tokens": 100,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 50,
            "output_tokens": 20,
            "usd": 0.0,
        },
    )
    pending = provider(
        store, "reviewer", "unknown", Reservation(input_tokens=100, output_tokens=200)
    )
    store.mark_started(pending)
    store.mark_uncertain(pending, "connection lost after dispatch")
    before = store.resource_allowance("run", "investigator")
    assert before["input_tokens"].role_remaining == 0
    assert before["output_tokens"].used == 220
    store.close()
    reopened = RunStore(path, ManualClock())
    assert reopened.resource_allowance("run", "investigator") == before
    with pytest.raises(BudgetExhausted, match="input_tokens:future_review"):
        provider(reopened, "investigator", "revision", Reservation(input_tokens=1))
    reopened.close()


def test_round_release_is_saved_review_driven_and_idempotent(tmp_path: Path) -> None:
    store = joint_store(tmp_path / "runs.sqlite3")
    assert store.resource_allowance("run", "reviewer")["tool_requests"].role_remaining == 5
    action = provider(store, "reviewer", "first-review", Reservation())
    store.complete_action(
        action,
        ActionState.completed,
        0,
        usage={"usage_known": True, "input_tokens": 0, "output_tokens": 0, "usd": 0},
    )
    findings, _ = store.save_findings("run", {"event_id": "event"}, action)
    store.save_review("run", {"decision": "request_more_evidence"}, 1, findings, action)
    store.save_review("run", {"decision": "request_more_evidence"}, 1, findings, action)
    assert len(store.reviews("run")) == 1
    assert store.resource_allowance("run", "investigator")["input_tokens"].protected_future == 100
    assert store.resource_allowance("run", "reviewer")["input_tokens"].protected_future == 0
    store.close()


def test_protection_covers_tool_retries_and_provider_attempts(tmp_path: Path) -> None:
    store = joint_store(tmp_path / "runs.sqlite3")
    for n in range(4):
        action = store.reserve_action(
            "run",
            f"tool-{n}",
            "retried",
            "investigator",
            ActionKind.tool_request,
            {},
            Reservation(tool_units=1),
            attempt_number=n + 1,
        )
        store.complete_action(action, ActionState.failed, 0, cache_hit=n == 0)
        provider(store, "investigator", f"call-{n}", Reservation())
    with pytest.raises(BudgetExhausted, match="tool_requests:future_review"):
        store.reserve_action(
            "run",
            "revision",
            "revision",
            "investigator",
            ActionKind.tool_request,
            {},
            Reservation(tool_units=1),
        )
    with pytest.raises(BudgetExhausted, match="provider_attempts:future_review"):
        provider(store, "investigator", "extra-call", Reservation())
    with pytest.raises(ValueError, match="resource pool"):
        provider(store, "other", "bypass", Reservation())
    store.close()


def test_invalid_protection_and_strict_configuration_are_rejected(tmp_path: Path) -> None:
    store = joint_store(tmp_path / "runs.sqlite3")
    for value in (-1, True, 1.2):
        with pytest.raises(ValueError):
            replace(store.run_limits("run"), review_round_input_reserve=value)
        with pytest.raises(ValueError):
            load_config(
                environment={}, cli_overrides={("runtime", "review_round_input_reserve"): value}
            )
    with pytest.raises(ValueError, match="leave resources"):
        replace(store.run_limits("run"), review_round_tool_reserve=3)
    store.close()
