"""Budget ledger, experiment allowance and run ownership lock."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from celltrace.clock import ManualClock
from celltrace.storage.lock import RunLock, RunLockedError
from celltrace.storage.run_store import (
    ActionKind,
    ActionState,
    BudgetExhausted,
    Reservation,
    RunLimits,
    RunStore,
)

LIMITS = RunLimits(
    maximum_tool_requests=3,
    maximum_provider_attempts=2,
    maximum_input_tokens=1000,
    maximum_output_tokens=500,
    maximum_estimated_usd=0.10,
    maximum_active_ms=10_000,
    maximum_review_rounds=2,
    maximum_request_retries=1,
)


def make_store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs.sqlite3", ManualClock())


def make_run(store: RunStore, run_id: str = "run_a", experiment_id: str | None = None) -> None:
    store.create_run(
        run_id,
        "case",
        "evt",
        "investigator_gated",
        "scripted_fixture",
        None,
        "snap",
        {"identity": 1},
        {"context": 1},
        LIMITS,
        experiment_id=experiment_id,
    )


def reserve_tool(store: RunStore, key: str, run_id: str = "run_a") -> int:
    return store.reserve_action(
        run_id, key, key, "investigator", ActionKind.tool_request, {}, Reservation(tool_units=1)
    )


def test_tool_units_are_charged_for_cache_hits_and_errors(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_run(store)
    first = reserve_tool(store, "t1")
    store.complete_action(first, ActionState.completed, 1, cache_hit=True)
    second = reserve_tool(store, "t2")
    store.complete_action(second, ActionState.failed, 1, error_code="INVALID_ARGUMENT")
    reserve_tool(store, "t3")
    usage = store.usage("run_a")
    assert usage.tool_requests == 3
    assert usage.tool_cache_hits == 1
    with pytest.raises(BudgetExhausted) as exhausted:
        reserve_tool(store, "t4")
    assert exhausted.value.limit_name == "tool_requests"


def test_duplicate_operation_key_is_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_run(store)
    reserve_tool(store, "same")
    with pytest.raises(ValueError, match="already reserved"):
        reserve_tool(store, "same")


def test_unknown_provider_usage_stays_charged_at_reservation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_run(store)
    action = store.reserve_action(
        "run_a",
        "p1",
        "p1",
        "investigator",
        ActionKind.provider_attempt,
        {},
        Reservation(input_tokens=400, output_tokens=200, usd=0.03),
    )
    store.mark_started(action)
    store.mark_uncertain(action, "timeout", duration_ms=120000)
    assert store.raw_actions("run_a")[-1]["duration_ms"] == 120000
    usage = store.usage("run_a")
    assert (usage.input_tokens, usage.output_tokens, usage.unknown_usage_attempts) == (400, 200, 1)
    assert usage.estimated_usd == pytest.approx(0.03)
    known = store.reserve_action(
        "run_a",
        "p2",
        "p2",
        "investigator",
        ActionKind.provider_attempt,
        {},
        Reservation(input_tokens=400, output_tokens=200, usd=0.03),
    )
    store.complete_action(
        known,
        ActionState.completed,
        5,
        usage={"usage_known": True, "input_tokens": 100, "output_tokens": 50, "usd": 0.01},
    )
    usage = store.usage("run_a")
    assert (usage.input_tokens, usage.output_tokens) == (500, 250)
    with pytest.raises(BudgetExhausted) as exhausted:
        store.reserve_action(
            "run_a", "p3", "p3", "investigator", ActionKind.provider_attempt, {}, Reservation()
        )
    assert exhausted.value.limit_name == "provider_attempts"


def test_token_and_money_ceilings_block_reservation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_run(store)
    with pytest.raises(BudgetExhausted) as exhausted:
        store.reserve_action(
            "run_a",
            "p1",
            "p1",
            "investigator",
            ActionKind.provider_attempt,
            {},
            Reservation(input_tokens=1001),
        )
    assert exhausted.value.limit_name == "input_tokens"
    with pytest.raises(BudgetExhausted) as exhausted:
        store.reserve_action(
            "run_a",
            "p1",
            "p1",
            "investigator",
            ActionKind.provider_attempt,
            {},
            Reservation(usd=0.2),
        )
    assert exhausted.value.limit_name == "estimated_usd"


def test_experiment_allowance_is_shared_across_runs(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_experiment("exp", 0.15, {})
    make_run(store, "run_a", "exp")
    make_run(store, "run_b", "exp")
    for run_id in ("run_a", "run_b"):
        store.reserve_action(
            run_id,
            f"{run_id}:p1",
            f"{run_id}:p1",
            "investigator",
            ActionKind.provider_attempt,
            {},
            Reservation(usd=0.07),
        )
    with pytest.raises(BudgetExhausted) as exhausted:
        store.reserve_action(
            "run_b",
            "run_b:p2",
            "run_b:p2",
            "investigator",
            ActionKind.provider_attempt,
            {},
            Reservation(usd=0.02),
        )
    assert exhausted.value.limit_name == "experiment_estimated_usd"


def test_active_time_ceiling(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_run(store)
    store.add_active_ms("run_a", 10_000)
    with pytest.raises(BudgetExhausted) as exhausted:
        reserve_tool(store, "late")
    assert exhausted.value.limit_name == "active_ms"


@pytest.mark.parametrize("cache_field", ["cache_read_input_tokens", "cache_creation_input_tokens"])
def test_cached_prompts_consume_the_input_allowance_after_reopening(
    tmp_path: Path, cache_field: str
) -> None:
    store = make_store(tmp_path)
    make_run(store)
    action = store.reserve_action(
        "run_a",
        "cached_attempt",
        "cached_attempt",
        "investigator",
        ActionKind.provider_attempt,
        {},
        Reservation(input_tokens=900, usd=0.03),
    )
    store.complete_action(
        action,
        ActionState.completed,
        5,
        usage={
            "usage_known": True,
            "input_tokens": 100,
            cache_field: 800,
            "output_tokens": 50,
            "usd": 0.01,
        },
    )
    store.close()
    store = make_store(tmp_path)
    with pytest.raises(BudgetExhausted) as exhausted:
        store.reserve_action(
            "run_a",
            "next_attempt",
            "next_attempt",
            "reviewer",
            ActionKind.provider_attempt,
            {},
            Reservation(input_tokens=101),
        )
    assert exhausted.value.limit_name == "input_tokens"
    assert exhausted.value.used == 900
    assert store.usage("run_a").estimated_usd == 0.01
    assert store.usage("run_a").provider_attempts == 1
    store.close()


def test_verification_queries_do_not_use_the_tool_allowance(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_run(store)
    for index in range(5):
        action = store.reserve_action(
            "run_a",
            f"v{index}",
            f"v{index}",
            "verifier",
            ActionKind.verification_query,
            {},
            Reservation(),
        )
        store.complete_action(action, ActionState.completed, 0, physical_query=True)
    usage = store.usage("run_a")
    assert usage.tool_requests == 0
    assert usage.verification_queries == 5


def test_findings_and_reviews_are_idempotent_per_action(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_run(store)
    first = store.save_findings("run_a", {"x": 1}, action_id=10)
    again = store.save_findings("run_a", {"x": 1}, action_id=10)
    second = store.save_findings("run_a", {"x": 2}, action_id=11)
    assert first == again
    assert second[1] == first[1] + 1
    review = store.save_review("run_a", {"decision": "accept"}, 1, first[0], action_id=12)
    assert review == store.save_review("run_a", {"decision": "accept"}, 1, first[0], action_id=12)


def hold_lock_then_exit(lock_directory: str, ready: object) -> None:
    lock = RunLock(Path(lock_directory), "run_x").acquire()
    ready.set()  # type: ignore[attr-defined]
    del lock
    # Exiting without release simulates a crash. The kernel drops the lock.


def test_run_lock_is_exclusive_and_released_by_a_dead_process(tmp_path: Path) -> None:
    with RunLock(tmp_path, "run_x"), pytest.raises(RunLockedError):
        RunLock(tmp_path, "run_x").acquire()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=hold_lock_then_exit, args=(str(tmp_path), ready))
    process.start()
    process.join(timeout=30)
    assert ready.is_set()
    RunLock(tmp_path, "run_x").acquire().release()


def test_reopening_an_experiment_retains_a_stricter_allowance(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_experiment("exp", 0.15, {})
    make_run(store, "run_a", "exp")
    store.reserve_action(
        "run_a",
        "provider_first",
        "provider_first",
        "investigator",
        ActionKind.provider_attempt,
        {},
        Reservation(usd=0.07),
    )
    store.create_experiment("exp", 0.08, {})
    store.create_experiment("exp", 0.15, {})
    with pytest.raises(BudgetExhausted) as exhausted:
        store.reserve_action(
            "run_a",
            "provider_second",
            "provider_second",
            "investigator",
            ActionKind.provider_attempt,
            {},
            Reservation(usd=0.02),
        )
    assert exhausted.value.limit_name == "experiment_estimated_usd"
    assert store.usage("run_a").provider_attempts == 1
    store.close()
