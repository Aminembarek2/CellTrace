"""Bounded execution of a frozen review plan.

The provider arrives as a factory, so the same loop runs against OpenRouter in a live run and
against a scripted judge in tests. One exclusive lock prevents two runners on one ledger. A STOP
file or a batch limit ends the loop between items, never inside a paid call, and running again
resumes the same ledger without resetting any charge.
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from celltrace.clock import Clock
from celltrace.evaluation.review.gates import dispatch_block
from celltrace.evaluation.review.judge import JudgeCall, judge_item
from celltrace.evaluation.review.ledger import LEDGER_FILE, experiment_usage, load_results
from celltrace.evaluation.review.persistence import write_json_atomic
from celltrace.evaluation.review.plan import experiment_id, load_plan, run_id, settings_from_plan
from celltrace.providers.base import ModelAdapter
from celltrace.providers.rate_card import RateCard
from celltrace.storage.run_store import RunStore

AdapterFactory = Callable[[str], ModelAdapter]
# Statuses after which continuing would spend money without learning anything new.
STOPPING_STATUSES = frozenset(
    {"budget_prevented", "provider_failed", "provider_uncertain", "unverified_provider_response"}
)
STOP_FILE = "STOP"
PROGRESS_FILE = "progress.json"


def run_review(
    output: Path,
    adapter_factory: AdapterFactory,
    clock: Clock,
    limit: int | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("batch limit must be positive")
    with (output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = load_plan(output)
        settings = settings_from_plan(plan)
        rates = {
            judge["model_id"]: RateCard.model_validate(judge["rate_card"])
            for judge in plan["judges"]
        }
        adapters: dict[str, ModelAdapter] = {}
        results = load_results(output, plan)
        store = RunStore(output / LEDGER_FILE, clock)
        try:
            store.create_experiment(
                experiment_id(plan["plan_hash"]), settings.maximum_estimated_usd, plan
            )
            dispatched = 0
            stopped_by = None
            for entry in plan["schedule"]:
                if (output / STOP_FILE).exists():
                    stopped_by = "stop_file"
                    break
                if limit is not None and dispatched >= limit:
                    stopped_by = "batch_limit"
                    break
                identifier = entry["item_id"]
                if results[identifier] is not None or dispatch_block(plan, results, entry):
                    continue
                model_id = entry["model_id"]
                if model_id not in adapters:
                    adapters[model_id] = adapter_factory(model_id)
                call = JudgeCall(
                    run_id=run_id(plan["plan_hash"], identifier),
                    experiment_id=experiment_id(plan["plan_hash"]),
                    model_id=model_id,
                    item=json.loads((output / "inputs" / f"{identifier}.json").read_text()),
                    input_hash=plan["input_hashes"][identifier],
                    plan_hash=plan["plan_hash"],
                    limits=settings.call_limits,
                )
                result = judge_item(
                    store,
                    adapters[model_id],
                    rates[model_id],
                    call,
                    output / "assessments" / f"{identifier}.json",
                )
                results[identifier] = result
                dispatched += 1
                if emit is not None:
                    emit(
                        {
                            "item_id": identifier,
                            "model_id": model_id,
                            "stratum": entry["stratum"],
                            "status": result["status"],
                            "usd": result["usage"]["estimated_usd"],
                        }
                    )
                if result["status"] in STOPPING_STATUSES:
                    stopped_by = result["status"]
                    break
        finally:
            store.close()
    blocked: dict[str, int] = {}
    for entry in plan["schedule"]:
        if results[entry["item_id"]] is None:
            reason = dispatch_block(plan, results, entry) or "not_reached"
            blocked[reason] = blocked.get(reason, 0) + 1
    progress = {
        "plan_hash": plan["plan_hash"],
        "scheduled_items": len(plan["schedule"]),
        "recorded_items": sum(result is not None for result in results.values()),
        "dispatched_this_run": dispatched,
        "stopped_by": stopped_by,
        "not_dispatched": blocked,
        "known_or_reserved_usd": experiment_usage(output, plan),
        "maximum_estimated_usd": settings.maximum_estimated_usd,
    }
    write_json_atomic(output / PROGRESS_FILE, progress)
    return progress
