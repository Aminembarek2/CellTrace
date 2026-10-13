"""Replay saved judge results against the durable ledger before anyone uses them.

A saved result counts only when its identity, status and cost equal the ledger, and an accepted
assessment equals the arguments of a recorded provider response. Edited files are refused.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from celltrace.canonical import canonical_json, content_hash
from celltrace.evaluation.review.judge import ASSESSMENT_METHOD
from celltrace.evaluation.review.plan import experiment_id, run_id
from celltrace.evaluation.review.response_contract import decode_assessment
from celltrace.evaluation.review.rubric import TOOL_NAME, validate_assessment

LEDGER_FILE = "runs.sqlite3"


def _read_only(output: Path) -> sqlite3.Connection:
    return sqlite3.connect((output / LEDGER_FILE).resolve().as_uri() + "?mode=ro", uri=True)


def _matches_response(
    response: dict[str, Any], assessment: dict[str, Any], item: dict[str, Any]
) -> bool:
    calls = response.get("tool_calls", [])
    complete = bool(
        response.get("usage", {}).get("usage_known", False)
        and not response.get("effective_settings", {}).get("settings_errors")
        and response.get("stop_reason") == "tool_calls"
        and len(calls) == 1
        and calls[0]["name"] == TOOL_NAME
    )
    if not complete:
        return False
    try:
        decoded = decode_assessment(calls[0]["arguments"], item)
    except (ValueError, KeyError, TypeError, IndexError):
        return False
    return content_hash(decoded.model_dump(mode="json")) == content_hash(assessment)


def replay_result(
    output: Path, plan_hash: str, item_id: str, model_id: str | None = None
) -> dict[str, Any] | None:
    path = output / "assessments" / f"{item_id}.json"
    if not path.exists():
        return None
    result: dict[str, Any] = json.loads(path.read_text())
    canonical_json(result)
    expected_run = run_id(plan_hash, item_id)
    if result["run_id"] != expected_run or result["item_id"] != item_id:
        raise ValueError(f"saved result identity changed: {item_id}")
    connection = _read_only(output)
    try:
        connection.execute("BEGIN")
        attempts = connection.execute(
            "SELECT usage_known, actual_usd, reserved_usd, response_json FROM actions "
            "WHERE run_id = ? AND kind = 'provider_attempt' ORDER BY action_id",
            (expected_run,),
        ).fetchall()
        terminal = connection.execute(
            "SELECT terminal_reason, model_id, identity_json, experiment_id FROM runs "
            "WHERE run_id = ?",
            (expected_run,),
        ).fetchone()
    finally:
        connection.close()
    if terminal is None or terminal[0] != result["status"]:
        raise ValueError(f"saved status differs from the ledger: {item_id}")
    if result["model_id"] != terminal[1] or (model_id is not None and model_id != terminal[1]):
        raise ValueError(f"saved model differs from the ledger or plan: {item_id}")
    if json.loads(terminal[2]).get("plan_hash") != plan_hash or terminal[3] != experiment_id(
        plan_hash
    ):
        raise ValueError(f"saved run belongs to another review plan: {item_id}")
    if result["assessment_method"] != ASSESSMENT_METHOD:
        raise ValueError(f"saved assessment method changed: {item_id}")
    cost = sum((actual or 0.0) if known else reserved for known, actual, reserved, _ in attempts)
    if abs(cost - result["usage"]["estimated_usd"]) > 1e-10:
        raise ValueError(f"saved cost differs from the ledger: {item_id}")
    assessment = result["assessment"]
    if assessment is not None:
        item = json.loads((output / "inputs" / f"{item_id}.json").read_text())
        if result["status"] != "assessed" or not any(
            response is not None and _matches_response(json.loads(response), assessment, item)
            for *_, response in attempts
        ):
            raise ValueError(f"saved assessment differs from its provider response: {item_id}")
    elif result["status"] == "assessed":
        raise ValueError(f"an assessed result cannot omit its assessment: {item_id}")
    return result


def load_results(output: Path, plan: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Every scheduled item mapped to its replayed result, or None when not yet judged."""
    results: dict[str, dict[str, Any] | None] = {}
    for entry in plan["schedule"]:
        identifier = entry["item_id"]
        result = replay_result(output, plan["plan_hash"], identifier, entry["model_id"])
        if result is not None and result["assessment"] is not None:
            item = json.loads((output / "inputs" / f"{identifier}.json").read_text())
            validate_assessment(result["assessment"], item)
        results[identifier] = result
    return results


def experiment_usage(output: Path, plan: dict[str, Any]) -> float:
    """Known or reserved USD charged to this plan's experiment, including failed attempts."""
    if not (output / LEDGER_FILE).exists():
        return 0.0
    identifier = experiment_id(plan["plan_hash"])
    connection = _read_only(output)
    try:
        connection.execute("BEGIN")
        row = connection.execute(
            "SELECT maximum_estimated_usd, payload_json FROM experiments WHERE experiment_id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            if connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0]:
                raise ValueError("ledger contains charges for another review plan")
            return 0.0
        if content_hash(json.loads(row[1])) != content_hash(plan):
            raise ValueError("ledger experiment identity differs from the frozen plan")
        if row[0] > plan["settings"]["maximum_estimated_usd"]:
            raise ValueError("ledger ceiling exceeds the frozen plan")
        used = connection.execute(
            "SELECT COALESCE(SUM(CASE WHEN a.usage_known = 1 THEN COALESCE(a.actual_usd, 0) "
            "ELSE a.reserved_usd END), 0) FROM actions a JOIN runs r ON a.run_id = r.run_id "
            "WHERE r.experiment_id = ?",
            (identifier,),
        ).fetchone()[0]
        return float(used)
    finally:
        connection.close()
