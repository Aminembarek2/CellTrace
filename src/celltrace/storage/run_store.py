"""SQLite run store: runs, the action ledger, evidence, messages and outputs.

The action ledger is the authority for budget usage. Every tool request and
provider attempt is inserted as a reservation inside an immediate transaction
before it executes, so a crash, retry or resume can never reset a counter.
Unknown provider usage stays charged at its reserved amount.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from celltrace.canonical import canonical_json, content_hash
from celltrace.clock import Clock

RUN_STORE_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    maximum_estimated_usd REAL NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    execution_mode TEXT NOT NULL,
    model_id TEXT,
    snapshot_hash TEXT NOT NULL,
    identity_hash TEXT NOT NULL,
    identity_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    limits_json TEXT NOT NULL,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    parent_run_id TEXT,
    status TEXT NOT NULL,
    workflow_outcome TEXT,
    terminal_reason TEXT,
    active_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_events (
    event_id TEXT PRIMARY KEY,
    snapshot_hash TEXT NOT NULL,
    detector_config_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    operation_key TEXT NOT NULL UNIQUE,
    logical_key TEXT NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    tool_name TEXT,
    attempt_number INTEGER NOT NULL,
    request_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    state TEXT NOT NULL,
    reserved_tool_units INTEGER NOT NULL DEFAULT 0,
    reserved_input_tokens INTEGER NOT NULL DEFAULT 0,
    reserved_output_tokens INTEGER NOT NULL DEFAULT 0,
    reserved_usd REAL NOT NULL DEFAULT 0,
    usage_known INTEGER NOT NULL DEFAULT 0,
    actual_input_tokens INTEGER,
    actual_output_tokens INTEGER,
    cache_read_input_tokens INTEGER,
    cache_creation_input_tokens INTEGER,
    actual_usd REAL,
    estimation_method TEXT,
    provider_request_id TEXT,
    response_json TEXT,
    evidence_id TEXT,
    error_code TEXT,
    error_json TEXT,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    physical_query INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS actions_by_run ON actions(run_id, kind);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    cache_key TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, cache_key)
);
CREATE TABLE IF NOT EXISTS agent_messages (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    role TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    message_json TEXT NOT NULL,
    action_id INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, role, sequence)
);
CREATE TABLE IF NOT EXISTS findings (
    findings_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    revision INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    action_id INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, revision)
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    round INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    independent_query_count INTEGER NOT NULL,
    findings_id TEXT NOT NULL,
    action_id INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, round)
);
CREATE TABLE IF NOT EXISTS verification_results (
    verification_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    findings_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_events (
    event_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class ActionKind(StrEnum):
    provider_attempt = "provider_attempt"
    tool_request = "tool_request"
    verification_query = "verification_query"


class ActionState(StrEnum):
    reserved = "reserved"
    started = "started"
    completed = "completed"
    failed = "failed"
    uncertain = "uncertain"


class BudgetExhausted(Exception):
    def __init__(self, limit_name: str, used: float, limit: float, requested: float):
        super().__init__(
            f"{limit_name} exhausted: used {used}, requested {requested}, limit {limit}"
        )
        self.limit_name = limit_name
        self.used = used
        self.limit = limit
        self.requested = requested


class RunNotFound(Exception):
    pass


@dataclass(frozen=True)
class RunLimits:
    maximum_tool_requests: int
    maximum_provider_attempts: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    maximum_estimated_usd: float | None
    maximum_active_ms: int
    maximum_review_rounds: int
    maximum_request_retries: int
    reviewer_tool_reserve: int = 0
    review_round_tool_reserve: int = 0
    review_round_provider_reserve: int = 0
    review_round_input_reserve: int = 0
    review_round_output_reserve: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.reviewer_tool_reserve, bool)
            or not isinstance(self.reviewer_tool_reserve, int)
            or self.reviewer_tool_reserve < 0
            or (
                self.reviewer_tool_reserve != 0
                and self.reviewer_tool_reserve >= self.maximum_tool_requests
            )
        ):
            raise ValueError("reviewer reserve must be nonnegative and smaller than total tools")
        for reserve, total in (
            (self.review_round_tool_reserve, self.maximum_tool_requests),
            (self.review_round_provider_reserve, self.maximum_provider_attempts),
            (self.review_round_input_reserve, self.maximum_input_tokens),
            (self.review_round_output_reserve, self.maximum_output_tokens),
        ):
            if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
                raise ValueError("review round reserves must be nonnegative integers")
            if reserve and reserve * self.maximum_review_rounds >= total:
                raise ValueError("review round protection must leave resources for investigation")


@dataclass(frozen=True)
class ResourceAllowance:
    """Charged totals and protected future work, in the resource's native units."""

    used: int
    total_remaining: int
    protected_future: int
    role_remaining: int


@dataclass(frozen=True)
class BudgetUsage:
    tool_requests: int
    tool_cache_hits: int
    physical_queries: int
    verification_queries: int
    provider_attempts: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    unknown_usage_attempts: int
    estimated_usd: float
    active_ms: int


@dataclass(frozen=True)
class Reservation:
    tool_units: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    estimation_method: str | None = None

    def __post_init__(self) -> None:
        for value in (self.tool_units, self.input_tokens, self.output_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("reservation counts must be nonnegative integers")
        if isinstance(self.usd, bool) or not math.isfinite(self.usd) or self.usd < 0:
            raise ValueError("reserved cost must be finite and nonnegative")


@dataclass(frozen=True)
class ActionRecord:
    action_id: int
    run_id: str
    operation_key: str
    logical_key: str
    role: str
    kind: ActionKind
    tool_name: str | None
    attempt_number: int
    state: ActionState
    request: dict[str, Any]
    response: dict[str, Any] | None
    evidence_id: str | None
    error_code: str | None
    error: dict[str, Any] | None
    cache_hit: bool


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def budget_usage(connection: sqlite3.Connection, run_id: str) -> BudgetUsage:
    """Charged usage for one run. Unknown provider usage counts at its reservation."""
    row = connection.execute(
        """SELECT
                COALESCE(SUM(CASE WHEN kind = 'tool_request' THEN reserved_tool_units END), 0),
                COALESCE(SUM(CASE WHEN kind = 'tool_request' THEN cache_hit END), 0),
                COALESCE(SUM(CASE WHEN kind = 'tool_request' THEN physical_query END), 0),
                COALESCE(SUM(CASE WHEN kind = 'verification_query' THEN 1 END), 0),
                COALESCE(SUM(CASE WHEN kind = 'provider_attempt' THEN 1 END), 0),
                COALESCE(SUM(CASE WHEN kind = 'provider_attempt' THEN
                    CASE WHEN usage_known = 1 THEN actual_input_tokens
                    ELSE reserved_input_tokens END END), 0),
                COALESCE(SUM(CASE WHEN kind = 'provider_attempt' THEN
                    CASE WHEN usage_known = 1 THEN actual_output_tokens
                    ELSE reserved_output_tokens END END), 0),
                COALESCE(SUM(cache_read_input_tokens), 0),
                COALESCE(SUM(cache_creation_input_tokens), 0),
                COALESCE(SUM(CASE WHEN kind = 'provider_attempt' AND usage_known = 0
                    THEN 1 END), 0),
                COALESCE(SUM(CASE WHEN usage_known = 1 THEN COALESCE(actual_usd, 0)
                    ELSE reserved_usd END), 0)
            FROM actions WHERE run_id = ?""",
        (run_id,),
    ).fetchone()
    active_ms = connection.execute(
        "SELECT active_ms FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return BudgetUsage(
        tool_requests=int(row[0]),
        tool_cache_hits=int(row[1]),
        physical_queries=int(row[2]),
        verification_queries=int(row[3]),
        provider_attempts=int(row[4]),
        input_tokens=int(row[5]),
        output_tokens=int(row[6]),
        cache_read_input_tokens=int(row[7]),
        cache_creation_input_tokens=int(row[8]),
        unknown_usage_attempts=int(row[9]),
        estimated_usd=float(row[10]),
        active_ms=int(active_ms[0]) if active_ms else 0,
    )


class RunStore:
    def __init__(self, path: Path, clock: Clock):
        self.path = path
        self.clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(_SCHEMA)
        self.connection.execute(
            "INSERT OR IGNORE INTO store_info(key, value) VALUES ('schema_version', ?)",
            (str(RUN_STORE_SCHEMA_VERSION),),
        )

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Immediate transaction so reservations are serialized against each other."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        self.connection.execute("COMMIT")

    # Experiments and runs

    def create_experiment(
        self, experiment_id: str, maximum_usd: float, payload: dict[str, Any]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO experiments VALUES (?, ?, ?, ?)",
                (experiment_id, maximum_usd, canonical_json(payload), self.clock.utc_now_iso()),
            )
            connection.execute(
                "UPDATE experiments SET maximum_estimated_usd = min(maximum_estimated_usd, ?) "
                "WHERE experiment_id = ?",
                (maximum_usd, experiment_id),
            )

    def create_run(
        self,
        run_id: str,
        case_id: str,
        event_id: str,
        arm: str,
        execution_mode: str,
        model_id: str | None,
        snapshot_hash: str,
        identity: dict[str, Any],
        context: dict[str, Any],
        limits: RunLimits,
        experiment_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        now = self.clock.utc_now_iso()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO runs (run_id, case_id, event_id, arm, execution_mode, model_id,
                snapshot_hash, identity_hash, identity_json, context_json, limits_json,
                experiment_id, parent_run_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
                (
                    run_id,
                    case_id,
                    event_id,
                    arm,
                    execution_mode,
                    model_id,
                    snapshot_hash,
                    content_hash(identity),
                    canonical_json(identity),
                    canonical_json(context),
                    canonical_json(limits.__dict__),
                    experiment_id,
                    parent_run_id,
                    now,
                    now,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        run = dict(row)
        for key in ("identity_json", "context_json", "limits_json"):
            run[key.removesuffix("_json")] = json.loads(run.pop(key))
        return run

    def run_limits(self, run_id: str) -> RunLimits:
        return RunLimits(**self.get_run(run_id)["limits"])

    def mark_terminal(self, run_id: str, outcome: str, reason: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE runs SET status = 'terminal', workflow_outcome = ?, terminal_reason = ?,
                updated_at = ? WHERE run_id = ?""",
                (outcome, reason, self.clock.utc_now_iso(), run_id),
            )

    def reopen_run(self, run_id: str, superseded_report: dict[str, Any] | None) -> None:
        """Operator authorized continuation of a run stopped by an uncertain provider call."""
        with self.transaction() as connection:
            connection.execute(
                """UPDATE runs SET status = 'running', workflow_outcome = NULL,
                terminal_reason = NULL, updated_at = ? WHERE run_id = ?""",
                (self.clock.utc_now_iso(), run_id),
            )
            connection.execute(
                "INSERT INTO run_events (run_id, event_type, payload_json, created_at)"
                " VALUES (?, 'run_reopened', ?, ?)",
                (
                    run_id,
                    canonical_json({"superseded_report": superseded_report}),
                    self.clock.utc_now_iso(),
                ),
            )

    def add_active_ms(self, run_id: str, milliseconds: int) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE runs SET active_ms = active_ms + ?, updated_at = ? WHERE run_id = ?",
                (max(milliseconds, 0), self.clock.utc_now_iso(), run_id),
            )

    def save_candidate(
        self, event_id: str, snapshot_hash: str, config_hash: str, payload: dict[str, Any]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO candidate_events VALUES (?, ?, ?, ?)",
                (event_id, snapshot_hash, config_hash, canonical_json(payload)),
            )

    def record_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO run_events (run_id, event_type, payload_json, created_at)"
                " VALUES (?, ?, ?, ?)",
                (run_id, event_type, canonical_json(payload), self.clock.utc_now_iso()),
            )

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT event_type, payload_json, created_at FROM run_events WHERE run_id = ?"
            " ORDER BY event_sequence",
            (run_id,),
        ).fetchall()
        return [
            {"event_type": row[0], "payload": json.loads(row[1]), "created_at": row[2]}
            for row in rows
        ]

    # Budget ledger

    def usage(self, run_id: str) -> BudgetUsage:
        return budget_usage(self.connection, run_id)

    def experiment_usd(self, experiment_id: str) -> tuple[float, float]:
        """Return (used or reserved USD across the experiment, experiment ceiling)."""
        ceiling = self.connection.execute(
            "SELECT maximum_estimated_usd FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        used = self.connection.execute(
            """SELECT COALESCE(SUM(CASE WHEN a.usage_known = 1 THEN COALESCE(a.actual_usd, 0)
            ELSE a.reserved_usd END), 0) FROM actions a JOIN runs r ON a.run_id = r.run_id
            WHERE r.experiment_id = ?""",
            (experiment_id,),
        ).fetchone()
        return float(used[0]), float(ceiling[0]) if ceiling else 0.0

    def tool_allowance(self, run_id: str, role: str) -> dict[str, int]:
        """Lifetime allowance, including every reserved/failed/cached/retried request."""
        total_used = self.usage(run_id).tool_requests
        role_used = int(
            self.connection.execute(
                "SELECT COALESCE(SUM(reserved_tool_units), 0) FROM actions "
                "WHERE run_id = ? AND kind = 'tool_request' AND role = ?",
                (run_id, role),
            ).fetchone()[0]
        )
        allowance = self.resource_allowance(run_id, role)["tool_requests"]
        total_left = allowance.total_remaining
        role_left = allowance.role_remaining
        return {
            "total_remaining": total_left,
            "role_remaining": role_left,
            "total_used": total_used,
            "role_used": role_used,
        }

    def resource_allowance(self, run_id: str, role: str) -> dict[str, ResourceAllowance]:
        """Derive round protection from saved reviews; reservation calls this in its transaction.

        Investigator protects every still-possible review. Reviewer can spend on its current
        round but protects subsequent rounds. No role/round change refunds charged usage.
        """
        limits, usage = self.run_limits(run_id), self.usage(run_id)
        reserves = {
            "tool_requests": limits.review_round_tool_reserve,
            "provider_attempts": limits.review_round_provider_reserve,
            "input_tokens": limits.review_round_input_reserve,
            "output_tokens": limits.review_round_output_reserve,
        }
        if (limits.reviewer_tool_reserve or any(reserves.values())) and role not in (
            "investigator",
            "reviewer",
        ):
            raise ValueError("a reserved resource pool accepts only investigator/reviewer roles")
        reviews_done = self.connection.execute(
            "SELECT COUNT(*) FROM reviews WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        future_rounds = max(
            0, limits.maximum_review_rounds - reviews_done - int(role == "reviewer")
        )
        values = {
            "tool_requests": (usage.tool_requests, limits.maximum_tool_requests),
            "provider_attempts": (usage.provider_attempts, limits.maximum_provider_attempts),
            "input_tokens": (
                usage.input_tokens
                + usage.cache_read_input_tokens
                + usage.cache_creation_input_tokens,
                limits.maximum_input_tokens,
            ),
            "output_tokens": (usage.output_tokens, limits.maximum_output_tokens),
        }
        result = {}
        for name, (used, maximum) in values.items():
            remaining = max(0, maximum - used)
            protected = reserves[name] * future_rounds
            available = max(0, remaining - protected)
            if name == "tool_requests" and limits.reviewer_tool_reserve and role == "investigator":
                role_used = self.connection.execute(
                    "SELECT COALESCE(SUM(reserved_tool_units),0) FROM actions "
                    "WHERE run_id=? AND kind='tool_request' AND role=?",
                    (run_id, role),
                ).fetchone()[0]
                available = min(
                    available, max(0, maximum - limits.reviewer_tool_reserve - role_used)
                )
            result[name] = ResourceAllowance(used, remaining, protected, available)
        return result

    def _check_limits(
        self, run_id: str, kind: ActionKind, reservation: Reservation, role: str
    ) -> None:
        limits = self.run_limits(run_id)
        used = self.usage(run_id)
        checks: list[tuple[str, float, float, float]] = []
        requested_resources = (
            {"tool_requests": reservation.tool_units}
            if kind == ActionKind.tool_request
            else {
                "provider_attempts": 1,
                "input_tokens": reservation.input_tokens,
                "output_tokens": reservation.output_tokens,
            }
            if kind == ActionKind.provider_attempt
            else {}
        )
        if requested_resources:
            for name, resource in self.resource_allowance(run_id, role).items():
                native_request = requested_resources.get(name, 0)
                if native_request > resource.role_remaining and resource.protected_future:
                    raise BudgetExhausted(
                        f"{name}:future_review",
                        resource.used,
                        resource.used + resource.role_remaining,
                        native_request,
                    )
        if kind == ActionKind.tool_request:
            if limits.reviewer_tool_reserve:
                allowance = self.tool_allowance(run_id, role)
                if reservation.tool_units > allowance["role_remaining"]:
                    name = (
                        "tool_requests:investigator"
                        if role == "investigator"
                        and (allowance["role_remaining"] < allowance["total_remaining"])
                        else "tool_requests"
                    )
                    raise BudgetExhausted(
                        name,
                        allowance["role_used"],
                        allowance["role_used"] + allowance["role_remaining"],
                        reservation.tool_units,
                    )
            checks.append(
                (
                    "tool_requests",
                    used.tool_requests,
                    limits.maximum_tool_requests,
                    reservation.tool_units,
                )
            )
        if kind == ActionKind.provider_attempt:
            checks += [
                ("provider_attempts", used.provider_attempts, limits.maximum_provider_attempts, 1),
                (
                    "input_tokens",
                    used.input_tokens
                    + used.cache_read_input_tokens
                    + used.cache_creation_input_tokens,
                    limits.maximum_input_tokens,
                    reservation.input_tokens,
                ),
                (
                    "output_tokens",
                    used.output_tokens,
                    limits.maximum_output_tokens,
                    reservation.output_tokens,
                ),
            ]
            if limits.maximum_estimated_usd is not None:
                checks.append(
                    (
                        "estimated_usd",
                        used.estimated_usd,
                        limits.maximum_estimated_usd,
                        reservation.usd,
                    )
                )
        if kind != ActionKind.verification_query:
            checks.append(("active_ms", used.active_ms, limits.maximum_active_ms, 0))
        for name, used_amount, limit, requested in checks:
            over_time = name == "active_ms" and used_amount >= limit
            if over_time or used_amount + requested > limit:
                raise BudgetExhausted(name, used_amount, limit, requested)
        run = self.get_run(run_id)
        if run["experiment_id"] and reservation.usd > 0:
            experiment_used, experiment_limit = self.experiment_usd(run["experiment_id"])
            if experiment_used + reservation.usd > experiment_limit:
                raise BudgetExhausted(
                    "experiment_estimated_usd", experiment_used, experiment_limit, reservation.usd
                )

    def reserve_action(
        self,
        run_id: str,
        operation_key: str,
        logical_key: str,
        role: str,
        kind: ActionKind,
        request: dict[str, Any],
        reservation: Reservation,
        attempt_number: int = 1,
        tool_name: str | None = None,
    ) -> int:
        """Charge the budget and insert a reservation, or raise BudgetExhausted."""
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT action_id FROM actions WHERE operation_key = ?", (operation_key,)
            ).fetchone()
            if existing is not None:
                raise ValueError(f"operation {operation_key} already reserved")
            self._check_limits(run_id, kind, reservation, role)
            cursor = connection.execute(
                """INSERT INTO actions (run_id, operation_key, logical_key, role, kind, tool_name,
                attempt_number, request_hash, request_json, state, reserved_tool_units,
                reserved_input_tokens, reserved_output_tokens, reserved_usd, estimation_method,
                created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    operation_key,
                    logical_key,
                    role,
                    kind.value,
                    tool_name,
                    attempt_number,
                    content_hash(request),
                    canonical_json(request),
                    reservation.tool_units,
                    reservation.input_tokens,
                    reservation.output_tokens,
                    reservation.usd,
                    reservation.estimation_method,
                    self.clock.utc_now_iso(),
                ),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def mark_started(self, action_id: int) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE actions SET state = 'started', started_at = ? WHERE action_id = ?",
                (self.clock.utc_now_iso(), action_id),
            )

    def complete_action(
        self,
        action_id: int,
        state: ActionState,
        duration_ms: int,
        response: dict[str, Any] | None = None,
        evidence_id: str | None = None,
        error_code: str | None = None,
        error: dict[str, Any] | None = None,
        cache_hit: bool = False,
        physical_query: bool = False,
        usage: dict[str, Any] | None = None,
        provider_request_id: str | None = None,
        message: tuple[str, str, int, dict[str, Any]] | None = None,
    ) -> None:
        """Finish an action. An optional transcript message is written in the same transaction."""
        usage = usage or {}
        with self.transaction() as connection:
            connection.execute(
                """UPDATE actions SET state = ?, completed_at = ?, duration_ms = ?,
                response_json = ?, evidence_id = ?, error_code = ?, error_json = ?, cache_hit = ?,
                physical_query = ?, usage_known = ?, actual_input_tokens = ?,
                actual_output_tokens = ?, cache_read_input_tokens = ?,
                cache_creation_input_tokens = ?, actual_usd = ?, provider_request_id = ?
                WHERE action_id = ?""",
                (
                    state.value,
                    self.clock.utc_now_iso(),
                    duration_ms,
                    None if response is None else canonical_json(response),
                    evidence_id,
                    error_code,
                    None if error is None else canonical_json(error),
                    int(cache_hit),
                    int(physical_query),
                    int(bool(usage.get("usage_known", False))),
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("cache_read_input_tokens"),
                    usage.get("cache_creation_input_tokens"),
                    usage.get("usd"),
                    provider_request_id,
                    action_id,
                ),
            )
            if message is not None:
                run_id, role, sequence, payload = message
                connection.execute(
                    "INSERT INTO agent_messages VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        role,
                        sequence,
                        canonical_json(payload),
                        action_id,
                        self.clock.utc_now_iso(),
                    ),
                )

    def mark_uncertain(self, action_id: int, reason: str, duration_ms: int | None = None) -> None:
        """Keep the reservation. A monotonic duration far below the wall clock interval
        shows the host was suspended during the call."""
        with self.transaction() as connection:
            connection.execute(
                """UPDATE actions SET state = 'uncertain', error_code = 'UNCERTAIN_OUTCOME',
                error_json = ?, completed_at = ?, duration_ms = ? WHERE action_id = ?""",
                (
                    canonical_json({"reason": reason}),
                    self.clock.utc_now_iso(),
                    duration_ms,
                    action_id,
                ),
            )

    def action_by_key(self, operation_key: str) -> ActionRecord | None:
        row = self.connection.execute(
            "SELECT * FROM actions WHERE operation_key = ?", (operation_key,)
        ).fetchone()
        return None if row is None else self._action(row)

    def actions(self, run_id: str, kind: ActionKind | None = None) -> list[ActionRecord]:
        query = "SELECT * FROM actions WHERE run_id = ?"
        parameters: tuple[Any, ...] = (run_id,)
        if kind is not None:
            query += " AND kind = ?"
            parameters += (kind.value,)
        rows = self.connection.execute(query + " ORDER BY action_id", parameters).fetchall()
        return [self._action(row) for row in rows]

    def raw_actions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM actions WHERE run_id = ? ORDER BY action_id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _action(row: sqlite3.Row) -> ActionRecord:
        return ActionRecord(
            action_id=row["action_id"],
            run_id=row["run_id"],
            operation_key=row["operation_key"],
            logical_key=row["logical_key"],
            role=row["role"],
            kind=ActionKind(row["kind"]),
            tool_name=row["tool_name"],
            attempt_number=row["attempt_number"],
            state=ActionState(row["state"]),
            request=json.loads(row["request_json"]),
            response=_loads(row["response_json"]),
            evidence_id=row["evidence_id"],
            error_code=row["error_code"],
            error=_loads(row["error_json"]),
            cache_hit=bool(row["cache_hit"]),
        )

    # Evidence

    def evidence_by_cache_key(self, run_id: str, cache_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT record_json FROM evidence WHERE run_id = ? AND cache_key = ?",
            (run_id, cache_key),
        ).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def save_evidence(self, record: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["evidence_id"],
                    record["run_id"],
                    record["cache_key"],
                    record["content_sha256"],
                    canonical_json(record),
                    self.clock.utc_now_iso(),
                ),
            )

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT record_json FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def run_evidence(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT record_json FROM evidence WHERE run_id = ? ORDER BY created_at, evidence_id",
            (run_id,),
        ).fetchall()
        return [dict(json.loads(row[0])) for row in rows]

    # Transcripts and outputs

    def messages(self, run_id: str, role: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT message_json FROM agent_messages WHERE run_id = ? AND role = ?"
            " ORDER BY sequence",
            (run_id, role),
        ).fetchall()
        return [dict(json.loads(row[0])) for row in rows]

    def append_message(
        self, run_id: str, role: str, payload: dict[str, Any], action_id: int | None = None
    ) -> int:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM agent_messages"
                " WHERE run_id = ? AND role = ?",
                (run_id, role),
            ).fetchone()
            sequence = int(row[0])
            connection.execute(
                "INSERT INTO agent_messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    role,
                    sequence,
                    canonical_json(payload),
                    action_id,
                    self.clock.utc_now_iso(),
                ),
            )
            return sequence

    def save_findings(
        self, run_id: str, payload: dict[str, Any], action_id: int
    ) -> tuple[str, int]:
        """Store a submission once per provider action. Returns (findings_id, revision)."""
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT findings_id, revision FROM findings WHERE run_id = ? AND action_id = ?",
                (run_id, action_id),
            ).fetchone()
            if existing is not None:
                return str(existing[0]), int(existing[1])
            revision = int(
                connection.execute(
                    "SELECT COUNT(*) FROM findings WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            )
            findings_id = f"fnd_{content_hash([run_id, revision])[:12]}"
            connection.execute(
                "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?)",
                (
                    findings_id,
                    run_id,
                    revision,
                    canonical_json(payload),
                    action_id,
                    self.clock.utc_now_iso(),
                ),
            )
            return findings_id, revision

    def findings(self, run_id: str) -> list[tuple[str, int, dict[str, Any]]]:
        rows = self.connection.execute(
            "SELECT findings_id, revision, payload_json FROM findings WHERE run_id = ?"
            " ORDER BY revision",
            (run_id,),
        ).fetchall()
        return [(row[0], row[1], json.loads(row[2])) for row in rows]

    def save_review(
        self,
        run_id: str,
        payload: dict[str, Any],
        independent_query_count: int,
        findings_id: str,
        action_id: int,
    ) -> tuple[str, int]:
        """Store a review once per provider action. Returns (review_id, round)."""
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT review_id, round FROM reviews WHERE run_id = ? AND action_id = ?",
                (run_id, action_id),
            ).fetchone()
            if existing is not None:
                return str(existing[0]), int(existing[1])
            review_round = 1 + int(
                connection.execute(
                    "SELECT COUNT(*) FROM reviews WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            )
            review_id = f"rev_{content_hash([run_id, review_round])[:12]}"
            connection.execute(
                "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    review_id,
                    run_id,
                    review_round,
                    canonical_json(payload),
                    independent_query_count,
                    findings_id,
                    action_id,
                    self.clock.utc_now_iso(),
                ),
            )
            return review_id, review_round

    def reviews(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT round, payload_json, independent_query_count, findings_id FROM reviews"
            " WHERE run_id = ? ORDER BY round",
            (run_id,),
        ).fetchall()
        return [
            {
                "round": row[0],
                "review": json.loads(row[1]),
                "independent_query_count": row[2],
                "findings_id": row[3],
            }
            for row in rows
        ]

    def save_verification(
        self, run_id: str, findings_id: str, passed: bool, payload: dict[str, Any]
    ) -> str:
        verification_id = f"ver_{content_hash([run_id, findings_id])[:12]}"
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO verification_results VALUES (?, ?, ?, ?, ?, ?)",
                (
                    verification_id,
                    run_id,
                    findings_id,
                    int(passed),
                    canonical_json(payload),
                    self.clock.utc_now_iso(),
                ),
            )
        return verification_id

    def verifications(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT findings_id, passed, payload_json FROM verification_results"
            " WHERE run_id = ? ORDER BY created_at, verification_id",
            (run_id,),
        ).fetchall()
        return [
            {"findings_id": row[0], "passed": bool(row[1]), "result": json.loads(row[2])}
            for row in rows
        ]

    def save_report(self, run_id: str, payload: dict[str, Any]) -> str:
        report_id = f"rpt_{content_hash([run_id])[:12]}"
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?)",
                (report_id, run_id, canonical_json(payload), self.clock.utc_now_iso()),
            )
        return report_id

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json FROM reports WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def list_runs(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id is None:
            rows = self.connection.execute("SELECT run_id FROM runs ORDER BY created_at").fetchall()
        else:
            rows = self.connection.execute(
                "SELECT run_id FROM runs WHERE experiment_id = ? ORDER BY created_at",
                (experiment_id,),
            ).fetchall()
        return [self.get_run(row[0]) for row in rows]
