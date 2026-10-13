"""Export stored runs as OpenTelemetry traces in the OTLP JSON encoding.

Span names and gen_ai attributes follow the OpenTelemetry GenAI semantic conventions, which are
still marked Development, so names may change. A run becomes an invoke_workflow span. Each
uninterrupted stretch of one agent's actions becomes an invoke_agent span holding its chat and
execute_tool spans. Code verification has no GenAI convention and keeps celltrace attributes.
Identifiers are derived from run and action identifiers, so exporting twice gives equal traces.
Request and response bodies are not exported, only tool arguments, which are typed parameters.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from celltrace.reporting.run_viewer import read_only

SCOPE_NAME = "celltrace.reporting.otel_export"
SCOPE_VERSION = "1"
SEMANTIC_CONVENTIONS = "OpenTelemetry GenAI semantic conventions, Development status"
SPAN_KIND_INTERNAL = 1
SPAN_KIND_CLIENT = 3
STATUS_OK = 1
STATUS_ERROR = 2
STATUS_UNSET = 0


def _identifier(text: str, length: int) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _nanoseconds(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp).timestamp() * 1_000_000_000)


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    elif isinstance(value, list):
        encoded = {"arrayValue": {"values": [{"stringValue": str(item)} for item in value]}}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _span(
    trace_id: str,
    span_id: str,
    parent_id: str | None,
    name: str,
    kind: int,
    start: int,
    end: int,
    attributes: dict[str, Any],
    error: str | None = None,
    completed: bool = False,
) -> dict[str, Any]:
    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(max(start, end)),
        "attributes": [
            _attribute(key, value) for key, value in attributes.items() if value is not None
        ],
        "status": {"code": STATUS_ERROR, "message": error}
        if error
        else {"code": STATUS_OK if completed else STATUS_UNSET},
    }
    if parent_id is not None:
        span["parentSpanId"] = parent_id
    return span


def _action_window(row: sqlite3.Row) -> tuple[int, int]:
    start = _nanoseconds(row["started_at"] or row["created_at"])
    if row["completed_at"]:
        return start, _nanoseconds(row["completed_at"])
    return start, start + int(row["duration_ms"] or 0) * 1_000_000


def _segments(rows: list[sqlite3.Row]) -> list[tuple[str, list[sqlite3.Row]]]:
    """Consecutive actions of one role. A reviewer handoff or a revision starts a new segment."""
    segments: list[tuple[str, list[sqlite3.Row]]] = []
    for row in rows:
        if segments and segments[-1][0] == row["role"]:
            segments[-1][1].append(row)
        else:
            segments.append((row["role"], [row]))
    return segments


def run_spans(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    run = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(run_id)
    identity = json.loads(run["identity_json"])
    provider = (identity.get("provider") or {}).get("provider")
    model = run["model_id"]
    trace_id = _identifier(f"trace:{run_id}", 32)
    root_id = _identifier(f"span:{run_id}", 16)
    rows = connection.execute(
        "SELECT * FROM actions WHERE run_id = ? ORDER BY action_id", (run_id,)
    ).fetchall()
    run_start = _nanoseconds(run["created_at"])
    run_end = max([_nanoseconds(run["updated_at"]), *(_action_window(row)[1] for row in rows)])
    terminal_error = None
    if run["workflow_outcome"] not in (None, "completed"):
        terminal_error = run["terminal_reason"] or run["workflow_outcome"]
    spans = [
        _span(
            trace_id,
            root_id,
            None,
            f"invoke_workflow {run['arm']}",
            SPAN_KIND_INTERNAL,
            run_start,
            run_end,
            {
                "gen_ai.operation.name": "invoke_workflow",
                "gen_ai.workflow.name": run["arm"],
                "gen_ai.conversation.id": run_id,
                "gen_ai.request.model": model,
                "gen_ai.provider.name": provider,
                "celltrace.run_id": run_id,
                "celltrace.case_id": run["case_id"],
                "celltrace.event_id": run["event_id"],
                "celltrace.execution_mode": run["execution_mode"],
                "celltrace.workflow_outcome": run["workflow_outcome"],
                "celltrace.run_status": run["status"],
                "celltrace.export_incomplete": run["status"] != "terminal",
                "celltrace.terminal_reason": run["terminal_reason"],
                "error.type": terminal_error,
            },
            terminal_error,
            completed=run["status"] == "terminal" and run["workflow_outcome"] == "completed",
        )
    ]
    for index, (role, members) in enumerate(_segments(rows)):
        segment_id = _identifier(f"segment:{run_id}:{index}", 16)
        windows = [_action_window(row) for row in members]
        start = min(window[0] for window in windows)
        end = max(window[1] for window in windows)
        if role == "verifier":
            spans.append(
                _span(
                    trace_id,
                    segment_id,
                    root_id,
                    "verify_claims",
                    SPAN_KIND_INTERNAL,
                    start,
                    end,
                    {"celltrace.verification_queries": len(members)},
                )
            )
            continue
        spans.append(
            _span(
                trace_id,
                segment_id,
                root_id,
                f"invoke_agent {role}",
                SPAN_KIND_INTERNAL,
                start,
                end,
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": role,
                    "gen_ai.conversation.id": run_id,
                    "gen_ai.request.model": model,
                    "gen_ai.provider.name": provider,
                    "celltrace.export_incomplete": any(
                        member["state"] in ("reserved", "started") for member in members
                    ),
                },
            )
        )
        for row, (action_start, action_end) in zip(members, windows, strict=True):
            spans.append(_action_span(row, trace_id, segment_id, action_start, action_end, model))
    return spans


def _action_span(
    row: sqlite3.Row, trace_id: str, parent_id: str, start: int, end: int, model: str | None
) -> dict[str, Any]:
    span_id = _identifier(f"action:{row['run_id']}:{row['action_id']}", 16)
    error = (row["error_code"] or row["state"]) if row["state"] in ("failed", "uncertain") else None
    common = {
        "celltrace.action_id": int(row["action_id"]),
        "celltrace.attempt_number": int(row["attempt_number"]),
        "celltrace.state": row["state"],
        "celltrace.export_incomplete": row["state"] in ("reserved", "started"),
        "error.type": error,
    }
    if row["kind"] == "provider_attempt":
        response = json.loads(row["response_json"]) if row["response_json"] else {}
        known = bool(row["usage_known"])
        return _span(
            trace_id,
            span_id,
            parent_id,
            f"chat {model}",
            SPAN_KIND_CLIENT,
            start,
            end,
            {
                **common,
                "gen_ai.operation.name": "chat",
                "gen_ai.agent.name": row["role"],
                "gen_ai.request.model": model,
                "gen_ai.response.model": response.get("reported_model"),
                "gen_ai.response.id": row["provider_request_id"],
                "gen_ai.response.finish_reasons": [response["stop_reason"]]
                if response.get("stop_reason")
                else None,
                "gen_ai.usage.input_tokens": row["actual_input_tokens"] if known else None,
                "gen_ai.usage.output_tokens": row["actual_output_tokens"] if known else None,
                "gen_ai.usage.cache_read.input_tokens": row["cache_read_input_tokens"],
                "celltrace.usage_known": known,
                "celltrace.usd": row["actual_usd"] if known else row["reserved_usd"],
            },
            error,
            completed=row["state"] == "completed",
        )
    if row["kind"] == "tool_request":
        request = json.loads(row["request_json"])
        return _span(
            trace_id,
            span_id,
            parent_id,
            f"execute_tool {row['tool_name']}",
            SPAN_KIND_INTERNAL,
            start,
            end,
            {
                **common,
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": row["tool_name"],
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call.arguments": json.dumps(
                    request.get("arguments", {}), sort_keys=True
                ),
                "celltrace.evidence_id": row["evidence_id"],
                "celltrace.cache_hit": bool(row["cache_hit"]),
            },
            error,
            completed=row["state"] == "completed",
        )
    return _span(
        trace_id,
        span_id,
        parent_id,
        row["kind"],
        SPAN_KIND_INTERNAL,
        start,
        end,
        common,
        error,
        completed=row["state"] == "completed",
    )


def export_traces(store_path: Path, output_path: Path) -> dict[str, Any]:
    with read_only(store_path) as connection:
        run_ids = [
            row["run_id"]
            for row in connection.execute("SELECT run_id FROM runs ORDER BY created_at, run_id")
        ]
        spans = [span for run_id in run_ids for span in run_spans(connection, run_id)]
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attribute("service.name", "celltrace")]},
                "scopeSpans": [
                    {"scope": {"name": SCOPE_NAME, "version": SCOPE_VERSION}, "spans": spans}
                ],
            }
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "traces": str(output_path),
        "runs": len(run_ids),
        "spans": len(spans),
        "semantic_conventions": SEMANTIC_CONVENTIONS,
    }
