"""Read only views of a run store for debugging and monitoring agent runs.

The same page is either written once with its data embedded, or served on the loopback
interface while runs are in progress. Every read opens the store in SQLite read only mode, so
the viewer can never change the ledger it displays. Model and dataset text stays data: the page
inserts it as text, never as markup.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from celltrace.storage.run_store import budget_usage

VIEWER_VERSION = "run_viewer_v1"
PREVIEW_CHARACTERS = 600
SUMMARY_CHARACTERS = 1500
LOOPBACK_HOST = "127.0.0.1"
CHECK_FIELDS = ("claim_id", "code", "message", "expected", "actual")
BUDGET_LIMITS = (
    ("tool_requests", "maximum_tool_requests", "Tool requests"),
    ("provider_attempts", "maximum_provider_attempts", "Provider attempts"),
    ("input_tokens", "maximum_input_tokens", "Input tokens"),
    ("output_tokens", "maximum_output_tokens", "Output tokens"),
    ("estimated_usd", "maximum_estimated_usd", "Known or reserved USD"),
    ("active_ms", "maximum_active_ms", "Active milliseconds"),
)


@contextmanager
def read_only(store_path: Path) -> Iterator[sqlite3.Connection]:
    if not store_path.is_file():
        raise FileNotFoundError(f"no run store at {store_path}")
    connection = sqlite3.connect(store_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        # A page/export uses several SELECTs. Pin them to one committed WAL snapshot;
        # mode=ro alone does not keep later SELECTs from seeing a newer writer commit.
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _preview(value: Any, limit: int = PREVIEW_CHARACTERS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


def _milliseconds(timestamp: str | None) -> float | None:
    return None if timestamp is None else datetime.fromisoformat(timestamp).timestamp() * 1000.0


def overview(connection: sqlite3.Connection) -> dict[str, Any]:
    """Run summaries plus failure and tool usage tables across the whole store."""
    rows = connection.execute(
        """SELECT r.run_id, r.case_id, r.event_id, r.arm, r.execution_mode, r.model_id,
                  r.experiment_id, r.status, r.workflow_outcome, r.terminal_reason, r.active_ms,
                  r.created_at, r.updated_at,
                  COALESCE(a.provider_attempts, 0) AS provider_attempts,
                  COALESCE(a.tool_requests, 0) AS tool_requests,
                  COALESCE(a.failed_actions, 0) AS failed_actions,
                  COALESCE(a.uncertain_actions, 0) AS uncertain_actions,
                  COALESCE(a.active_actions, 0) AS active_actions,
                  COALESCE(a.unknown_usage_attempts, 0) AS unknown_usage_attempts,
                  COALESCE(a.input_tokens, 0) AS input_tokens,
                  COALESCE(a.output_tokens, 0) AS output_tokens,
                  COALESCE(a.estimated_usd, 0) AS estimated_usd,
                  (SELECT COUNT(*) FROM findings f WHERE f.run_id = r.run_id) AS findings_revisions,
                  (SELECT COUNT(*) FROM reviews v WHERE v.run_id = r.run_id) AS review_rounds,
                  (SELECT COUNT(*) FROM verification_results c WHERE c.run_id = r.run_id)
                      AS verification_rounds,
                  (SELECT COUNT(*) FROM verification_results c
                      WHERE c.run_id = r.run_id AND c.passed = 0) AS failed_verifications,
                  json_extract(p.payload_json, '$.fact_check_status') AS fact_check_status,
                  json_extract(p.payload_json, '$.review_status') AS review_status,
                  json_extract(p.payload_json, '$.cause_status') AS cause_status,
                  json_array_length(p.payload_json, '$.verified_claims') AS verified_claims,
                  json_array_length(p.payload_json, '$.rejected_claims') AS rejected_claims
           FROM runs r
           LEFT JOIN (
               SELECT run_id,
                      SUM(kind = 'provider_attempt') AS provider_attempts,
                      SUM(CASE WHEN kind = 'tool_request' THEN reserved_tool_units ELSE 0 END)
                          AS tool_requests,
                      SUM(state = 'failed') AS failed_actions,
                      SUM(state = 'uncertain') AS uncertain_actions,
                      SUM(state IN ('reserved', 'started')) AS active_actions,
                      SUM(kind = 'provider_attempt' AND usage_known = 0)
                          AS unknown_usage_attempts,
                      SUM(CASE WHEN kind = 'provider_attempt' THEN CASE WHEN usage_known = 1
                          THEN actual_input_tokens ELSE reserved_input_tokens END END)
                          AS input_tokens,
                      SUM(CASE WHEN kind = 'provider_attempt' THEN CASE WHEN usage_known = 1
                          THEN actual_output_tokens ELSE reserved_output_tokens END END)
                          AS output_tokens,
                      SUM(CASE WHEN usage_known = 1 THEN COALESCE(actual_usd, 0)
                          ELSE reserved_usd END) AS estimated_usd
               FROM actions GROUP BY run_id
           ) a ON a.run_id = r.run_id
           LEFT JOIN reports p ON p.run_id = r.run_id
           ORDER BY r.created_at, r.run_id"""
    ).fetchall()
    failures = connection.execute(
        """SELECT role, kind, COALESCE(tool_name, '') AS tool_name, state,
                  COALESCE(error_code, '') AS error_code, COUNT(*) AS count,
                  COUNT(DISTINCT run_id) AS runs
           FROM actions WHERE state NOT IN ('completed')
           GROUP BY role, kind, tool_name, state, error_code
           ORDER BY count DESC, role, tool_name LIMIT 30"""
    ).fetchall()
    tools = connection.execute(
        """SELECT role, tool_name, COUNT(*) AS requests, SUM(cache_hit) AS cache_hits,
                  SUM(state = 'failed') AS failed, AVG(duration_ms) AS mean_duration_ms
           FROM actions WHERE kind = 'tool_request'
           GROUP BY role, tool_name ORDER BY role, requests DESC"""
    ).fetchall()
    experiments = connection.execute(
        """SELECT e.experiment_id, e.maximum_estimated_usd, e.created_at,
                  COALESCE(SUM(CASE WHEN a.usage_known = 1 THEN COALESCE(a.actual_usd, 0)
                      ELSE a.reserved_usd END), 0) AS known_or_reserved_usd
           FROM experiments e
           LEFT JOIN runs r ON r.experiment_id = e.experiment_id
           LEFT JOIN actions a ON a.run_id = r.run_id
           GROUP BY e.experiment_id ORDER BY e.created_at"""
    ).fetchall()
    return {
        "runs": [dict(row) for row in rows],
        "failures": [dict(row) for row in failures],
        "tools": [dict(row) for row in tools],
        "experiments": [dict(row) for row in experiments],
    }


def _provider_step(response: dict[str, Any] | None) -> dict[str, Any]:
    if response is None:
        return {}
    tool_calls = response.get("tool_calls") or []
    return {
        "stop_reason": response.get("stop_reason"),
        "tool_calls": [call.get("name") for call in tool_calls],
        "text_preview": _preview(response.get("text") or "", 400),
        "reasoning_tokens": (response.get("usage") or {}).get("reasoning_tokens"),
    }


def _step(row: sqlite3.Row, origin_ms: float) -> dict[str, Any]:
    request = json.loads(row["request_json"])
    error = None if row["error_json"] is None else json.loads(row["error_json"])
    start = _milliseconds(row["started_at"] or row["created_at"]) or origin_ms
    step: dict[str, Any] = {
        "action_id": row["action_id"],
        "role": row["role"],
        "kind": row["kind"],
        "tool_name": row["tool_name"],
        "attempt_number": row["attempt_number"],
        "state": row["state"],
        "offset_ms": round(start - origin_ms, 1),
        "duration_ms": row["duration_ms"],
        "input_tokens": row["actual_input_tokens"]
        if row["usage_known"]
        else row["reserved_input_tokens"],
        "output_tokens": row["actual_output_tokens"]
        if row["usage_known"]
        else row["reserved_output_tokens"],
        "usd": row["actual_usd"] if row["usage_known"] else row["reserved_usd"],
        "usage_known": bool(row["usage_known"]),
        "cache_hit": bool(row["cache_hit"]),
        "evidence_id": row["evidence_id"],
        "error_code": row["error_code"],
        "error_message": None if error is None else _preview(error.get("message", error), 400),
    }
    if row["kind"] == "tool_request":
        step["arguments"] = request.get("arguments", {})
    elif row["kind"] == "provider_attempt":
        response = None if row["response_json"] is None else json.loads(row["response_json"])
        step.update(_provider_step(response))
        step["turn"] = request.get("turn")
    else:
        step["description"] = request.get("description")
    return step


def _findings_summary(revision: int, findings_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "revision": revision,
        "findings_id": findings_id,
        "cause_status": payload.get("cause_status"),
        "decision_summary": _preview(payload.get("decision_summary", ""), SUMMARY_CHARACTERS),
        "factual_claims": len(payload.get("factual_claims", [])),
        "hypotheses": [
            {"hypothesis": entry.get("hypothesis_id"), "status": entry.get("status")}
            for entry in payload.get("hypotheses", [])
        ],
        "observation_patterns": payload.get("observation_patterns", []),
    }


def run_detail(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    run_row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run_row is None:
        raise KeyError(run_id)
    run = dict(run_row)
    limits = json.loads(run.pop("limits_json"))
    context = json.loads(run.pop("context_json"))
    identity = json.loads(run.pop("identity_json"))
    origin_ms = _milliseconds(run["created_at"]) or 0.0
    usage = budget_usage(connection, run_id).__dict__
    actions = connection.execute(
        "SELECT * FROM actions WHERE run_id = ? ORDER BY action_id", (run_id,)
    ).fetchall()
    evidence = {
        row["evidence_id"]: json.loads(row["record_json"])
        for row in connection.execute(
            "SELECT evidence_id, record_json FROM evidence WHERE run_id = ?", (run_id,)
        )
    }
    events = [
        {
            "event_type": row["event_type"],
            "offset_ms": round((_milliseconds(row["created_at"]) or origin_ms) - origin_ms, 1),
            "payload": row["payload_json"] and json.loads(row["payload_json"]),
        }
        for row in connection.execute(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY event_sequence", (run_id,)
        )
    ]
    findings = [
        _findings_summary(row["revision"], row["findings_id"], json.loads(row["payload_json"]))
        for row in connection.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY revision", (run_id,)
        )
    ]
    verifications = []
    for row in connection.execute(
        "SELECT * FROM verification_results WHERE run_id = ? ORDER BY created_at", (run_id,)
    ):
        payload = json.loads(row["payload_json"])
        verifications.append(
            {
                "findings_id": row["findings_id"],
                "passed": bool(row["passed"]),
                "claim_checks": len(payload.get("claim_checks", [])),
                "failed_checks": [
                    {key: check.get(key) for key in CHECK_FIELDS}
                    for check in payload.get("claim_checks", [])
                    if not check.get("passed")
                ],
                "policy_errors": payload.get("policy_errors", []),
            }
        )
    reviews = []
    review_rows = connection.execute(
        "SELECT * FROM reviews WHERE run_id = ? ORDER BY round", (run_id,)
    ).fetchall()
    for row in review_rows:
        payload = json.loads(row["payload_json"])
        reviews.append(
            {
                "round": row["round"],
                "decision": payload.get("decision"),
                "disputed_claim_ids": payload.get("disputed_claim_ids", []),
                "requested_queries": payload.get("requested_queries", []),
                "independent_query_count": row["independent_query_count"],
                "reason_summary": _preview(payload.get("reason_summary", ""), SUMMARY_CHARACTERS),
                "review_limitation": payload.get("review_limitation"),
            }
        )
    report_row = connection.execute(
        "SELECT payload_json FROM reports WHERE run_id = ?", (run_id,)
    ).fetchone()
    report = None
    if report_row is not None:
        payload = json.loads(report_row["payload_json"])
        report = {
            key: payload.get(key)
            for key in (
                "workflow_outcome",
                "terminal_reason",
                "fact_check_status",
                "review_status",
                "cause_status",
                "report_basis",
            )
        }
        for key in ("verified_claims", "rejected_claims", "unverified_claims"):
            report[key] = len(payload.get(key) or [])
        report["limitations"] = payload.get("limitations", [])
    candidate = context.get("candidate", {})
    return {
        "run": run,
        "model_id": identity.get("model_id"),
        "candidate": {
            key: candidate.get(key)
            for key in (
                "event_id",
                "session_id",
                "detector_name",
                "event_window",
                "reference_median_mbps",
                "event_median_mbps",
                "relative_drop",
            )
        },
        "budget": [
            {
                "name": label,
                "used": usage[used_key],
                "limit": limits.get(limit_key),
            }
            for used_key, limit_key, label in BUDGET_LIMITS
        ],
        "usage": usage,
        "events": events,
        "steps": [_step(row, origin_ms) for row in actions],
        "evidence": {
            identifier: {
                "tool_name": record.get("tool_name"),
                "effective_bounds": record.get("effective_bounds"),
                "quality_flags": record.get("quality_flags"),
                "units": record.get("units"),
                "result_preview": _preview(record.get("result")),
            }
            for identifier, record in evidence.items()
        },
        "findings": findings,
        "verifications": verifications,
        "reviews": reviews,
        "report": report,
    }


def page_template() -> str:
    return resources.files("celltrace.reporting").joinpath("run_viewer.html").read_text("utf-8")


def _embed(payload: dict[str, Any]) -> str:
    """JSON that cannot close the script element it is embedded in."""
    return json.dumps(payload, sort_keys=True, default=str).replace("</", "<\\/")


def render_static(store_path: Path, output_path: Path) -> dict[str, Any]:
    with read_only(store_path) as connection:
        summary = overview(connection)
        details = {run["run_id"]: run_detail(connection, run["run_id"]) for run in summary["runs"]}
    payload = {
        "viewer_version": VIEWER_VERSION,
        "mode": "snapshot",
        "store": str(store_path),
        "overview": summary,
        "details": details,
    }
    page = page_template().replace("__VIEWER_DATA__", _embed(payload))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return {"page": str(output_path), "runs": len(details), "bytes": output_path.stat().st_size}


class _ViewerHandler(BaseHTTPRequestHandler):
    store_path: Path

    def log_message(self, format: str, *arguments: Any) -> None:
        return

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        self._send(status, body, "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                live = {
                    "viewer_version": VIEWER_VERSION,
                    "mode": "live",
                    "store": str(self.store_path),
                }
                page = page_template().replace("__VIEWER_DATA__", _embed(live))
                self._send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/overview":
                with read_only(self.store_path) as connection:
                    self._json(overview(connection))
            elif path.startswith("/api/runs/"):
                run_id = unquote(path.removeprefix("/api/runs/"))
                with read_only(self.store_path) as connection:
                    self._json(run_detail(connection, run_id))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except KeyError:
            self._json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
        except (sqlite3.Error, OSError) as failure:
            self._json({"error": str(failure)}, HTTPStatus.SERVICE_UNAVAILABLE)


def make_server(store_path: Path, port: int) -> ThreadingHTTPServer:
    """A loopback only server. Port 0 asks the operating system for a free port."""
    if not store_path.is_file():
        raise FileNotFoundError(f"no run store at {store_path}")
    handler = type("ViewerHandler", (_ViewerHandler,), {"store_path": store_path})
    return ThreadingHTTPServer((LOOPBACK_HOST, port), handler)
