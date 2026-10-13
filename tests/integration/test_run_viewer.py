"""The run viewer reads a real demo run store without changing it, and serves only on loopback."""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from celltrace.canonical import file_sha256
from celltrace.reporting.run_viewer import (
    LOOPBACK_HOST,
    make_server,
    overview,
    read_only,
    render_static,
    run_detail,
)
from celltrace.storage.run_store import budget_usage
from celltrace.workflow.demo import run_offline_demo


@pytest.fixture(scope="module")
def demo_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("viewer")
    summary = run_offline_demo(root / "public", root / "private")
    assert summary["status"] == "passed"
    stores = sorted((root / "private").glob("*/runs.sqlite3"))
    assert len(stores) == 1
    return stores[0]


def test_overview_matches_the_ledger_and_lists_failures(demo_store: Path) -> None:
    with read_only(demo_store) as connection:
        summary = overview(connection)
        runs = {run["run_id"]: run for run in summary["runs"]}
        for run_id, run in runs.items():
            usage = budget_usage(connection, run_id)
            assert run["provider_attempts"] == usage.provider_attempts
            assert run["tool_requests"] == usage.tool_requests
            assert run["estimated_usd"] == pytest.approx(usage.estimated_usd)
    assert runs["run_demo_budget_exhausted"]["terminal_reason"] == "budget_exhausted:tool_requests"
    codes = {entry["error_code"] for entry in summary["failures"]}
    assert {"STORE_UNAVAILABLE", "PROVIDER_TRANSIENT"} <= codes


def test_detail_keeps_order_arguments_errors_and_budget_limits(demo_store: Path) -> None:
    with read_only(demo_store) as connection:
        detail = run_detail(connection, "run_demo_tool_and_provider_failure")
    action_ids = [step["action_id"] for step in detail["steps"]]
    assert action_ids == sorted(action_ids)
    failed = [step for step in detail["steps"] if step["state"] == "failed"]
    assert {step["error_code"] for step in failed} == {"STORE_UNAVAILABLE", "PROVIDER_TRANSIENT"}
    tool_steps = [step for step in detail["steps"] if step["kind"] == "tool_request"]
    assert all(isinstance(step["arguments"], dict) for step in tool_steps)
    limits = {entry["name"]: entry["limit"] for entry in detail["budget"]}
    assert limits["Tool requests"] == 12


def test_viewer_cannot_write_to_the_store(demo_store: Path) -> None:
    before = file_sha256(demo_store)
    with read_only(demo_store) as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM runs")
    assert file_sha256(demo_store) == before


def test_multi_query_read_keeps_one_snapshot_until_connection_closes(
    demo_store: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "runs # snapshot?.sqlite3"
    copy.write_bytes(demo_store.read_bytes())
    run_id = "run_demo_budget_exhausted"
    with sqlite3.connect(copy) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        with read_only(copy) as reader:
            original = run_detail(reader, run_id)["run"]["terminal_reason"]
            writer.execute(
                "UPDATE runs SET terminal_reason = ? WHERE run_id = ?", ("changed", run_id)
            )
            writer.commit()
            assert run_detail(reader, run_id)["run"]["terminal_reason"] == original
        with read_only(copy) as fresh:
            assert run_detail(fresh, run_id)["run"]["terminal_reason"] == "changed"


def test_static_page_embeds_data_that_cannot_close_its_script(
    demo_store: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "runs.sqlite3"
    copy.write_bytes(demo_store.read_bytes())
    hostile = "</script><script>alert(1)</script>"
    with sqlite3.connect(copy) as connection:
        connection.execute(
            "UPDATE runs SET terminal_reason = ? WHERE run_id = 'run_demo_budget_exhausted'",
            (hostile,),
        )
    result = render_static(copy, tmp_path / "viewer.html")
    page = (tmp_path / "viewer.html").read_text()
    assert result["runs"] == 6
    assert hostile not in page
    assert page.count("</script>") == 2
    embedded = page.split('id="viewer-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["overview"]["runs"]


@pytest.fixture
def served(demo_store: Path) -> Iterator[str]:
    server = make_server(demo_store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{LOOPBACK_HOST}:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_live_server_binds_loopback_and_serves_overview_and_details(served: str) -> None:
    assert served.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(served + "/api/overview") as response:
        summary = json.load(response)
        assert response.headers["Cache-Control"] == "no-store"
    assert len(summary["runs"]) == 6
    with urllib.request.urlopen(served + "/api/runs/run_demo_budget_exhausted") as response:
        assert json.load(response)["run"]["workflow_outcome"] == "exhausted"
    with urllib.request.urlopen(served + "/") as response:
        assert '"mode": "live"' in response.read().decode()
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(served + "/api/runs/run_absent")
    assert missing.value.code == 404


def test_live_server_sees_a_run_while_its_writer_is_still_open(
    demo_store: Path, tmp_path: Path
) -> None:
    from celltrace.clock import ManualClock
    from celltrace.storage.run_store import ActionKind, Reservation, RunLimits, RunStore

    copy = tmp_path / "runs.sqlite3"
    copy.write_bytes(demo_store.read_bytes())
    writer = RunStore(copy, ManualClock())
    server = make_server(copy, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://{LOOPBACK_HOST}:{server.server_port}"
    try:
        writer.create_run(
            "run_in_progress",
            "case_fixture",
            "event_fixture",
            "investigator_gated",
            "scripted_fixture",
            None,
            "snapshot_fixture",
            {},
            {},
            RunLimits(12, 12, 40000, 12000, None, 300000, 2, 1),
        )
        action_id = writer.reserve_action(
            "run_in_progress",
            "attempt_fixture",
            "logical_fixture",
            "investigator",
            ActionKind.provider_attempt,
            {"turn": 1},
            Reservation(input_tokens=100, output_tokens=100),
        )
        writer.mark_started(action_id)
        with urllib.request.urlopen(address + "/api/overview") as response:
            runs = {run["run_id"]: run for run in json.load(response)["runs"]}
        assert runs["run_in_progress"]["status"] != "terminal"
        assert runs["run_in_progress"]["active_actions"] == 1
    finally:
        server.shutdown()
        server.server_close()
        writer.close()


def test_otel_export_is_deterministic_and_every_parent_exists(
    demo_store: Path, tmp_path: Path
) -> None:
    from celltrace.reporting.otel_export import export_traces

    first = export_traces(demo_store, tmp_path / "first.json")
    export_traces(demo_store, tmp_path / "second.json")
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    payload = json.loads((tmp_path / "first.json").read_text())
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == first["spans"]
    identifiers = {span["spanId"] for span in spans}
    assert all(len(span["traceId"]) == 32 and len(span["spanId"]) == 16 for span in spans)
    assert all(span.get("parentSpanId", span["spanId"]) in identifiers for span in spans)
    operations = {span["name"].split(" ")[0] for span in spans}
    assert {"invoke_workflow", "invoke_agent", "chat", "execute_tool"} <= operations
    failed = [span for span in spans if span["status"]["code"] == 2]
    error_types = {
        attribute["value"]["stringValue"]
        for span in failed
        for attribute in span["attributes"]
        if attribute["key"] == "error.type"
    }
    assert {"STORE_UNAVAILABLE", "PROVIDER_TRANSIENT"} <= error_types


def test_unfinished_export_does_not_claim_success_or_execution_failure(
    demo_store: Path, tmp_path: Path
) -> None:
    from celltrace.reporting.otel_export import run_spans

    copy = tmp_path / "runs.sqlite3"
    copy.write_bytes(demo_store.read_bytes())
    run_id = "run_demo_budget_exhausted"
    with sqlite3.connect(copy) as writer:
        writer.execute(
            "UPDATE runs SET status = 'running', workflow_outcome = NULL, terminal_reason = NULL "
            "WHERE run_id = ?",
            (run_id,),
        )
        writer.execute(
            "UPDATE actions SET state = 'started', error_code = NULL, completed_at = NULL "
            "WHERE run_id = ? AND kind = 'provider_attempt'",
            (run_id,),
        )
    with read_only(copy) as reader:
        spans = run_spans(reader, run_id)
    assert spans[0]["status"]["code"] == 0
    calls = [span for span in spans if span["name"].startswith("chat ")]
    assert calls and all(span["status"]["code"] == 0 for span in calls)
    assert all(
        any(
            attribute["key"] == "celltrace.export_incomplete" and attribute["value"]["boolValue"]
            for attribute in span["attributes"]
        )
        for span in calls
    )
