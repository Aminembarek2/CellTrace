"""The runs command group: a static run viewer page and a live loopback monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXIT_OK = 0


def command_view(arguments: argparse.Namespace) -> int:
    from celltrace.reporting.run_viewer import render_static

    result = render_static(Path(arguments.store), Path(arguments.out))
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_OK


def command_watch(arguments: argparse.Namespace) -> int:
    from celltrace.reporting.run_viewer import LOOPBACK_HOST, make_server

    server = make_server(Path(arguments.store), arguments.port)
    address = f"http://{LOOPBACK_HOST}:{server.server_port}/"
    print(f"Watching {arguments.store} read only at {address} (Ctrl+C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return EXIT_OK


def command_export_otel(arguments: argparse.Namespace) -> int:
    from celltrace.reporting.otel_export import export_traces

    result = export_traces(Path(arguments.store), Path(arguments.out))
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_OK


def register(subparsers: Any) -> None:
    runs = subparsers.add_parser("runs", help="inspect and monitor agent runs in a run store")
    actions = runs.add_subparsers(dest="runs_command", required=True)

    view = actions.add_parser("view", help="write a self contained HTML run viewer")
    view.add_argument("--store", required=True, help="runs.sqlite3 written by investigations")
    view.add_argument("--out", required=True, help="HTML file to write")
    view.set_defaults(handler=command_view)

    watch = actions.add_parser("watch", help="serve a live read only viewer on 127.0.0.1")
    watch.add_argument("--store", required=True, help="runs.sqlite3 written by investigations")
    watch.add_argument("--port", type=int, default=8765)
    watch.set_defaults(handler=command_watch)

    export = actions.add_parser(
        "export-otel", help="write runs as OTLP JSON traces with GenAI span conventions"
    )
    export.add_argument("--store", required=True, help="runs.sqlite3 written by investigations")
    export.add_argument("--out", required=True, help="JSON file to write")
    export.set_defaults(handler=command_export_otel)
