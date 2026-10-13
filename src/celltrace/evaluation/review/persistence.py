"""Atomic JSON publication after ledger commits, so recovery never reads a partial result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from celltrace.canonical import canonical_json


def write_json_atomic(path: Path, value: Any) -> None:
    canonical_json(value)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
