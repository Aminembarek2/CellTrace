"""Canonical JSON and hashing.

Keys are sorted, separators are compact, floats use Python's shortest
round trip repr, None becomes null and nonfinite numbers are rejected.
Every content hash in the project goes through these two functions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class NonFiniteValueError(ValueError):
    pass


def _reject_nonfinite(value: Any, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteValueError(f"nonfinite number at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{location}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{location}[{index}]")


def canonical_json(value: Any) -> str:
    _reject_nonfinite(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
