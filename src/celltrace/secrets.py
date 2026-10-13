"""Explicit loading of one credential without evaluating shell expressions."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path


def load_openrouter_key(path: Path, environment: MutableMapping[str, str]) -> None:
    if environment.get("OPENROUTER_API_KEY"):
        return
    values = []
    for original_line in path.read_text(encoding="utf-8").splitlines():
        line = original_line.strip()
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        name, separator, value = line.partition("=")
        if not separator or name.strip() != "OPENROUTER_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
            raise ValueError("credential file must contain a literal OPENROUTER_API_KEY")
        values.append(value)
    if len(values) != 1:
        raise ValueError("credential file must contain exactly one OPENROUTER_API_KEY")
    environment["OPENROUTER_API_KEY"] = values[0]
