"""Injectable clocks so tests control time and timeouts."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def monotonic_ms(self) -> int: ...

    def utc_now_iso(self) -> str: ...


class SystemClock:
    def monotonic_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def utc_now_iso(self) -> str:
        return datetime.now(UTC).isoformat(timespec="milliseconds")


class ManualClock:
    """Deterministic clock for tests and offline fixtures."""

    def __init__(self, start_ms: int = 0, wall_clock: str = "2026-10-08T00:00:00.000+00:00"):
        self.current_ms = start_ms
        self.wall_clock = wall_clock

    def advance(self, milliseconds: int) -> None:
        self.current_ms += milliseconds

    def monotonic_ms(self) -> int:
        return self.current_ms

    def utc_now_iso(self) -> str:
        return self.wall_clock
