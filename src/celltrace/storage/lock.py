"""Exclusive run ownership through an OS released file lock.

``fcntl.flock`` locks belong to the open file description, so the kernel
releases them when the owning process exits, including after a crash. A
second process that tries to own the same run fails immediately.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType


class RunLockedError(RuntimeError):
    pass


class RunLock:
    def __init__(self, lock_directory: Path, run_id: str):
        lock_directory.mkdir(parents=True, exist_ok=True)
        self.path = lock_directory / f"{run_id}.lock"
        self.file_descriptor: int | None = None

    def acquire(self) -> RunLock:
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RunLockedError(f"run is owned by another process: {self.path}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode())
        self.file_descriptor = descriptor
        return self

    def release(self) -> None:
        if self.file_descriptor is not None:
            fcntl.flock(self.file_descriptor, fcntl.LOCK_UN)
            os.close(self.file_descriptor)
            self.file_descriptor = None

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
