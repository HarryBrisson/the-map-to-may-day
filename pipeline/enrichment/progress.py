from __future__ import annotations

import threading


_PRINT_LOCK = threading.Lock()


class StageProgress:
    """Tiny thread-safe per-page stage progress printer."""

    enabled: bool = True

    def __init__(self, page_id: str, stage: str, enabled: bool | None = None) -> None:
        self.page_id = page_id
        self.stage = stage
        self._enabled = self.enabled if enabled is None else enabled

    def _emit(self, message: str) -> None:
        if not self._enabled:
            return
        line = f"[{self.page_id}] {self.stage}: {message}"
        with _PRINT_LOCK:
            print(line, flush=True)

    def info(self, message: str) -> None:
        self._emit(message)

    def tick(self, current: int, total: int, label: str = "") -> None:
        msg = f"{current:>3}/{total} {label}".rstrip()
        self._emit(msg)

    def done(self, message: str = "", duration_s: float | None = None) -> None:
        if duration_s is not None:
            suffix = f"done in {duration_s:.1f}s"
            if message:
                suffix = f"{suffix} ({message})"
        else:
            suffix = f"done{(' ' + message) if message else ''}"
        self._emit(suffix)


def set_streaming_enabled(enabled: bool) -> None:
    StageProgress.enabled = enabled
