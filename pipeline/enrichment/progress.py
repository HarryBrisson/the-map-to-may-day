from __future__ import annotations

import sys
import threading


_PRINT_LOCK = threading.Lock()


class StageProgress:
    def __init__(self, page_id: str, stage: str, enabled: bool = True) -> None:
        self.page_id = page_id
        self.stage = stage
        self.enabled = enabled

    def info(self, message: str) -> None:
        self._emit(message)

    def tick(self, current: int, total: int, label: str = "") -> None:
        suffix = f" ({label})" if label else ""
        self._emit(f"{current:>3}/{total} {suffix}".rstrip())

    def done(self, message: str = "", duration_s: float | None = None) -> None:
        parts = []
        if duration_s is not None:
            parts.append(f"done in {duration_s:.1f}s")
        if message:
            parts.append(message)
        self._emit(" ".join(parts) if parts else "done")

    def _emit(self, message: str) -> None:
        if not self.enabled:
            return
        line = f"[{self.page_id}] {self.stage}: {message}"
        with _PRINT_LOCK:
            print(line, flush=True)
            sys.stdout.flush()
