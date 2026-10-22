"""Per-turn JSONL traces for debugging."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol


class Tracer(Protocol):
    def record(self, kind: str, data: dict) -> None: ...


class NullTracer:
    def record(self, kind: str, data: dict) -> None:
        pass


class JsonlTracer:
    def __init__(self, trace_dir: Path, session_id: str):
        self._dir = Path(trace_dir)
        self._session = session_id
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    def record(self, kind: str, data: dict) -> None:
        if self._path is None:
            self._dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._path = self._dir / f"{stamp}-{self._session[:8]}.jsonl"
        line = json.dumps({"kind": kind, "ts": time.time(), **data},
                          ensure_ascii=False, default=repr)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
