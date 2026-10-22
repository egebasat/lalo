"""Events streamed from the agent loop to the GUI (one JSON object per line)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Union


class _EventBase:
    type: ClassVar[str]

    def to_json(self) -> str:
        data: dict[str, Any] = {"type": self.type, "ts": self.ts}  # type: ignore[attr-defined]
        for f in fields(self):  # type: ignore[arg-type]
            if f.name != "ts":
                data[f.name] = getattr(self, f.name)
        return json.dumps(data, ensure_ascii=False)


@dataclass(frozen=True)
class Token(_EventBase):
    type: ClassVar[str] = "token"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Thinking(_EventBase):
    type: ClassVar[str] = "thinking"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolCallEvent(_EventBase):
    type: ClassVar[str] = "tool_call"
    id: str
    name: str
    arguments: dict
    source: str
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolResultEvent(_EventBase):
    type: ClassVar[str] = "tool_result"
    id: str
    content: str
    is_error: bool
    duration_ms: int
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Answer(_EventBase):
    type: ClassVar[str] = "answer"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Error(_EventBase):
    type: ClassVar[str] = "error"
    message: str
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Done(_EventBase):
    type: ClassVar[str] = "done"
    iterations: int
    elapsed_s: float
    tokens_generated: int
    ts: float = field(default_factory=time.time)


Event = Union[Token, Thinking, ToolCallEvent, ToolResultEvent, Answer, Error, Done]
