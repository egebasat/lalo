"""Test doubles for the agent loop: a scripted backend and an in-memory tracer."""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator


class FakeBackend:
    """Replays scripted outputs, one per generate() call, in 3 chunks each.

    An output that is an Exception instance is raised instead of streamed.
    """

    def __init__(self, outputs: list, family: str = "qwen3", block: threading.Event | None = None):
        self._outputs = list(outputs)
        self._family = family
        self.block = block
        self.last_token_count = 3
        self.calls: list[dict] = []  # each: {"messages": [...], "enable_thinking": bool}

    @property
    def model_family(self) -> str:
        return self._family

    @property
    def model_name(self) -> str:
        return "fake"

    def render_prompt(self, messages: list[dict], tools: list[dict], *, enable_thinking: bool) -> str:
        return json.dumps(messages)

    def generate(self, messages, tools, *, enable_thinking, max_new_tokens, temperature) -> Iterator[str]:
        self.calls.append({"messages": [dict(m) for m in messages], "enable_thinking": enable_thinking})
        if self.block is not None:
            self.block.wait()
        output = self._outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        third = max(1, len(output) // 3)
        yield output[:third]
        yield output[third : 2 * third]
        yield output[2 * third :]


class ListTracer:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []

    def record(self, kind: str, data: dict) -> None:
        self.records.append((kind, data))
