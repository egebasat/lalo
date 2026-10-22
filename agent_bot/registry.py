"""Tool abstraction shared by Python-function tools and MCP tools."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

import jsonschema

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Awaitable[str]]
    source: str


@dataclass
class ToolResult:
    content: str
    is_error: bool
    duration_ms: int


def tool(fn):
    """Mark a function as an agent tool."""
    fn.__agent_tool__ = True
    return fn


def function_tool(fn: Callable, source: str = "python") -> Tool:
    """Build a Tool from a function with a Google-style docstring."""
    from transformers.utils import get_json_schema

    spec = get_json_schema(fn)["function"]

    if inspect.iscoroutinefunction(fn):
        async def func(**kwargs: Any) -> str:
            return str(await fn(**kwargs))
    else:
        async def func(**kwargs: Any) -> str:
            return str(await asyncio.to_thread(fn, **kwargs))

    return Tool(
        name=spec["name"],
        description=spec.get("description", ""),
        parameters=spec["parameters"],
        func=func,
        source=source,
    )


class ToolRegistry:
    def __init__(self, max_result_chars: int = 4000, disabled: Iterable[str] = ()):
        self.max_result_chars = max_result_chars
        self.disabled = set(disabled)
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self.disabled:
            log.info("Tool '%s' is disabled; not registering", tool.name)
            return
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name '{tool.name}'")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def _truncate(self, content: str) -> str:
        extra = len(content) - self.max_result_chars
        if extra > 0:
            return content[: self.max_result_chars] + f"…[truncated {extra} chars]"
        return content

    async def call(self, name: str, arguments: dict, timeout_s: float) -> ToolResult:
        start = time.perf_counter()

        def done(content: str, is_error: bool) -> ToolResult:
            ms = int((time.perf_counter() - start) * 1000)
            return ToolResult(self._truncate(content), is_error, ms)

        t = self._tools.get(name)
        if t is None:
            available = ", ".join(self._tools)
            return done(f"Unknown tool '{name}'. Available tools: {available}", True)
        try:
            jsonschema.validate(arguments, t.parameters)
        except jsonschema.ValidationError as e:
            return done(f"Invalid arguments for '{name}': {e.message}", True)
        except Exception as e:  # e.g. malformed schema or non-dict arguments
            return done(f"Invalid arguments for '{name}': {e}", True)
        try:
            result = await asyncio.wait_for(t.func(**arguments), timeout_s)
        except asyncio.TimeoutError:
            return done(f"Tool '{name}' timed out after {timeout_s:g}s", True)
        except Exception as e:
            return done(f"Tool '{name}' failed: {type(e).__name__}: {e}", True)
        return done(str(result), False)
