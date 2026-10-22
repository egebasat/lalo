"""The agent loop: a hand-written ReAct cycle of generate -> parse -> call tools -> repeat.

Follow one request through `run()`:
  1. The model sees the system prompt + history + tool schemas and streams raw text.
  2. The parser splits that text into thinking, tool calls and visible content.
  3. No tool calls -> that content is the answer and the turn ends.
  4. Tool calls -> run each one, append its result to history as a `tool` message,
     and go back to 1 so the model can read the results.

Everything the outside world sees is an `Event` (see events.py). This module knows
nothing about models, MCP or concrete tools: it only talks to the Backend, Parser and
ToolRegistry interfaces, so swapping any of them never touches the loop.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from agent_bot.events import (Answer, Done, Error, Event, Thinking, Token,
                              ToolCallEvent, ToolResultEvent)
from agent_bot.inference import Backend
from agent_bot.parsing import Parser
from agent_bot.registry import ToolRegistry
from agent_bot.trace import NullTracer, Tracer

# Small models sometimes get stuck re-issuing the same call; the 3rd repeat ends the turn.
MAX_IDENTICAL_CALLS = 3


@dataclass
class RunSettings:
    system_prompt: str
    enable_thinking: bool
    max_new_tokens: int
    temperature: float
    max_iterations: int
    tool_timeout_s: float


async def stream_in_thread(it: Iterator[str]) -> AsyncIterator[str]:
    """Drain a blocking iterator on a worker thread without blocking the event loop.

    Generation blocks for seconds; running it on the loop's thread would freeze every
    other coroutine (e.g. the HTTP server). The worker pushes each item into an
    asyncio.Queue via call_soon_threadsafe, the only thread-safe way into the loop.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    end = object()  # sentinel marking a clean finish

    def produce() -> None:
        try:
            for item in it:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            loop.call_soon_threadsafe(queue.put_nowait, end)
        except BaseException as exc:  # hand the error to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    threading.Thread(target=produce, daemon=True, name="generate").start()
    while (item := await queue.get()) is not end:
        if isinstance(item, BaseException):
            raise item
        yield item


@dataclass
class _Stats:
    """Counters the loop updates so `run` can report them even after an error."""
    iterations: int = 0
    tokens: int = 0


async def run(
    history: list[dict],
    *,
    backend: Backend,
    parser: Parser,
    registry: ToolRegistry,
    settings: RunSettings,
    tracer: Tracer = NullTracer(),
) -> AsyncIterator[Event]:
    """Run one user turn. Appends assistant/tool messages to `history`; `Done` is always last."""
    start = time.perf_counter()
    stats = _Stats()
    try:
        async for event in _loop(history, backend, parser, registry, settings, tracer, stats):
            yield event
    except Exception as exc:  # tool errors never raise (registry catches them), so this is the model
        yield Error(f"Model error: {exc}")
    yield Done(stats.iterations, round(time.perf_counter() - start, 1), stats.tokens)


async def _loop(history, backend, parser, registry, settings, tracer, stats) -> AsyncIterator[Event]:
    tools = registry.schemas()
    seen: Counter = Counter()  # (name, args) -> times called this turn, for the repeat guard
    call_number = 0  # tool-call ids c1, c2, ... are unique within the turn

    while stats.iterations < settings.max_iterations:
        stats.iterations += 1
        # The system prompt is rebuilt per call, never stored: history stays pure
        # conversation, and a changed prompt (e.g. a new date) applies immediately.
        messages = [{"role": "system", "content": settings.system_prompt}, *history]
        tracer.record("prompt", {"text": backend.render_prompt(
            messages, tools, enable_thinking=settings.enable_thinking)})

        # 1. Generate, streaming each chunk out as it arrives.
        chunks: list[str] = []
        stream = backend.generate(messages, tools, enable_thinking=settings.enable_thinking,
                                  max_new_tokens=settings.max_new_tokens,
                                  temperature=settings.temperature)
        async for chunk in stream_in_thread(stream):
            chunks.append(chunk)
            yield Token(chunk)
        stats.tokens += backend.last_token_count
        raw = "".join(chunks)

        # 2. Parse the raw text.
        parsed = parser.parse(raw)
        tracer.record("raw_output", {"text": raw})
        tracer.record("parsed", dataclasses.asdict(parsed))
        if parsed.thinking:
            yield Thinking(parsed.thinking)

        # Thinking is shown but never stored: the Qwen3 chat template drops it from
        # past turns anyway, and it would waste context on every later call.
        assistant: dict = {"role": "assistant", "content": parsed.content}
        if parsed.tool_calls:
            assistant["tool_calls"] = [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in parsed.tool_calls]
        history.append(assistant)

        # 3a. The model tried to call a tool but the markup was broken.
        if not parsed.tool_calls and parsed.parse_errors:
            errors = "; ".join(parsed.parse_errors)
            if len(parsed.parse_errors) == 1 and errors.startswith("unclosed <think>"):
                # It hit max_new_tokens mid-thought. Retrying would just think again,
                # so give the user something visible instead of a silent stop.
                yield Answer("(no answer: the model ran out of tokens while thinking)")
                return
            yield Error(f"Malformed tool call: {errors}")
            # Tell the model what went wrong, as the user; small models usually fix it.
            history.append({"role": "user", "content": (
                f"Your previous tool call could not be parsed ({errors}). Reply with a valid "
                '<tool_call>{"name": ..., "arguments": {...}}</tool_call> block, '
                "or answer directly.")})
            continue

        # 3b. No tool calls: the visible content is the final answer.
        if not parsed.tool_calls:
            yield Answer(parsed.content or "(the model returned an empty answer)")
            return

        # 4. Run the calls one by one (sequential keeps results in a predictable order
        #    and avoids overloading a local machine), feeding each result back.
        for call in parsed.tool_calls:
            key = (call.name, json.dumps(call.arguments, sort_keys=True))
            seen[key] += 1
            if seen[key] >= MAX_IDENTICAL_CALLS:
                yield Error(f"Stopped: '{call.name}' was called "
                            f"{MAX_IDENTICAL_CALLS} times with the same arguments")
                return

            call_number += 1
            call_id = f"c{call_number}"
            found = registry.get(call.name)
            source = found.source if found else "unknown"
            yield ToolCallEvent(call_id, call.name, call.arguments, source)
            tracer.record("tool_call", {"id": call_id, "name": call.name,
                                        "arguments": call.arguments, "source": source})

            # Never raises: unknown tools, bad arguments, timeouts and exceptions all
            # come back as is_error results, which the model reads and can recover from.
            result = await registry.call(call.name, call.arguments, settings.tool_timeout_s)
            yield ToolResultEvent(call_id, result.content, result.is_error, result.duration_ms)
            tracer.record("tool_result", {"id": call_id, "content": result.content,
                                          "is_error": result.is_error,
                                          "duration_ms": result.duration_ms})
            history.append({"role": "tool", "name": call.name, "content": result.content})

    yield Error(f"Stopped after {settings.max_iterations} iterations without a final answer")
