import ast
import json
from pathlib import Path

import pytest

from agent_bot.agent import RunSettings, run
from agent_bot.parsing import get_parser
from agent_bot.registry import ToolRegistry, function_tool, tool
from agent_bot.tools import get_current_time
from tests.fakes import FakeBackend, ListTracer


@tool
def boom() -> str:
    """Always fails."""
    raise ValueError("bad")


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(function_tool(get_current_time))
    reg.register(function_tool(boom))
    return reg


def TC(name: str, args: dict) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"


def types(events) -> list[str]:
    return [e.type for e in events]


def no_tokens(events) -> list[str]:
    return [t for t in types(events) if t != "token"]


def settings(**kw) -> RunSettings:
    base = dict(system_prompt="SYS", enable_thinking=True, max_new_tokens=64,
                temperature=0.0, max_iterations=6, tool_timeout_s=5)
    base.update(kw)
    return RunSettings(**base)


async def collect(outputs, registry, history=None, tracer=None, **kw):
    backend = FakeBackend(outputs)
    history = history if history is not None else [{"role": "user", "content": "hi"}]
    extra = {"tracer": tracer} if tracer is not None else {}
    events = [e async for e in run(history, backend=backend, parser=get_parser("qwen3"),
                                   registry=registry, settings=settings(**kw), **extra)]
    return events, history, backend


async def test_no_tool_answer(registry):
    events, history, _ = await collect(["Hello"], registry)
    assert types(events) == ["token"] * 3 + ["answer", "done"]
    assert history[-1] == {"role": "assistant", "content": "Hello"}


async def test_single_tool_call(registry):
    events, history, _ = await collect([TC("get_current_time", {"timezone": "UTC"}), "It is noon"], registry)
    assert no_tokens(events) == ["tool_call", "tool_result", "answer", "done"]
    assert events[-1].iterations == 2
    assert any(m["role"] == "tool" and m["name"] == "get_current_time" for m in history)


async def test_multiple_calls_in_one_reply(registry):
    two = TC("get_current_time", {"timezone": "UTC"}) + TC("get_current_time", {"timezone": "Asia/Tokyo"})
    events, _, _ = await collect([two, "done"], registry)
    tool_events = [e for e in events if e.type in ("tool_call", "tool_result")]
    assert [(e.type, e.id) for e in tool_events] == [
        ("tool_call", "c1"), ("tool_result", "c1"), ("tool_call", "c2"), ("tool_result", "c2")]


async def test_tool_error_fed_back(registry):
    events, _, backend = await collect([TC("boom", {}), "sorry"], registry)
    result = next(e for e in events if e.type == "tool_result")
    assert result.is_error
    assert backend.calls[1]["messages"][-1]["content"].startswith("Tool 'boom' failed")


async def test_malformed_json_retry(registry):
    events, _, backend = await collect(["<tool_call>{bad</tool_call>", "ok"], registry)
    kinds = no_tokens(events)
    assert kinds == ["error", "answer", "done"]
    assert events[[e.type for e in events].index("error")].message.startswith("Malformed tool call")
    assert next(e for e in events if e.type == "answer").text == "ok"
    assert backend.calls[1]["messages"][-1]["role"] == "user"


async def test_max_iterations(registry):
    zones = ["UTC", "Asia/Tokyo", "Europe/Istanbul", "Europe/Paris", "America/New_York", "Asia/Dubai"]
    events, _, _ = await collect([TC("get_current_time", {"timezone": z}) for z in zones], registry,
                                 max_iterations=6)
    assert types(events)[-2:] == ["error", "done"]
    assert "6 iterations" in events[-2].message


async def test_repeat_guard(registry):
    call = TC("get_current_time", {"timezone": "UTC"})
    events, _, _ = await collect([call] * 3, registry)
    assert types(events).count("tool_result") == 2
    assert no_tokens(events)[-2:] == ["error", "done"]
    assert "3 times" in events[-2].message


async def test_unclosed_think_gives_visible_answer(registry):
    events, _, _ = await collect(["<think>never ends"], registry)
    assert no_tokens(events) == ["thinking", "answer", "done"]
    assert events[-2].text.startswith("(no answer")


async def test_thinking_not_in_history(registry):
    events, history, _ = await collect(["<think>secret plan</think>Hi there"], registry)
    assert next(e for e in events if e.type == "thinking").text == "secret plan"
    assert history[-1] == {"role": "assistant", "content": "Hi there"}
    assert "secret plan" not in json.dumps(history)


async def test_system_prompt_sent_not_stored(registry):
    _, history, backend = await collect(["Hello"], registry)
    sent = backend.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": "SYS"}
    assert all(m["role"] != "system" for m in history)


async def test_backend_exception(registry):
    events, _, _ = await collect([RuntimeError("GPU on fire")], registry)
    assert types(events) == ["error", "done"]
    assert events[0].message == "Model error: GPU on fire"


async def test_tool_calls_traced(registry):
    tracer = ListTracer()
    events, _, _ = await collect([TC("get_current_time", {"timezone": "UTC"}), "It is noon"], registry,
                                 tracer=tracer)
    kinds = [k for k, _ in tracer.records]
    assert kinds == ["prompt", "raw_output", "parsed", "tool_call", "tool_result",
                     "prompt", "raw_output", "parsed"]
    call = dict(tracer.records)["tool_call"]
    assert call["id"] == "c1" and call["name"] == "get_current_time"
    assert call["arguments"] == {"timezone": "UTC"}
    result = dict(tracer.records)["tool_result"]
    event = next(e for e in events if e.type == "tool_result")
    assert result["id"] == "c1" and result["is_error"] is False
    assert result["content"] == event.content


def test_agent_has_no_forbidden_imports():
    tree = ast.parse((Path(__file__).parent.parent / "agent_bot" / "agent.py").read_text())
    forbidden = {"transformers", "torch", "mlx", "mlx_lm", "mcp", "agent_bot.tools"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    for name in imported:
        assert not any(name == f or name.startswith(f + ".") for f in forbidden), name
