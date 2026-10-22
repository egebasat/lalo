import asyncio

import pytest

from agent_bot.registry import ToolRegistry, function_tool, tool


@tool
def add(a: int, b: int = 2) -> str:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


@tool
async def aecho(text: str) -> str:
    """Echo text.

    Args:
        text: Text to echo.
    """
    return text


@tool
def boom() -> str:
    """Always fails."""
    raise ValueError("bad")


@tool
async def slow() -> str:
    """Sleeps."""
    await asyncio.sleep(1)
    return "done"


@tool
def long_result() -> str:
    """Returns a long string."""
    return "x" * 25


def make(*fns, **kw):
    reg = ToolRegistry(**kw)
    for fn in fns:
        reg.register(function_tool(fn))
    return reg


def test_schema_from_decorated_function():
    assert add.__agent_tool__ is True
    fn = make(add).schemas()[0]["function"]
    assert fn["name"] == "add"
    assert fn["parameters"]["required"] == ["a"]
    assert fn["parameters"]["properties"]["a"]["type"] == "integer"


async def test_call_sync_and_async_functions():
    reg = make(add, aecho)
    r = await reg.call("add", {"a": 1}, 5)
    assert (r.content, r.is_error) == ("3", False)
    r = await reg.call("aecho", {"text": "hi"}, 5)
    assert (r.content, r.is_error) == ("hi", False)


async def test_invalid_arguments():
    r = await make(add).call("add", {"a": "x"}, 5)
    assert r.is_error
    assert r.content.startswith("Invalid arguments for 'add'")


async def test_unknown_tool():
    r = await make(add).call("nope", {}, 5)
    assert r.is_error
    assert r.content.startswith("Unknown tool 'nope'")
    assert "Available tools: add" in r.content


async def test_exception_becomes_error():
    r = await make(boom).call("boom", {}, 5)
    assert r.is_error
    assert r.content == "Tool 'boom' failed: ValueError: bad"


async def test_timeout():
    r = await make(slow).call("slow", {}, 0.05)
    assert r.is_error
    assert "timed out" in r.content


async def test_truncation():
    r = await make(long_result, max_result_chars=10).call("long_result", {}, 5)
    assert r.content == "x" * 10 + "…[truncated 15 chars]"


def test_duplicate_name_raises():
    reg = make(add)
    with pytest.raises(ValueError):
        reg.register(function_tool(add))


def test_disabled_tool_not_registered():
    reg = make(add, disabled=["add"])
    assert "add" not in reg
    assert reg.schemas() == []
