import sys
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

from agent_bot.config import McpServerConfig
from agent_bot.mcp_tools import connect_mcp_servers, expand_env, flatten_content, open_server
from agent_bot.registry import Tool, ToolRegistry


def echo_cfg():
    return McpServerConfig(command=sys.executable, args=["-m", "tests.echo_mcp_server"])


def test_expand_env():
    assert expand_env("Bearer ${K}", {"K": "x"}) == "Bearer x"
    with pytest.raises(KeyError):
        expand_env("${NOPE}", {})


async def test_roundtrip_over_stdio():
    reg = ToolRegistry()
    async with AsyncExitStack() as stack:
        st = await connect_mcp_servers({"echo": echo_cfg()}, reg, stack, env={})
        assert st[0].status == "connected" and st[0].tool_count == 1
        r = await reg.call("echo", {"text": "I like tea"}, 30)
        assert not r.is_error and r.content == "I like tea"
        assert reg.get("echo").source == "mcp:echo"


async def test_mcp_error_result_is_verbatim():
    import agent_bot.mcp_tools as m

    text = "Tool 'x' failed: ValueError: bad"

    class Session:
        async def call_tool(self, name, args):
            return SimpleNamespace(is_error=True, content=[SimpleNamespace(type="text", text=text)])

    # exercise the real wrapper through open_server's tool builder
    tool = m._wrap_tool(
        "srv",
        SimpleNamespace(name="x", description=None, input_schema={"type": "object"}, meta=None),
        Session(),
    )
    reg = ToolRegistry()
    reg.register(tool)
    r = await reg.call("x", {}, 5)
    assert r.is_error and r.content == text


async def test_failed_server_is_skipped():
    reg = ToolRegistry()
    async with AsyncExitStack() as stack:
        st = await connect_mcp_servers(
            {"bad": McpServerConfig(command="definitely-not-a-binary")}, reg, stack, env={}
        )
    assert st[0].status == "failed" and st[0].error


async def test_missing_env_var_fails_server():
    reg = ToolRegistry()
    async with AsyncExitStack() as stack:
        st = await connect_mcp_servers(
            {"s": McpServerConfig(command=sys.executable, env={"X": "${NOPE}"})}, reg, stack, env={}
        )
    assert st[0].status == "failed" and "NOPE" in st[0].error


async def test_disabled_server():
    reg = ToolRegistry()
    async with AsyncExitStack() as stack:
        st = await connect_mcp_servers(
            {"s": McpServerConfig(command="x", enabled=False)}, reg, stack, env={}
        )
    assert st[0].status == "disabled" and st[0].tool_count == 0


async def test_name_collision_prefixed():
    reg = ToolRegistry()

    async def f(**kw):
        return "py"

    reg.register(Tool("echo", "", {"type": "object"}, f, "python"))
    async with AsyncExitStack() as stack:
        await connect_mcp_servers({"echo": echo_cfg()}, reg, stack, env={})
        assert "echo__echo" in reg
        assert reg.get("echo").source == "python"


def test_flatten_content():
    blocks = [
        SimpleNamespace(type="text", text="a"),
        SimpleNamespace(type="image"),
        SimpleNamespace(type="audio"),
        SimpleNamespace(type="text", text="b"),
    ]
    assert flatten_content(blocks) == "a\n[image omitted]\n[audio omitted]\nb"
