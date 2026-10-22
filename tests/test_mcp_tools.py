import json
import sys
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

from agent_bot.config import McpServerConfig
from agent_bot.mcp_tools import connect_mcp_servers, expand_env, flatten_content, open_server
from agent_bot.registry import Tool, ToolRegistry


def notes_cfg(path):
    return McpServerConfig(
        command=sys.executable,
        args=["-m", "mcp_servers.notes_server"],
        env={"NOTES_FILE": str(path)},
    )


def test_expand_env():
    assert expand_env("Bearer ${K}", {"K": "x"}) == "Bearer x"
    with pytest.raises(KeyError):
        expand_env("${NOPE}", {})


async def test_notes_roundtrip_over_stdio(tmp_path):
    reg = ToolRegistry()
    async with AsyncExitStack() as stack:
        st = await connect_mcp_servers({"notes": notes_cfg(tmp_path / "n.json")}, reg, stack, env={})
        assert st[0].status == "connected" and st[0].tool_count == 3
        r = await reg.call("add_note", {"text": "I like tea"}, 30)
        assert not r.is_error and "Saved note" in r.content
        r = await reg.call("list_notes", {}, 30)
        assert "I like tea" in r.content
        assert reg.get("list_notes").source == "mcp:notes"
        r = await reg.call("search_notes", {"query": "TEA"}, 30)
        assert "I like tea" in r.content
        r = await reg.call("search_notes", {"query": "zzz"}, 30)
        assert r.content == "No notes match 'zzz'."


async def test_notes_reads_file_each_call(tmp_path):
    f = tmp_path / "n.json"
    reg = ToolRegistry()
    async with AsyncExitStack() as stack:
        await connect_mcp_servers({"notes": notes_cfg(f)}, reg, stack, env={})
        await reg.call("add_note", {"text": "a"}, 30)
        f.write_text(json.dumps([]))
        r = await reg.call("list_notes", {}, 30)
        assert r.content == "No notes yet."


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


async def test_name_collision_prefixed(tmp_path):
    reg = ToolRegistry()

    async def f(**kw):
        return "py"

    reg.register(Tool("list_notes", "", {"type": "object"}, f, "python"))
    async with AsyncExitStack() as stack:
        await connect_mcp_servers({"notes": notes_cfg(tmp_path / "n.json")}, reg, stack, env={})
        assert "notes__list_notes" in reg
        assert reg.get("list_notes").source == "python"


def test_flatten_content():
    blocks = [
        SimpleNamespace(type="text", text="a"),
        SimpleNamespace(type="image"),
        SimpleNamespace(type="audio"),
        SimpleNamespace(type="text", text="b"),
    ]
    assert flatten_content(blocks) == "a\n[image omitted]\n[audio omitted]\nb"
