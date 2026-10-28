"""The sandbox tool server, run in-process on a free port (no Docker), and the host's
per-turn connection to it."""
import ast
import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from agent_bot.config import AppConfig, McpServerConfig, ModelConfig, SandboxConfig, ToolsConfig
from agent_bot.mcp_tools import turn_tools
from agent_bot.registry import ToolRegistry, function_tool
from sandbox import tool_server

ROOT = Path(__file__).parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sandbox_cfg(*, python_tools=True, timeout_s=20, extra_servers=None) -> AppConfig:
    servers = {"echo": McpServerConfig(command=sys.executable, args=["-m", "tests.echo_mcp_server"])}
    servers.update(extra_servers or {})
    return AppConfig(model=ModelConfig(path="models/fake"),
                     tools=ToolsConfig(timeout_s=timeout_s),
                     sandbox=SandboxConfig(python_tools=python_tools, mcp_servers=servers))


class RunningServer:
    """create_app(cfg) under uvicorn in a background thread, like the container runs it."""

    def __init__(self, cfg: AppConfig, port: int):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.server = uvicorn.Server(uvicorn.Config(tool_server.create_app(cfg), host="127.0.0.1",
                                                    port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise RuntimeError("tool server did not start")

    def health(self) -> dict:
        return httpx.get(f"{self.url}/health", timeout=5).json()

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15)


@pytest.fixture
def start_server():
    running = []

    def start(cfg: AppConfig, port: int | None = None) -> RunningServer:
        server = RunningServer(cfg, port or free_port())
        running.append(server)
        return server

    yield start
    for server in running:
        server.stop()


@pytest.fixture
def tool_server_url(start_server):
    return start_server(sandbox_cfg()).url


def sandbox(url: str) -> dict[str, McpServerConfig]:
    return {"sandbox": McpServerConfig(url=f"{url}/mcp")}


def test_health_lists_counts(start_server):
    body = start_server(sandbox_cfg()).health()
    assert body["status"] == "ok"
    assert body["tools"] == 8
    assert body["mcp_servers"][0]["name"] == "echo"
    assert body["mcp_servers"][0]["status"] == "connected"


async def test_host_sees_sandbox_tools_with_sources(tool_server_url):
    reg = ToolRegistry()
    async with turn_tools(sandbox(tool_server_url), reg) as statuses:
        assert statuses[0].status == "connected" and statuses[0].tool_count == 8
        assert reg.get("calculator").source == "sandbox/python"
        assert reg.get("add_note").source == "sandbox/python"
        assert reg.get("echo").source == "sandbox/mcp:echo"
        assert all(t.origin == "sandbox" for t in reg.tools())


async def test_call_through_sandbox(tool_server_url):
    reg = ToolRegistry()
    async with turn_tools(sandbox(tool_server_url), reg):
        result = await reg.call("calculator", {"expression": "0.175 * 2340"}, 25)
    assert not result.is_error
    assert result.content == "409.5"


async def test_error_crosses_boundary_verbatim(tool_server_url):
    reg = ToolRegistry()
    async with turn_tools(sandbox(tool_server_url), reg):
        result = await reg.call("get_current_time", {"timezone": "Mars/Base"}, 25)
    assert result.is_error
    assert result.content.startswith("Tool 'get_current_time' failed: ValueError: Unknown timezone")
    assert "ToolError" not in result.content
    assert "RuntimeError" not in result.content


async def test_sandbox_timeout_message_wins(start_server, monkeypatch):
    async def sleepy() -> str:
        """Sleep for a long time."""
        await asyncio.sleep(10)
        return "woke up"

    real = tool_server.register_python_tools

    def with_sleepy(registry):
        real(registry)
        registry.register(function_tool(sleepy))

    monkeypatch.setattr(tool_server, "register_python_tools", with_sleepy)
    url = start_server(sandbox_cfg(timeout_s=0.2)).url
    reg = ToolRegistry()
    async with turn_tools(sandbox(url), reg):
        result = await reg.call("sleepy", {}, 5)
    assert result.is_error
    assert result.content == "Tool 'sleepy' timed out after 0.2s"


async def test_unreachable_sandbox():
    reg = ToolRegistry()
    async with turn_tools(sandbox(f"http://127.0.0.1:{free_port()}"), reg) as statuses:
        assert statuses[0].name == "sandbox"
        assert statuses[0].status == "failed"
        assert statuses[0].error
        assert [t for t in reg.tools() if t.origin == "sandbox"] == []


async def test_unreachable_sandbox_drops_tools_from_an_earlier_turn(tool_server_url):
    reg = ToolRegistry()
    async with turn_tools(sandbox(tool_server_url), reg):
        assert reg.get("calculator") is not None
    dead = {"sandbox": McpServerConfig(url=f"http://127.0.0.1:{free_port()}/mcp")}
    async with turn_tools(dead, reg):
        assert reg.tools() == []


async def test_tools_refresh_per_turn(start_server):
    first = start_server(sandbox_cfg())
    reg = ToolRegistry()
    async with turn_tools(sandbox(first.url), reg):
        assert len(reg.tools()) == 8
    first.stop()  # like `docker compose restart` with a changed config
    second = start_server(sandbox_cfg(python_tools=False), port=first.port)
    async with turn_tools(sandbox(second.url), reg):
        assert sorted(t.name for t in reg.tools()) == ["echo"]


def test_broken_inner_server_does_not_stop_sandbox(start_server):
    broken = {"broken": McpServerConfig(command="definitely-not-a-binary")}
    body = start_server(sandbox_cfg(extra_servers=broken)).health()
    assert body["status"] == "ok"
    status = {s["name"]: s for s in body["mcp_servers"]}
    assert status["broken"]["status"] == "failed"
    assert status["broken"]["error"]
    assert status["echo"]["status"] == "connected"
    assert body["tools"] == 8


async def test_host_header_must_be_local(tool_server_url):
    # A web page can't reach the tools through DNS rebinding: a foreign Host is refused.
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {"Accept": "application/json, text/event-stream",
               "MCP-Protocol-Version": "2025-06-18"}
    async with httpx.AsyncClient() as client:
        local = await client.post(f"{tool_server_url}/mcp", headers=headers, json=body)
        foreign = await client.post(f"{tool_server_url}/mcp", json=body,
                                    headers={**headers, "Host": "evil.example"})
    assert local.status_code == 200
    assert foreign.status_code == 421


def test_tool_server_has_no_forbidden_imports():
    tree = ast.parse((ROOT / "sandbox" / "tool_server.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    forbidden = {"agent_bot.inference", "agent_bot.agent", "mlx", "mlx_lm", "torch"}
    assert not {n for n in names if n in forbidden or n.split(".")[0] in {"mlx", "mlx_lm", "torch"}}
