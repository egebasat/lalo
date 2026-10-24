"""The tool server: runs every tool inside the Docker sandbox and serves them over MCP.

    python -m sandbox.tool_server       # in the container; host/port from config.yaml

The host agent connects to http://127.0.0.1:8001/mcp once per turn and sees one MCP
server ("sandbox") whose tools are this process's Python tools plus the tools of the
MCP servers listed under `sandbox.mcp_servers`. Each tool's original source travels in
its `_meta`, so the GUI can still tell "sandbox/python" from "sandbox/mcp:local_notes".

This module must never import the model side (agent_bot.inference / agent_bot.agent,
mlx, torch): the image has no weights and no torch.
"""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

import mcp_types as types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_bot.config import AppConfig, load_config
from agent_bot.mcp_tools import SOURCE_META_KEY, McpServerStatus, connect_mcp_servers
from agent_bot.registry import ToolRegistry
from agent_bot.tools import register_python_tools

log = logging.getLogger(__name__)

# Only requests addressed to localhost are served. Docker publishes the port on the
# host's 127.0.0.1 only, and this also stops a web page from reaching the tools through
# DNS rebinding (its requests would carry the attacker's Host header).
LOCAL_HOSTS = ["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"]


async def build_registry(
    cfg: AppConfig, stack: AsyncExitStack
) -> tuple[ToolRegistry, list[McpServerStatus]]:
    """Python tools (if enabled), then the inner MCP servers, whose lifetime is `stack`."""
    registry = ToolRegistry(cfg.tools.max_result_chars, cfg.tools.disabled)
    if cfg.sandbox.python_tools:
        register_python_tools(registry)
    statuses = await connect_mcp_servers(cfg.sandbox.mcp_servers, registry, stack)
    return registry, statuses


def create_mcp_server(registry: ToolRegistry, timeout_s: float) -> Server:
    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=t.name, description=t.description, input_schema=t.parameters,
                       meta={SOURCE_META_KEY: t.source})
            for t in registry.tools()
        ])

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        # registry.call never raises: errors and timeouts come back as text, and we
        # return that text ourselves so the SDK doesn't wrap it a second time. The
        # host passes it to the model verbatim.
        result = await registry.call(params.name, params.arguments or {}, timeout_s)
        return types.CallToolResult(content=[types.TextContent(type="text", text=result.content)],
                                    is_error=result.is_error)

    return Server("agent_bot-sandbox", on_list_tools=list_tools, on_call_tool=call_tool)


def create_app(cfg: AppConfig) -> Starlette:
    state: dict = {"registry": ToolRegistry(), "statuses": []}
    manager: StreamableHTTPSessionManager | None = None

    @asynccontextmanager
    async def lifespan(app: Starlette):
        nonlocal manager
        async with AsyncExitStack() as stack:
            registry, statuses = await build_registry(cfg, stack)
            state.update(registry=registry, statuses=statuses)
            log.info("tool server: %d tools; %s", len(registry.tools()),
                     ", ".join(f"{s.name} {s.status}" for s in statuses) or "no MCP servers")
            manager = StreamableHTTPSessionManager(
                app=create_mcp_server(registry, cfg.tools.timeout_s), stateless=True,
                security_settings=TransportSecuritySettings(
                    enable_dns_rebinding_protection=True, allowed_hosts=LOCAL_HOSTS))
            async with manager.run():
                yield

    class McpEndpoint:  # a class instance, so Starlette routes the raw ASGI call to it
        async def __call__(self, scope, receive, send) -> None:
            await manager.handle_request(scope, receive, send)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "tools": len(state["registry"].tools()),
                             "mcp_servers": [s.to_dict() for s in state["statuses"]]})

    return Starlette(routes=[Route("/health", health), Route("/mcp", McpEndpoint())],
                     lifespan=lifespan)


def main() -> None:
    cfg = load_config()
    logging.basicConfig(level=cfg.logging.level)
    uvicorn.run(create_app(cfg), host=cfg.sandbox.host, port=cfg.sandbox.port)


if __name__ == "__main__":
    main()
