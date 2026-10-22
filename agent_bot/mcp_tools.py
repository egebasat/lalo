"""MCP client: connect to stdio / Streamable HTTP servers and wrap their tools."""
from __future__ import annotations

import logging
import math
import os
import re
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from agent_bot.config import McpServerConfig
from agent_bot.registry import Tool, ToolError, ToolRegistry

log = logging.getLogger(__name__)

SOURCE_META_KEY = "agent_bot/source"
CONNECT_TIMEOUT_S = 60

_VAR = re.compile(r"\$\{([^}]+)\}")


def expand_env(value: str, env: Mapping[str, str]) -> str:
    """Replace ${VAR} with env[VAR]; raise KeyError(VAR) if unset."""
    return _VAR.sub(lambda m: env[m.group(1)], value)


@dataclass
class McpServerStatus:
    name: str
    status: Literal["connected", "failed", "disabled"]
    error: str | None
    tool_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def flatten_content(blocks: list) -> str:
    parts = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            parts.append(b.text)
        elif t == "image":
            parts.append("[image omitted]")
        else:
            parts.append(f"[{t} omitted]")
    return "\n".join(parts)


def _wrap_tool(server: str, t: Any, session: Any, registered: Any = ()) -> Tool:
    meta = getattr(t, "meta", None) or {}
    source = (
        f"{server}/{meta[SOURCE_META_KEY]}" if SOURCE_META_KEY in meta else f"mcp:{server}"
    )
    tool_name = t.name

    async def func(**args: Any) -> str:
        try:
            result = await session.call_tool(tool_name, args)
        except Exception as e:
            raise ToolError(f"MCP server '{server}' error: {type(e).__name__}: {e}") from e
        text = flatten_content(result.content)
        if result.is_error:
            raise ToolError(text)
        return text

    name = tool_name
    if name in registered:
        name = f"{server}__{tool_name}"
        log.warning("MCP tool '%s' from '%s' clashes with existing tool; using '%s'", tool_name, server, name)
    return Tool(
        name=name,
        description=t.description or "",
        parameters=t.input_schema,
        func=func,
        source=source,
        origin=server,
    )


async def open_server(
    name: str,
    cfg: McpServerConfig,
    stack: AsyncExitStack,
    env: Mapping[str, str],
    registered: Any = (),
) -> tuple[ClientSession, list[Tool]]:
    """Connect one server (lifetime tied to `stack`) and return its wrapped tools."""
    if cfg.command:
        expanded = {k: expand_env(v, env) for k, v in cfg.env.items()}
        params = StdioServerParameters(
            command=cfg.command,
            args=[expand_env(a, env) for a in cfg.args],
            env=expanded or None,
            cwd=os.getcwd(),
        )
        read, write = await stack.enter_async_context(stdio_client(params))
    else:
        headers = {k: expand_env(v, env) for k, v in cfg.headers.items()}
        client = await stack.enter_async_context(
            create_mcp_http_client(headers=headers or None)
        )
        read, write = (
            await stack.enter_async_context(
                streamable_http_client(expand_env(cfg.url, env), http_client=client)
            )
        )[:2]
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    listing = await session.list_tools()
    tools = [_wrap_tool(name, t, session, registered) for t in listing.tools]
    return session, tools


async def connect_mcp_servers(
    configs: dict[str, McpServerConfig],
    registry: ToolRegistry,
    stack: AsyncExitStack,
    env: Mapping[str, str] | None = None,
) -> list[McpServerStatus]:
    env = os.environ if env is None else env
    statuses: list[McpServerStatus] = []
    for name, cfg in configs.items():
        if not cfg.enabled:
            statuses.append(McpServerStatus(name, "disabled", None, 0))
            continue
        sub = AsyncExitStack()
        try:
            # The deadline scope is the outermost context on `sub`, so it nests
            # correctly around the transport's task groups; once connected we lift it.
            scope = sub.enter_context(anyio.fail_after(CONNECT_TIMEOUT_S))
            taken = {t.name for t in registry.tools()}
            _, tools = await open_server(name, cfg, sub, env, taken)
            scope.deadline = math.inf
            for t in tools:
                registry.register(t)
            await stack.enter_async_context(sub)
            statuses.append(McpServerStatus(name, "connected", None, len(tools)))
        except BaseException as exc:
            err: BaseException = exc
            try:
                await sub.__aexit__(type(exc), exc, exc.__traceback__)
            except BaseException as exc2:
                err = exc2
            if not isinstance(err, Exception):
                raise
            log.warning("MCP server '%s' failed: %s", name, err)
            statuses.append(
                McpServerStatus(name, "failed", f"{type(err).__name__}: {err}", 0)
            )
    return statuses
