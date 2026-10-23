# Tool calling and MCP configuration

## How the model calls a tool

The model never runs anything itself. A tool call is just text the model writes,
which our code recognizes and runs for it.

1. **Tools become schemas.** Every tool lives in the `ToolRegistry`
   (`agent_bot/registry.py`) as a `Tool`: name, description, JSON-Schema
   `parameters`, and an async `func`. `registry.schemas()` turns them into the
   OpenAI-style list `[{"type": "function", "function": {name, description, parameters}}]`.
2. **Schemas go into the prompt.** The backend (`agent_bot/inference.py`) passes
   that list as `tools=` to the tokenizer's `apply_chat_template`. Qwen3's chat
   template writes the schemas into the system turn and tells the model to answer
   with a block like:

   ```
   <tool_call>
   {"name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}}
   </tool_call>
   ```
3. **The output is parsed.** `Qwen3Parser` (`agent_bot/parsing.py`) splits the raw
   text into `<think>` content, `<tool_call>` blocks and the visible rest. Each block
   must be a JSON object with a string `name` and an object `arguments` (a JSON
   string or a ```` ``` ```` fence around it is tolerated). Anything broken is
   reported in `parse_errors` instead of raising.
4. **The registry runs the call.** `registry.call(name, arguments, timeout_s)`
   looks the tool up, validates the arguments against its schema with `jsonschema`,
   and awaits `func` under a timeout. It never raises: unknown tools, bad arguments,
   timeouts and exceptions all come back as a `ToolResult` with `is_error=True`
   and a message the model can read. Long results are cut to `tools.max_result_chars`.
5. **The result goes back to the model** as a `{"role": "tool", ...}` message in
   history; the chat template renders it as a `<tool_response>` on the next call.

### Where tools come from

| Kind | Defined in | `source` shown in events |
|---|---|---|
| Python tools | `@tool` functions in `agent_bot/tools.py` (`web_search`, `fetch_page`, `get_current_time`, `calculator`) | `python` |
| MCP tools | any connected MCP server, e.g. the bundled notes server | `mcp:<server>` (or `<server>/<origin>` when the server tags it) |

For Python tools, `function_tool()` builds the JSON schema from the type hints
and Google-style docstring (`transformers.utils.get_json_schema`), so the
docstring *is* what the model reads — keep the description and `Args:` clear.
Sync functions are run with `asyncio.to_thread`.

Names in `tools.disabled` are skipped at registration. Duplicate names are an
error, except for MCP tools, which get renamed to `<server>__<tool>` on a clash.

## How MCP is configured

MCP servers are listed in `config.yaml`. Each entry needs **exactly one**
transport:

```yaml
mcp_servers:
  local_notes:                       # stdio: we spawn the process
    command: python
    args: ["-m", "mcp_servers.notes_server"]
    env: { NOTES_FILE: "${NOTES_FILE}" }
  some_remote:                       # Streamable HTTP: we connect to a URL
    url: https://example.com/mcp
    headers: { Authorization: "Bearer ${SOME_API_KEY}" }
    enabled: false                   # listed but not started
```

- `${VAR}` in `args`, `env`, `url` and `headers` is filled from the environment
  (put secrets in `.env`). An unset variable makes only that server fail.
- A stdio server only sees the variables listed under its `env:`.
- `enabled: false` keeps the entry but reports it as `disabled`.

There are two lists:

- **Top-level `mcp_servers`** — what the agent process connects to. By default
  that is a single `sandbox` entry pointing at the tool server on
  `http://127.0.0.1:8001/mcp`.
- **`sandbox.mcp_servers`** — what the tool server (the Docker container) runs and
  re-exposes, e.g. `local_notes`. This keeps tool code off the host.

> The tool server and Docker setup are Task 13 in `tasks.md` and not built yet.
> Today the CLI registers only the in-process Python tools.

### What happens at connect time

`connect_mcp_servers()` in `agent_bot/mcp_tools.py`, per enabled server:

1. Opens the transport (`stdio_client` or `streamable_http_client`) and an MCP
   `ClientSession`, with a 60 s connect timeout.
2. Calls `session.list_tools()` and wraps each tool with `_wrap_tool`: the MCP
   `inputSchema` becomes the `Tool.parameters`, and `func` forwards to
   `session.call_tool()`. Text content is joined; images and other blocks become
   placeholders like `[image omitted]`. An MCP error result is raised as
   `ToolError`, which the registry turns into an `is_error` result.
3. Registers the tools and returns a status per server
   (`connected` / `failed` / `disabled`). A failing server is logged and skipped;
   it never stops the app.

From the agent loop's point of view, an MCP tool and a Python tool are the same.

### Adding a tool

- **Python:** write a function with type hints and a docstring with `Args:` in
  `agent_bot/tools.py`, and decorate it with `@tool`.
- **MCP:** add an entry under `mcp_servers` (or `sandbox.mcp_servers`). The bundled
  `mcp_servers/notes_server.py` is a minimal example of writing your own.
