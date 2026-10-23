# agent_bot — Design & Build Plan

*Written 2026-10-22. Revised 2026-10-23: the agent and inference move to the Mac host (MLX on the Apple GPU), and Docker becomes the tool sandbox. Status: approved design; revision 2 pending review.*

## 1. What this is, and why

A **local, customizable AI agent** for a Mac. The user types a task into a small desktop window ("save a note that I like tea", "search for the latest Python release and summarize it", "text Alex that I'm running late"). A small Hugging Face model running natively on the Mac's GPU decides which tools to call. The tools run inside a Docker container (Python functions and MCP servers), and the model answers using their results. The window shows every step: the model's **thinking**, each **tool call**, each **tool result**, and the final **response**.

**Why:** it's a learning project. The goal is to understand, by building it, how real agents work: the agent loop, tool schemas, tool-call parsing, and MCP. It should also be a proof of concept that does basic tasks for its owner. It mirrors the architecture of production agents (Claude Code, OpenAI Agents SDK, smolagents) at a scale one person can read end to end. That includes their split between where the model runs and where tools run: the model runs where the compute is, and the tools run in a sandbox.

**Why the revision:** Docker on a Mac cannot use the Apple GPU. Measured on this Mac (M2 Pro, 16 GB) with Qwen3-1.7B:
- MLX on the host: **47 tok/s**
- `transformers` on CPU, float32: 4.6 tok/s
- `transformers` on CPU, bf16: 0.33 tok/s

Inference performance matters more than having the agent itself inside Docker, which the owner confirmed is not a requirement. So inference and the agent loop run natively on the Mac. Docker keeps what actually benefits from isolation: the tools, which touch the network and files and run third-party `uvx`/`npx` packages.

**Later:** the agent may move to a cloud GPU machine with an Unsloth or `transformers` backend. The tool sandbox doesn't need to move. The design must keep inference behind the `Backend` interface so that move is a backend swap.

## 2. Goals, non-goals, anti-goals

### Goals
1. Runs fully locally on a Mac, with no paid API keys required:
   - inference on the Apple GPU via MLX;
   - tools in a Docker container.
2. **Customizable in three ways, each by touching one place:**
   - **Add a Python tool:** write a decorated function in `tools.py`, then `docker compose restart`. Nothing else changes.
   - **Add an MCP server:** add an entry to `config.yaml` (plus any secrets in `.env`). Entries under `sandbox.mcp_servers` run inside the container (stdio via `python`, `uvx` or `npx`, or remote Streamable HTTP). Servers that need the user's own Mac files also need a volume mount in `docker-compose.yml`; the README says so.
   - **Swap the model:** download different Hugging Face weights and change `model.path` in `config.yaml`.
3. The agent loop is **hand-written and readable** (no agent framework). A reader can follow one request from GUI → loop → model → tool sandbox → back.
4. A tkinter GUI shows thinking, tool calls, tool results and the final answer as visually distinct parts, streamed as they happen.
5. Inference sits behind a small backend interface:
   - `mlx` (Apple GPU, the default)
   - `transformers` (CPU fallback and future cloud use)
   - `unsloth` (CUDA, later)

   They're interchangeable through config. **Our code renders the prompt and parses the output for every backend**; no backend hides that step.

### Non-goals (for this version)
- Multi-user use, accounts, auth, or exposing anything beyond `localhost`.
- Running the agent or the model inside Docker.
- Fine-tuning or training.
- Long-term memory across app restarts, RAG or vector stores.
- Multi-agent orchestration, sub-agents, planning/reflection layers.
- Implementing the Unsloth backend now. Only the interface slot is in scope (see §11).
- Using an OpenAI-compatible inference server (Docker Model Runner, `mlx_lm.server`, llama.cpp server) as the backend. They build prompts and extract tool calls themselves, which defeats goal 5.
- Production hardening (rate limits, metrics, horizontal scaling).

### Anti-goals (would make this a failure even if it "works")
- **A black box.** If the loop is hidden inside LangChain, smolagents or similar, or prompt rendering and tool-call parsing happen inside an inference server, the learning goal fails.
- **Tool plumbing scattered across files.** Adding a tool must not mean editing the loop, the parser and the GUI.
- **Model-specific hacks in the loop.** Anything specific to one model family (prompt format, tool-call syntax, `<think>` tags) lives in the parser/backend layer, not in `agent.py`.
- **Silent failures.** A malformed tool call, tool exception, timeout or an unreachable sandbox must be visible in the GUI and fed back to the model, never swallowed or crashing the app.

## 3. Constraints and facts the design rests on

- **Docker on a Mac has no GPU access**, neither NVIDIA nor Metal. Model Runner and vllm-metal get around this by running inference as host processes, not in containers.
- **MLX runs only natively on macOS.** `mlx-lm` 0.32 loads the original Qwen3 safetensors in `models/` directly (no conversion). Its tokenizer passes `tools=` and `enable_thinking=` through to the chat template. Verified on 2026-10-23: the model emitted a correct `<think>` + `<tool_call>` for a `get_current_time` tool at 47 tok/s, with 3.7 GB peak memory. Generating on a thread other than the one that loaded the model also works.
- **Unsloth requires an NVIDIA (CUDA) GPU.** It is not usable on the Mac.
- **On CPU, `transformers` must use float32.** bf16 is ~14× slower on Mac CPUs.
- **Small models are unreliable at tool calling**, especially across multiple turns: most sub-4B models score under 50% on BFCL multi-turn. The design must tolerate bad tool calls (§6.8) rather than assume correctness.
- **The MCP Python SDK is async**, while generation is synchronous. The agent server bridges the two (§6.9).
- **Naming:** a local file called `mcp.py` shadows the `mcp` package. The MCP module is `mcp_tools.py`.
- **The host's Python:** Homebrew `python3` is 3.14 without tkinter. The host venv uses the python.org **3.12** at `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`, which has tkinter 8.6.

## 4. Architecture

```
 ┌──────────────────────── Mac (native, Apple GPU) ─────────────────────────┐        ┌──────── Docker container: tool sandbox ────────┐
 │                                                                            │        │                                                 │
 │  gui/app.py (tkinter)          HTTP / NDJSON        agent server           │        │  sandbox/tool_server.py                         │
 │   prompt, transcript, status ◀───────────────▶  agent_bot/server.py       │        │   one MCP server over Streamable HTTP           │
 │                                 127.0.0.1:8000      │                      │  MCP   │   at 0.0.0.0:8001/mcp (published 127.0.0.1)     │
 │                                                     ▼                      │ over   │        │                                        │
 │                                               agent.py — THE LOOP  ───────────HTTP──▶│   ToolRegistry (same class as host)            │
 │                                                │            │              │ :8001  │    ├ tools.py  (Python tools)                   │
 │                                                ▼            ▼              │        │    └ mcp_tools.py → in-container MCP servers    │
 │                                          inference.py   registry.py        │        │        (notes_server, uvx/npx entries)          │
 │                                           MLXBackend     host registry:    │        │                                                 │
 │                                           Transformers…  tools from the    │        │  /data volume: notes.json                       │
 │                                                │         sandbox (MCP)     │        └─────────────────────────────────────────────────┘
 │                                           parsing.py                       │
 │  ./models  (weights)   ./data (traces, evals)                              │
 └────────────────────────────────────────────────────────────────────────────┘
```

**Two processes on the host, one container.** The GUI and the agent server are separate processes, so a model crash or memory spike doesn't take the window down. The GUI stays a thin HTTP client and works unchanged against a future cloud agent server.

**Responsibilities (one job per file):**

| File | Runs on | Job | Must not |
|---|---|---|---|
| `agent_bot/server.py` | host | HTTP API, startup/shutdown, session store, streaming, per-turn sandbox connection | contain loop logic |
| `agent_bot/agent.py` | host | The ReAct loop: call model, run tools, append results, repeat, emit events | know prompt formats, transports or which tools exist |
| `agent_bot/inference.py` | host | `Backend` interface: `MLXBackend`, `TransformersBackend` (+ the `unsloth` slot). Messages + tool schemas in, raw text out (streamed) | parse tool calls or run tools |
| `agent_bot/parsing.py` | host | Turn raw model text into `{thinking, tool_calls, content}` for a model family | call the model |
| `agent_bot/registry.py` | both | `Tool`, `ToolRegistry` (register, list schemas, call by name with timeout) | know about specific tools |
| `agent_bot/mcp_tools.py` | both | MCP **client**: connect to configured servers and wrap their tools as `Tool`s | — |
| `agent_bot/tools.py` | sandbox (host only with `local_python_tools`) | Built-in Python tools | import from `agent.py` |
| `sandbox/tool_server.py` | sandbox | MCP **server**: expose the sandbox's whole registry as one MCP server | import `inference`, `agent`, `mlx` or `torch` |
| `mcp_servers/notes_server.py` | sandbox | Example stdio MCP server | — |
| `agent_bot/config.py` + `config.yaml` | both | All settings in one file | — |
| `gui/app.py` | host | tkinter client | import anything from `agent_bot` |

## 5. The request lifecycle (one turn, end to end)

1. User types a prompt in the GUI and presses **Send** (or Cmd+Enter).
2. GUI `POST /chat {session_id, message}` to the agent server and reads the NDJSON stream on a background thread.
3. The agent server **opens a fresh MCP session to the sandbox** (`http://127.0.0.1:8001/mcp`) and lists its tools into the host registry (§6.7).
   - If the sandbox is unreachable, it emits an `error` event ("Tool sandbox unreachable at … — is the container running?") and continues the turn with no sandbox tools.
4. It appends the user message to the session history and runs `agent.run(history)`, which yields events.
5. **Loop iteration** (`agent.py`):
   1. `backend.generate(messages, tool_schemas)` streams raw text on the GPU. Chunks are emitted as `token` events.
   2. `parser.parse(raw_text)` → `thinking`, `tool_calls[]`, `content`.
   3. Emit `thinking` (if any). Append the assistant message (with `tool_calls`) to history.
   4. If there are **no tool calls**, emit `answer` and stop.
   5. Otherwise, for each call: emit `tool_call`, then run it through the registry. For a sandbox tool, that is an MCP `call_tool` over HTTP; the sandbox runs it in its own registry with its own timeout. Emit `tool_result` (or `is_error: true`) and append `{"role":"tool",…}` to history.
   6. Go to 5.1. Stop with an `error` event if `max_iterations` is reached.
6. Emit `done` with timing stats. The sandbox MCP session closes. The GUI re-enables the input.

## 6. Components in detail

### 6.1 Model and weights

- **Default model: `Qwen/Qwen3-1.7B`** (original bf16 safetensors in `./models/Qwen3-1.7B`, ~3.4 GB). It is in the hybrid thinking family (`enable_thinking`), and its template has native `<tool_call>{json}</tool_call>` support.
- **Upgrade path:** `Qwen/Qwen3-4B` (same family and format). With MLX on 16 GB it runs at bf16 (~8 GB) or as an 8-bit MLX conversion (`mlx_lm.convert --hf-path Qwen/Qwen3-4B -q --q-bits 8`, ~4.3 GB). Note that `Qwen3-4B-Instruct-2507` is a *non-thinking* variant.
- **Other families** (Gemma, Phi, Llama) need their own parser in `parsing.py` (§6.4).
- **Download:** `scripts/download_model.py <repo_id>` → `./models/<repo_name>/` (safetensors + tokenizer + config). `HF_TOKEN` is optional, for gated models.
- Weights live only on the host. The sandbox never sees them.

### 6.2 `inference.py` — backends

```python
class Backend(Protocol):
    def generate(self, messages: list[dict], tools: list[dict], *, enable_thinking: bool,
                 max_new_tokens: int, temperature: float) -> Iterator[str]: ...
    def render_prompt(self, messages: list[dict], tools: list[dict], *, enable_thinking: bool) -> str: ...
    model_family: str; model_name: str; last_token_count: int
```

- **`MLXBackend`** (default, `model.backend: mlx`):
  - `mlx_lm.load(model.path)`.
  - The prompt comes from the returned tokenizer's `apply_chat_template(messages, tools=…, add_generation_prompt=True, enable_thinking=…, tokenize=False)`, the same call as `transformers`.
  - Streams with `mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=…, sampler=make_sampler(temp, top_p, top_k))`, yielding each response's `.text`.
  - `last_token_count` = the final response's `generation_tokens`.
  - **One dedicated inference thread.** MLX keeps its GPU streams per thread, so loading the model and every generation must run on the same thread. The backend owns a `ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")` for its whole life: `__init__` loads the model through it, and `generate` submits a producer that runs `stream_generate` there and puts each chunk on a `queue.Queue`. The caller's thread only reads that queue, so `generate()` can be called from any thread (an asyncio worker, a pool thread, the CLI's main thread). The single worker also makes generations one at a time, so no separate lock is needed.
  - Errors on the MLX thread are put on the queue and re-raised in the caller, never hung or swallowed.
  - This per-thread requirement is MLX-specific, so it lives in `MLXBackend` only. `agent.py` and the server stay unaware of it.
  - Imports `mlx`/`mlx_lm` lazily, on the MLX thread.
- **`TransformersBackend`** (already built): CPU, **float32** default. It's kept as a fallback and for the future cloud.
- **`unsloth`**: config value reserved. `create_backend` raises `NotImplementedError("unsloth backend not implemented yet — see plan §11")`.
- **Sampling:** `temperature` 0.6, `top_p` 0.95, `top_k` 20 (Qwen3's recommended thinking settings), from config.

### 6.3 Message format (internal)

OpenAI/HF-style dicts, as HF chat templates expect:

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [
    {"type": "function", "function": {"name": "web_search", "arguments": {"query": "..."}}}]}
{"role": "tool", "name": "web_search", "content": "...result text..."}
```

- **Thinking is never kept in history.** It's shown in the GUI only.
- **The system prompt** comes from `prompts/system.md` (with a `{today}` placeholder). It's prepended to the messages sent to the backend, never stored in history.

### 6.4 `parsing.py` — model output → structure *(built)*

It produces `ParsedOutput(thinking, tool_calls, content, parse_errors)`, with `get_parser(model_family)` → `Qwen3Parser`. It is lenient but explicit: anything unparseable goes into `parse_errors`, including an unclosed `<tool_call>` or `<think>`.

### 6.5 `registry.py` — one tool abstraction *(built; small additions)*

`Tool(name, description, parameters, func, source)`, `ToolResult(content, is_error, duration_ms)` and `ToolRegistry` (register, schemas, call with jsonschema validation, timeout, truncation, `disabled` list). Additions for this revision:
- **`ToolError(Exception)`:** when a tool raises it, the registry uses its message **verbatim** as the error content, with no `"Tool 'X' failed: Type:"` prefix. MCP-backed tools raise it, so a sandbox error reads the same on the host as in the sandbox, without double-wrapping.
- **`Tool.origin`** (the MCP server a tool came from; `None` for in-process Python tools) and **`replace_origin(origin, tools)`**, which atomically swaps every tool from one server. The agent server uses it to refresh sandbox tools each turn. The display label `source` stays separate.

### 6.6 `tools.py` — Python tools *(built; runs in the sandbox)*

`web_search` (DuckDuckGo via `ddgs`), `fetch_page`, `get_current_time` and `calculator`, registered with `register_python_tools(registry)`. They run **in the container**. The host registers them in-process only when `agent.local_python_tools: true`, which is for development and running without Docker.

`fetch_page` tries plain HTTP + trafilatura first. When that yields no readable text (JavaScript-rendered pages such as Workday or React careers sites), it falls back to rendering the page in **headless Chromium** (Playwright) and extracting text from the rendered HTML. The model never drives the browser: no clicking or typing tools, just a better fetcher. If Playwright isn't installed (e.g. the host with `local_python_tools`), the fallback is skipped and the original error is returned.

### 6.7 MCP — client on both sides, server in the sandbox

**Config:**

```yaml
mcp_servers:              # what the AGENT (host) connects to
  sandbox:
    url: http://127.0.0.1:8001/mcp
sandbox:                  # what the TOOL SERVER (container) runs
  host: 0.0.0.0
  port: 8001
  python_tools: true
  mcp_servers:
    local_notes:
      command: python
      args: ["-m", "mcp_servers.notes_server"]
      env: { NOTES_FILE: "${NOTES_FILE}" }
    fetch:
      command: uvx
      args: ["mcp-server-fetch"]
      enabled: false
    some_remote:
      url: https://example.com/mcp
      headers: { Authorization: "Bearer ${SOME_API_KEY}" }
      enabled: false
```

**`mcp_tools.py` (client, used on both sides):**
- An entry with `command` → stdio. An entry with `url` → Streamable HTTP. Both end in a `ClientSession` → `initialize()` → `list_tools()`.
- Each tool becomes a `Tool` whose `func` calls `call_tool`. It raises `ToolError` with the flattened text when `isError` is set or the transport fails.
- `${VAR}` substitution applies to `env`, `args` and `headers`, from the process environment (`.env` in the container).
- A failing server is skipped with a `failed` status, and the process keeps running.
- On a name clash, the second tool is registered as `<server>__<tool>`.
- **Lifetimes:**
  - **stdio servers** are children of the process. They are connected once at startup and stay open (the sandbox's notes server).
  - **URL servers on the host** (the sandbox) are connected **per turn**: open a session, `replace_origin(...)` with the fresh tool list, run the turn, close. This makes `docker compose restart` (the way you add a tool) work with no reconnection logic. A new tool shows up on the next turn. The cost is a localhost handshake, a few ms, against seconds of generation.
- **Tool source labels:** the sandbox reports each tool's original source (`python` or `mcp:local_notes`) in the MCP tool's `_meta` (`{"agent_bot/source": …}`). The host shows `sandbox/<source>` (e.g. `sandbox/python`), falling back to `mcp:sandbox`. The GUI keeps distinguishing the Python and MCP approaches.

**`sandbox/tool_server.py` (server):**
- At startup, it builds a `ToolRegistry` from `register_python_tools` (if `sandbox.python_tools`) plus `connect_mcp_servers(sandbox.mcp_servers)`.
- It exposes the registry with the MCP SDK's low-level `Server`: `list_tools` → `registry.tools()` as MCP `Tool`s, and `call_tool` → `registry.call(...)`, returning `CallToolResult(content=[TextContent], isError=result.is_error)`.
- It's served over Streamable HTTP in **stateless** mode with uvicorn on `sandbox.host:port`, path `/mcp`.
- It also serves `GET /health` → `{"status": "ok", "tools": N, "mcp_servers": [statuses]}`.
- Timeouts use `tools.timeout_s` in the sandbox registry. The host's MCP call timeout is `tools.timeout_s + 5`, so the sandbox's own timeout message is the one the model sees.
- **Bundled example server** `mcp_servers/notes_server.py` (FastMCP): `add_note`, `list_notes` and `search_notes`, persisted to `$NOTES_FILE` (`/data/notes.json`). It **reads the file on every call**, so the eval can reset it from the host via the `./data` bind mount.

### 6.7a Mac apps — `mcp_servers/mac_server.py` on the host

Controlling Mac apps (Notes, Reminders, Messages, Mail, Shortcuts) can't happen in the Linux container, so this MCP server runs **on the host** as a stdio child of the agent server. It is a top-level `mcp_servers` entry, connected once in the agent server's lifespan (§6.7):

```yaml
mcp_servers:
  sandbox: { url: http://127.0.0.1:8001/mcp }
  mac:
    command: .venv/bin/python          # relative to the repo root (the server's cwd)
    args: ["-m", "mcp_servers.mac_server"]
```

**Mechanisms**, all through `subprocess.run` with an argument list (never a shell) and a 15 s timeout:
- **AppleScript via `osascript`** for apps with a scripting dictionary (Notes, Reminders, Messages, Mail). User text is passed as `argv` to an `on run argv` script, never spliced into the script source, so quotes and newlines can't break or inject AppleScript.
- **`open -a`** to bring an app to the front.
- **`shortcuts run`** for anything else the user builds in the Shortcuts app.

**Tools:**

| Tool | Does | Mechanism | Needs approval |
|---|---|---|---|
| `create_apple_note(title, body)` | New note in the default Notes account | AppleScript; body HTML-escaped, newlines → `<br>` | no |
| `search_apple_notes(query, max_results=5)` | Titles + first 200 chars of matching notes | AppleScript `notes whose name contains … or plaintext contains …` | no |
| `add_reminder(title, due="")` | Reminder in the default list; `due` is ISO `YYYY-MM-DD[THH:MM]` | AppleScript | no |
| `open_app(name)` | Brings an app to the front | `open -a <name>` | no |
| `list_shortcuts()` | Names of the user's Shortcuts | `shortcuts list` | no |
| `send_imessage(recipient, text)` | Sends an iMessage to a phone number or Apple ID email | AppleScript: `send … to participant … of (1st account whose service type = iMessage)` | **yes** |
| `send_email(to, subject, body)` | Sends an email from Mail's default account | AppleScript: `make new outgoing message`, add a `to recipient`, `send` | **yes** |
| `run_shortcut(name, input="")` | Runs a Shortcut and returns its text output | `shortcuts run <name> --input-path … --output-path …` | **yes** (a Shortcut can do anything) |

Tool names avoid the notes server's `search_notes`/`add_note` so the model can tell "the agent's own notes" from Apple Notes.

**Safety rule: nothing leaves the Mac without the user's yes.** Send tools are available to the model, but every call goes through the approval step in §6.8a first: the user sees the exact recipient and text and approves or declines. There is no keystroke simulation, which would need the Accessibility permission and break whenever an app's UI changes.

**Permissions:** the first `osascript` call to each app triggers macOS's "… wants to control Notes" prompt for the process that started the agent server (Terminal, iTerm, VS Code…). A refusal comes back as AppleScript error `-1743`, which the tool turns into a message telling the user where to enable it (System Settings → Privacy & Security → Automation).

### 6.8 `agent.py` — the loop

The loop is as in §5. Its contract is unchanged from revision 1:
- `max_iterations` (default 6) and per-tool timeouts.
- Parse failures fed back as a user-role correction message.
- A repeat guard: the 3rd identical call stops the turn.
- Sequential tool execution.
- `done` is always last.
- Tools listed in `tools.require_approval` wait for the user's approval (§6.8a).

`agent.py` must not import `transformers`, `torch`, `mlx`, `mcp` or any concrete tool module. It is ≤ 200 lines and thoroughly commented.

### 6.8a Approval before send tools

Tools that act on the user's behalf outside the Mac (sending a message or an email, running a Shortcut) need the user's approval for **every call**. The model may call them freely; the loop asks the user before running one.

- **Which tools:** `tools.require_approval` in `config.yaml`, a list of tool names. Default: `[send_imessage, send_email, run_shortcut]`. Users can add any tool (e.g. a sandbox MCP tool that posts somewhere).
- **The loop** (`agent.py`): for a call whose name is in the list, after the `tool_call` event it emits an **`approval_request`** event `{id, name, arguments}` and awaits `settings.approve(id, name, arguments) -> bool`, a callback the caller provides.
  - Approved → the tool runs as normal.
  - Declined, timed out (`tools.approval_timeout_s`, default 120) or no callback (the eval, scripts) → the tool does **not** run. The model gets a normal, non-error tool result: `"The user declined to run <name>. Don't retry it unless they ask; ask what they'd like to change."`
  - Either way an `approval_result` event `{id, approved}` follows, so the transcript and the trace record the decision.
- **The system prompt** tells the model: before a send tool, make sure the recipient and the exact text are what the user asked for. The app shows the user the call and asks for approval, so the model doesn't need to ask in chat first. If declined, don't retry; ask what to change.
- **CLI:** prints the call and asks `Approve? [y/N]` on stdin (read with `asyncio.to_thread(input)`).
- **Agent server:** `/chat` streams `approval_request` like any event and waits on a per-call future. **`POST /approve {session_id, call_id, approved}`** resolves it (404 for an unknown or finished call). A client disconnect mid-wait counts as declined.
- **GUI:** an `approval_request` shows the tool name, the recipient/subject/text in full, and **Approve** / **Decline** buttons, which post to `/approve`. Input is disabled until the user decides.

### 6.9 `server.py` — agent HTTP API (host)

FastAPI + uvicorn on **`127.0.0.1:8000`** (native, not in Docker).

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{status, model, backend, model_loaded}`. The model loads in a background thread at startup. |
| `GET /tools` | Opens a sandbox session on demand. Returns tools (name, description, source) and statuses for the sandbox and any host-side servers. |
| `POST /chat` `{session_id, message, enable_thinking?}` | NDJSON event stream. 503 while the model is loading, 409 while another turn runs. |
| `POST /approve` `{session_id, call_id, approved}` | Answers a pending `approval_request` (§6.8a). 404 if there is none with that id. |
| `POST /reset` `{session_id}` | Clears that session's history. |

- Sessions are in memory. Generation runs on a worker thread and streams into the async event stream.
- The lock is released in the stream's `finally`, which also runs on client disconnect.
- The **event schema**: `token`, `thinking`, `tool_call`, `approval_request`, `approval_result`, `tool_result`, `answer`, `error`, `done`.

### 6.10 `gui/app.py` — tkinter client (host)

Unchanged from revision 1:
- Top bar: status, Thinking checkbox, New chat, Tools window.
- The transcript, with a distinct style per event kind.
- A live "generating…" area for tokens.
- An input with Send / Cmd+Enter and an elapsed timer.
- Threads hand events over a queue that the Tk loop polls; Tk widgets are touched only from the main thread.
- `AGENT_URL` defaults to `http://127.0.0.1:8000`.

The Tools window also shows sandbox status (connected / unreachable).

### 6.11 Configuration

One `config.yaml` is read by both sides; each reads only its own sections. Env overrides use `AGENT__SECTION__KEY`.

```yaml
model:
  path: models/Qwen3-1.7B   # host-relative now
  family: qwen3
  backend: mlx              # mlx | transformers | unsloth
  dtype: float32            # transformers only (bf16 is ~14x slower on Mac CPU)
  num_threads: null         # transformers only
generation:
  max_new_tokens: 1024
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  enable_thinking: true
agent:
  max_iterations: 6
  system_prompt_file: prompts/system.md
  local_python_tools: false # true = run tools.py in-process on the host (no Docker)
tools:
  timeout_s: 20
  max_result_chars: 4000
  disabled: []
  require_approval: [send_imessage, send_email, run_shortcut]
  approval_timeout_s: 120
mcp_servers: { sandbox: { url: http://127.0.0.1:8001/mcp } }
sandbox: { ... see §6.7 ... }
server:
  host: 127.0.0.1
  port: 8000
logging:
  level: INFO
  trace_dir: data/traces
```

### 6.12 Logging and traces

Standard `logging` to stdout. One JSONL trace per turn in `data/traces/` (host), with the rendered prompt, raw output, parsed output and tool calls/results.

## 7. Docker — the tool sandbox

### 7.1 Image
- `python:3.12-slim-bookworm`, with **no torch, no transformers model code and no weights**. `requirements-sandbox.txt` is a subset: `mcp`, `httpx`, `ddgs`, `trafilatura`, `jsonschema`, `pydantic`, `pyyaml`, `uvicorn`, `starlette`, `playwright`, plus `transformers` for `get_json_schema` only. `transformers` imports without torch.
- **Headless Chromium** for the `fetch_page` fallback: `playwright install --with-deps chromium` at build time (as root, before switching user), with `PLAYWRIGHT_BROWSERS_PATH=/ms-playwright` so the non-root user can run it. Adds roughly 1 GB to the image. Playwright launches Chromium without its own sandbox by default; the container is the sandbox.
- Includes Node.js LTS (for `npx`) and `uv` (for `uvx`). Runs as a non-root user. Builds for `linux/arm64` and `linux/amd64`.
- `CMD ["python", "-m", "sandbox.tool_server"]`.

### 7.2 `docker-compose.yml`
```yaml
services:
  tools:
    build: .
    ports: ["127.0.0.1:8001:8001"]
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./agent_bot/tools.py:/app/agent_bot/tools.py:ro   # add tools without rebuilding
      - ./mcp_servers:/app/mcp_servers:ro
      - ./data:/data
    env_file: [{ path: .env, required: false }]
    environment:
      - NOTES_FILE=/data/notes.json
    shm_size: 1gb            # Chromium crashes with Docker's 64 MB /dev/shm default
    deploy: { resources: { limits: { memory: 2g } } }
```

### 7.3 Host requirements
- **Mac:** M2 Pro (12 cores, 16 GB) with **Docker Desktop**, context `desktop-linux`. Rancher Desktop is installed but unused, and its `~/.rd/bin/docker` may come first on `PATH`; the README says to check `docker context ls`.
- **Host venv:** python.org Python 3.12 with tkinter. `requirements.txt` covers the agent server, GUI and tests: `mlx-lm`, `transformers`, `fastapi`, `uvicorn`, `mcp`, … and torch optionally, for the CPU backend.
- **Memory:** the model uses the Mac's unified memory (3.7 GB peak for 1.7B). The container needs ≤ 2 GB, including one headless Chromium page (~300–500 MB) while `fetch_page` falls back to it.
- `docker compose restart` picks up changes to `tools.py`, `config.yaml` and `mcp_servers/` without a rebuild. A new pip dependency for a tool needs `docker compose up --build`.

### 7.4 Speed
Measured: 47 tok/s generation with Qwen3-1.7B on MLX, so a tool-using turn with thinking (2 model calls, ~300–600 tokens) should take ~10–20 s plus tool time. The README records the real numbers from the eval.

## 8. Repository layout

```
agent_bot/
├── plan.md  README.md  config.yaml  .env.example
├── requirements.txt            # host: agent server, GUI, tests (mlx-lm, …)
├── requirements-sandbox.txt    # container: tool server only
├── requirements-dev.txt
├── Dockerfile  docker-compose.yml  .dockerignore
├── prompts/system.md
├── agent_bot/   server.py agent.py inference.py parsing.py registry.py tools.py mcp_tools.py config.py events.py trace.py cli.py
├── sandbox/     __init__.py tool_server.py
├── mcp_servers/ notes_server.py  mac_server.py (host)
├── gui/         app.py
├── scripts/     download_model.py generate.py eval.py
├── evals/       prompts.yaml
├── models/  data/              # git-ignored
└── tests/
```

## 9. Testing

- **No real model, network or Docker** is needed for the default `pytest` run. A `FakeBackend` scripts model output for loop and server tests.
- The tool server is tested **in-process**: start `sandbox.tool_server` on a free local port in a test fixture, then connect with `mcp_tools` over Streamable HTTP. The same test proves both MCP directions.
- `-m slow` tests load the real model (MLX) and run on the host.
- The GUI is tested manually (checklist in the README).

### 9.1 Reliability target (the proof-of-concept bar)

5 fixed prompts, each run 3 times. **Target: at least 4 of 5 prompts pass in at least 2 of 3 runs.** The prompts and checks:

| # | Prompt | Must call | Answer check |
|---|---|---|---|
| 1 | "Read https://peps.python.org/pep-0020/ and tell me the first line of the Zen of Python." | `fetch_page`, url ∋ "pep-0020" | contains "Beautiful is better than ugly" |
| 2 | "Search the web for the latest stable Python release and tell me its version number." | `web_search` | a `3.x` version that appears in a tool result |
| 3 | "Save a note that my dentist appointment is Friday at 10am, then list all my notes." | `add_note` then `list_notes` | mentions "dentist" |
| 4 | "What is 17.5% of 2,340?" | `calculator` | contains `409.5` |
| 5 | "What time is it in Tokyo right now?" | `get_current_time` with `Asia/Tokyo` | a time within ±2 min |

- `scripts/eval.py` runs **on the host** against the real backend and the running sandbox. It needs the container up and checks `http://127.0.0.1:8001/health` first.
- For prompt 3 it backs up `data/notes.json`, writes `[]`, and restores the file afterwards.
- Network failures count as `infra` and are re-run.
- **Tuning levers, in order:**
  1. prompt and tool wording
  2. thinking on vs off
  3. lower temperature
  4. switch to `Qwen3-4B` (now cheap with MLX)

## 10. Build order (milestones)

Done: (1) skeleton/config, (2) transformers backend, (3) parser, (4) registry, (5) Python tools.

6. **Revision amendments:**
   - config gets the `mlx` backend, the `sandbox` section, `local_python_tools` and host defaults;
   - the registry gets `ToolError`, `Tool.origin` and `replace_origin`;
   - requirements split.
7. **MLX backend.** ✅ when `scripts/generate.py` streams from MLX at tens of tok/s.
8. **Events + traces.**
9. **Agent loop + CLI** (with `--local-tools` to run without Docker). ✅ when the real model calls a tool and answers.
10. **Agent server** on the host. ✅ when `curl -N` shows the event stream.
11. **GUI.** ✅ when a full turn renders all event kinds (using `local_python_tools: true` until the sandbox exists).
12. **MCP client + notes server.**
13. **Tool sandbox:** `tool_server.py` + Docker, with the host connecting per turn. ✅ when GUI turns use container tools, and `docker compose restart` after adding a tool makes it appear on the next turn.
14. **Reliability eval** + tuning. ✅ when ≥ 4/5 prompts pass ≥ 2/3 runs.
15. **README, smoke test, polish.**
16. **Mac apps MCP server** on the host (§6.7a). ✅ when a GUI turn creates an Apple Note, and `send_imessage` runs only after the user clicks Approve.

## 11. Later: cloud GPU

- Run the **agent server** on a cloud GPU machine, using `TransformersBackend` with `device_map="cuda"`, or a new `UnslothBackend` (`FastLanguageModel.from_pretrained` + `for_inference`, then the same `apply_chat_template` + streamer flow).
- The GUI points at it with `AGENT_URL`. The tool sandbox either moves with it (same image) or stays local behind a tunnel.
- At that point both servers need auth (an API key header at minimum) and must not be published on `0.0.0.0` without it.
- Nothing in `agent.py`, `parsing.py`, `registry.py`, the tools or the GUI changes.

## 12. Decisions log

**Made by the user:**
- Hugging Face model with locally downloaded weights; an inference backend runs it.
- Separate inference, tools and MCP modules; tools both as Python functions and via MCP.
- tkinter GUI showing tool calls, thinking and response, on the Mac host.
- Learning project, customizable by adding tools and swapping models; a hand-written agent loop.
- Docker Desktop (not Rancher Desktop).
- A reliability target on 5 fixed prompts.
- **Revision 2:** inference performance matters more than the agent running in Docker. The inference and agent loop move to the host, next to the GUI. Tool calls and results pass over HTTP to the containerized tools.

**Made by Claude (check these):**
- MLX (`mlx-lm` Python API) as the default backend, not an OpenAI-compatible server (Docker Model Runner, `mlx_lm.server`), so prompt rendering and parsing stay in our code.
- `MLXBackend` runs model loading and all generation on one dedicated thread it owns (a single-worker `ThreadPoolExecutor`), handing chunks to the caller through a queue. No custom thread classes, and no thread awareness outside the backend.
- The sandbox is exposed to the host as **one MCP server over Streamable HTTP** (stateless), reusing `mcp_tools.py`, rather than a custom tool API.
- **Lifetimes:**
  - host ↔ sandbox sessions open **per turn** (survives container restarts, picks up new tools);
  - stdio servers inside the sandbox stay open for the process lifetime.
- The agent server and GUI are separate host processes.
- `ToolError` for verbatim error passthrough; `Tool.origin` + `replace_origin` for refreshing sandbox tools.
- Tool source carried in MCP `_meta` and shown as `sandbox/<source>`.
- `agent.local_python_tools` flag for running without Docker.
- Mac app control as a separate **host-side** stdio MCP server (the container can't reach Mac apps), with AppleScript text passed as `argv`.
- Send tools are allowed but gated by a per-call user approval enforced in the loop (`tools.require_approval`), not left to the model's judgement: a 1.7B model can't be trusted to always ask first.
- Headless Chromium in the sandbox only as a `fetch_page` fallback, not as a model-driven browser: a 1.7B model can't reliably run a multi-step click/read loop, and the sites worth browsing (LinkedIn, Indeed, Google) block headless browsers anyway.
- `transformers` backend kept with float32 default; Unsloth reduced to a reserved config value.
- From revision 1, still in force:
  - Default model Qwen3-1.7B.
  - DuckDuckGo search.
  - Notes as the example MCP server.
  - In-memory sessions; thinking stripped from history.
  - 6 iterations max, 20 s tool timeout, 4000-char result cap.
  - The eval prompts and pass bar.

## 13. Left to the builder

- Exact system prompt wording, GUI styling, and the HTML-to-text library for `fetch_page`.
- How `_meta` is set on MCP tools if the installed SDK version names the field differently. The fallback is the `mcp:sandbox` label.
- **Nice-to-haves:**
  - stop button
  - model switcher
  - tok/s display
  - a Gemma parser
  - a comparison script against Docker Model Runner's built-in tool-call parsing
