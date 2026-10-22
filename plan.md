# agent_bot — Design & Build Plan

*Written 2026-10-22. Status: draft for review.*

## 1. What this is, and why

A **local, customizable AI agent** that runs inside Docker on a Mac. The user types a task into a small desktop window ("what's the weather in Istanbul tomorrow?", "search for the latest Python release and summarize it"). A small Hugging Face model running in the container decides which tools to call, calls them (Python functions or MCP servers), and answers using the results. The window shows every step: the model's **thinking**, each **tool call**, each **tool result**, and the final **response**.

**Why:** it's a learning project. The goal is to understand, by building it, how real agents work: the agent loop, tool schemas, tool-call parsing, and MCP. It should also be a proof of concept that does basic tasks for its owner. It mirrors the architecture of production agents (Claude Code, OpenAI Agents SDK, smolagents) at a scale one person can read end to end.

**Later:** if CPU inference is too slow, the same container moves to a cloud GPU machine and switches to an Unsloth backend. Nothing in the design may assume it runs on a Mac.

## 2. Goals, non-goals, anti-goals

### Goals
1. Runs fully locally on a Mac (Apple Silicon, CPU only inside Docker), with no paid API keys required.
2. **Customizable in three ways, each by touching one place:**
   - **Add a Python tool:** write a decorated function in `tools.py`. Nothing else changes.
   - **Add an MCP server:** add an entry to `config.yaml` (plus any secrets in `.env`). This covers stdio servers launched with `python`, `uvx` or `npx`, and remote servers over Streamable HTTP. Servers that need the user's own Mac files also need a volume mount added to `docker-compose.yml`; the README says so.
   - **Swap the model:** download different Hugging Face weights and change `model.path` in `config.yaml`.
3. The agent loop is **hand-written and readable** (no agent framework). A reader can follow one request from GUI → loop → model → tool → back.
4. A tkinter GUI shows thinking, tool calls, tool results and the final answer as visually distinct parts, streamed as they happen.
5. Inference sits behind a small backend interface so `transformers` (CPU, now) and `unsloth` (GPU, later) are interchangeable through config.

### Non-goals (for this version)
- Multi-user use, accounts, auth, or exposing the server beyond `localhost`.
- Running the GUI inside Docker.
- Fine-tuning or training.
- Long-term memory across app restarts, RAG or vector stores.
- Multi-agent orchestration, sub-agents, planning/reflection layers.
- Implementing the Unsloth backend now. Only the interface slot and a stub are in scope (see §11).
- Production hardening (rate limits, metrics, horizontal scaling).

### Anti-goals (would make this a failure even if it "works")
- **A black box.** If the loop is hidden inside LangChain, smolagents or similar, the learning goal fails.
- **Tool plumbing scattered across files.** Adding a tool must not mean editing the loop, the parser and the GUI.
- **Model-specific hacks in the loop.** Anything specific to one model family (prompt format, tool-call syntax, `<think>` tags) lives in the parser/backend layer, not in `agent.py`.
- **Silent failures.** A malformed tool call, tool exception or timeout must be visible in the GUI and fed back to the model, never swallowed or crashing the app.

## 3. Constraints and facts the design rests on

- **Docker on a Mac has no GPU access**, neither NVIDIA nor the Apple GPU (Metal). All inference in the container is on the CPU.
- **Unsloth requires an NVIDIA (CUDA) GPU** for its Linux/Python library. It cannot be the backend on the Mac. Its Mac support is limited to its desktop Studio app for GGUF chat, which doesn't apply here.
- **tkinter can't easily display out of a Mac Docker container** (it would need XQuartz and X11 forwarding). So the GUI runs natively on the Mac and talks to the container over HTTP.
- **Small models are unreliable at tool calling**, especially across multiple turns: most sub-4B models score under 50% on BFCL multi-turn. The design must tolerate bad tool calls (§7.4) rather than assume correctness.
- **The MCP Python SDK is async** (`ClientSession`, `stdio_client`), while `transformers.generate` is synchronous and CPU-heavy. The server must bridge the two (§8).
- **Naming:** a local file called `mcp.py` shadows the `mcp` package and breaks `from mcp import ClientSession`. The MCP module is therefore named `mcp_tools.py` (deviates from the original file list on purpose).

## 4. Architecture

```
 ┌──────────────── Mac (host) ────────────────┐      ┌──────────────── Docker container (CPU) ─────────────────┐
 │                                             │      │                                                         │
 │  gui/app.py  (tkinter)                      │ HTTP │  server.py (FastAPI)                                    │
 │   - prompt box, transcript, status          │─────▶│   POST /chat  → streams NDJSON events                  │
 │   - renders events by type                  │◀─────│   GET /health, GET /tools, POST /reset                  │
 │                                             │ NDJSON│        │                                               │
 └─────────────────────────────────────────────┘      │        ▼                                                │
                                                      │  agent.py  — THE LOOP                                   │
                                                      │   messages → backend.generate → parse → run tools → … │
                                                      │        │                         │                      │
                                                      │        ▼                         ▼                      │
                                                      │  inference.py              registry.py                  │
                                                      │   Backend interface         one list of Tool objects    │
                                                      │   ├ TransformersBackend      ▲               ▲          │
                                                      │   └ UnslothBackend (stub)    │               │          │
                                                      │        │                  tools.py      mcp_tools.py    │
                                                      │        ▼                  (Python fns)  (MCP clients)   │
                                                      │  parsing.py                                 │           │
                                                      │   model output → thinking /                 ▼           │
                                                      │   tool_calls / text                 MCP servers (stdio  │
                                                      │                                     subprocesses)       │
                                                      │  /models  (volume: downloaded weights)                  │
                                                      └─────────────────────────────────────────────────────────┘
```

**Responsibilities (one job per file):**

| File | Job | Must not |
|---|---|---|
| `server.py` | HTTP API, app startup/shutdown, session store, streaming | contain loop logic |
| `agent.py` | The ReAct loop: call model, run tools, append results, repeat, emit events | know prompt formats or which tools exist |
| `inference.py` | `Backend` interface + `TransformersBackend` (+ `UnslothBackend` stub): messages + tool schemas in, raw text out (streamed) | parse tool calls or run tools |
| `parsing.py` | Turn raw model text into `{thinking, tool_calls, content}` for a model family | call the model |
| `registry.py` | `Tool` dataclass, `ToolRegistry` (register, list schemas, call by name with timeout) | know about specific tools |
| `tools.py` | Built-in Python tools, declared with a decorator | import from `agent.py` |
| `mcp_tools.py` | Connect to configured MCP servers, wrap their tools as `Tool` objects | — |
| `config.py` + `config.yaml` | All settings in one place | — |
| `gui/app.py` | tkinter client | import anything from the server package |
| `scripts/download_model.py` | Download HF weights into `./models/` | — |

## 5. The request lifecycle (one turn, end to end)

1. User types a prompt in the GUI and presses **Send** (or Cmd+Enter).
2. GUI `POST /chat {session_id, message}` and reads the NDJSON stream on a background thread.
3. Server appends `{"role":"user","content":message}` to that session's history and runs `agent.run(history)`, which yields events.
4. **Loop iteration** (`agent.py`):
   1. `backend.generate(messages, tool_schemas)` streams raw text. Token chunks are emitted as `token` events so the GUI shows progress.
   2. `parser.parse(raw_text)` → `thinking`, `tool_calls[]`, `content`.
   3. Emit `thinking` (if any). Append the assistant message (with `tool_calls`) to history.
   4. If there are **no tool calls**, emit `answer` and stop.
   5. Otherwise, for each call: emit `tool_call`, run it via the registry (with a timeout), emit `tool_result` (or a `tool_result` with `is_error: true`), and append `{"role":"tool","name":…,"content":…}` to history.
   6. Go to 4.1. Stop with an `error` event if `max_iterations` is reached.
5. Emit `done` with timing stats. GUI re-enables the input.

## 6. Components in detail

### 6.1 Model and weights

- **Default model: `Qwen/Qwen3-1.7B`** *(my call)*. Reasons: it's in the Qwen3 hybrid family, so it supports a thinking on/off switch (`enable_thinking`) for the thinking panel; it has native tool-call support in its chat template (`<tool_call>{json}</tool_call>`); and it's ~3.4 GB in bf16, so it fits Docker Desktop's memory on a typical Mac and gives usable CPU speed.
- **Upgrade path:** `Qwen/Qwen3-4B` (same family and format, ~8 GB bf16, needs Docker memory raised to ~12 GB; better tool use, slower). Note that `Qwen3-4B-Instruct-2507` is a *non-thinking* variant: it works, but the thinking panel stays empty.
- **Other families** (Gemma, Phi, Llama) need their own parser in `parsing.py` (§6.4). This is how "plug in different HF models" stays honest: same backend, a parser per format.
- **Download:** `scripts/download_model.py <repo_id>` uses `huggingface_hub.snapshot_download` into `./models/<repo_name>/` on the host (safetensors + tokenizer + config, with `allow_patterns` that skip `.bin`/GGUF/ONNX duplicates). It supports an optional `HF_TOKEN` env var for gated models.
- **Weights are mounted, not baked in.** `./models` mounts read-only at `/models` in the container. Swapping models means no image rebuild and keeps the image small.
- **dtype:** `bfloat16` on CPU by default (half the RAM of fp32; modern PyTorch CPU kernels support it on ARM). `float32` is a config option if bf16 turns out slow or broken on the builder's machine.

### 6.2 `inference.py` — backends

```python
class Backend(Protocol):
    def generate(self, messages: list[dict], tools: list[dict], *,
                 enable_thinking: bool, max_new_tokens: int,
                 temperature: float) -> Iterator[str]: ...  # yields text chunks
    @property
    def model_family(self) -> str: ...   # e.g. "qwen3", selects the parser
```

- **`TransformersBackend`**: loads `AutoTokenizer` + `AutoModelForCausalLM` from `model.path` (`device_map="cpu"`, configured dtype). The prompt comes from `tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, enable_thinking=…)`, so the model's own template formats the tools; there is no hand-written prompt. Generation runs in a worker thread with `TextIteratorStreamer` so chunks stream out. Sets `torch.set_num_threads` from config (default: all cores Docker gives).
- **`UnslothBackend`**: a stub class that raises `NotImplementedError("needs CUDA GPU — see plan §11")`, selected by `model.backend: unsloth`. It exists so the seam is real and tested.
- **Backend selection:** `config.model.backend` ∈ {`transformers`, `unsloth`}, through a small factory.
- **Stop conditions:** generation stops at EOS (the template's end-of-turn token) or `max_new_tokens`.
- **Concurrency:** one generation at a time, guarded by a lock. A second request while one is running gets HTTP 409 (single user).

### 6.3 Message format (internal)

OpenAI/HF-style dicts, which is what HF chat templates expect:

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [
    {"type": "function", "function": {"name": "web_search", "arguments": {"query": "..."}}}]}
{"role": "tool", "name": "web_search", "content": "...result text..."}
```

- **Thinking is not kept in history.** Previous turns' `<think>` blocks are dropped before re-sending, as Qwen recommends and its template does. Thinking is shown in the GUI only.
- **The system prompt** comes from `prompts/system.md`, loaded at startup. It includes today's date and short rules: use tools for current facts, cite URLs from results, answer directly when no tool is needed. *(Content is the builder's call.)*

### 6.4 `parsing.py` — model output → structure

```python
@dataclass
class ParsedOutput:
    thinking: str | None
    tool_calls: list[ToolCall]   # ToolCall(name: str, arguments: dict, raw: str)
    content: str                 # visible text with thinking/tool tags removed
    parse_errors: list[str]      # e.g. "invalid JSON in <tool_call> #2"
```

- `get_parser(model_family)` returns a parser. The first one is **`Qwen3Parser`**: it extracts `<think>…</think>`, then every `<tool_call>…</tool_call>` block parsed as JSON `{"name", "arguments"}`.
- Lenient but explicit: it tolerates surrounding whitespace, code fences inside the tag, and `arguments` given as a JSON string. Anything unparseable goes into `parse_errors`. It is never silently dropped.
- *Builder's call:* if the installed `transformers` version offers a working built-in response parser for Qwen3 (newer releases have been adding this), it may be used inside `Qwen3Parser`, keeping the same `ParsedOutput` contract.

### 6.5 `registry.py` — one tool abstraction for both kinds

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict              # JSON Schema (object)
    func: Callable[..., Awaitable[str]]   # always async, returns text
    source: str                   # "python" or "mcp:<server_name>"

class ToolRegistry:
    def register(self, tool: Tool) -> None        # error on duplicate name
    def schemas(self) -> list[dict]               # HF/OpenAI "function" format for apply_chat_template
    async def call(self, name, arguments, timeout_s) -> ToolResult  # ToolResult(content: str, is_error: bool, duration_ms)
```

- `call` validates arguments against `parameters` (`jsonschema`). On an unknown tool, invalid arguments, an exception or a timeout, it returns `is_error=True` with a clear message the model can act on. It never raises into the loop.
- Results longer than `tools.max_result_chars` (default 4000) are truncated with a `…[truncated N chars]` marker, so small context windows don't blow up.
- Tools can be **enabled/disabled in config** by name (`tools.disabled: [...]`).

### 6.6 `tools.py` — Python tools

A decorator builds the schema from the function signature and docstring:

```python
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, URLs and snippets.

    Args:
        query: What to search for.
        max_results: How many results to return (1-10).
    """
```

- *Builder's call:* use `transformers.utils.get_json_schema` (it does exactly this, Google-style docstrings → JSON schema) or a small hand-rolled equivalent. Sync functions are wrapped to run in a thread so the registry is uniformly async.
- `tools.py` exposes `register_python_tools(registry)`, which registers every decorated function in the module.

**Starter tools** *(my call; no API keys needed)*:

| Tool | What | Backed by |
|---|---|---|
| `web_search(query, max_results=5)` | Titles, URLs, snippets | `ddgs` (DuckDuckGo) package. Google can't be scraped reliably and its API needs a key. |
| `fetch_page(url, max_chars=4000)` | Readable text of a web page | `httpx` + `trafilatura` (or BeautifulSoup) |
| `get_weather(location, days=1)` | Current + forecast | Open-Meteo geocoding + forecast APIs (free, no key) |
| `get_current_time(timezone="UTC")` | Current date/time | stdlib `zoneinfo` |
| `calculator(expression)` | Safe arithmetic | `ast`-based evaluator, **no `eval`** |

The weather tool exists because "what's the weather in X" is the headline example: a direct tool gives a small model a much higher success rate than search → fetch → extract.

### 6.7 `mcp_tools.py` — MCP tools

- Reads `mcp_servers` from `config.yaml`:

```yaml
mcp_servers:
  local_notes:
    command: python
    args: ["-m", "mcp_servers.notes_server"]
    enabled: true
  fetch:
    command: uvx
    args: ["mcp-server-fetch"]
    enabled: false
  some_remote:                 # example shape, disabled
    url: https://example.com/mcp
    headers: { Authorization: "Bearer ${SOME_API_KEY}" }
    enabled: false
```

- **Two transports:** an entry with `command` uses stdio (`stdio_client`). An entry with `url` uses Streamable HTTP (`streamablehttp_client` from the MCP SDK). Both end in the same `ClientSession`, so everything after the connection is shared.
- **Secrets:** stdio entries may have an `env:` map, and both kinds allow `${VAR}` substitution in `env`, `args` and `headers`. Values come from a git-ignored `.env` file that compose loads into the container. Secrets never go in `config.yaml` directly.

- At startup, for each enabled server: `stdio_client(StdioServerParameters(...))` → `ClientSession` → `initialize()` → `list_tools()`. Each MCP tool becomes a `Tool` whose `func` calls `session.call_tool(name, arguments)` and flattens text content blocks into a string (non-text blocks become a short placeholder like `[image omitted]`).
- **Name collisions:** MCP tool names are registered as-is. If a name clashes with an existing tool, it's prefixed `<server>__<tool>`, and a warning is logged.
- Sessions stay open for the app's lifetime (an `AsyncExitStack` in the FastAPI lifespan) and close cleanly on shutdown.
- A server that fails to start is **logged and skipped**, and the app still starts. Its status shows in `GET /tools` and in the GUI.
- **Bundled example server** `mcp_servers/notes_server.py`, written with the SDK's `FastMCP`: `add_note(text)`, `list_notes()`, `search_notes(query)`, persisted to a JSON file in a `/data` volume. It's deliberately a different kind of tool from the Python ones (stateful, local), so the two approaches can be compared. This also gives the user a template for writing their own MCP servers.
- The `fetch` entry ships disabled, as an example of plugging in a third-party server. The Docker image includes `uv` and Node.js (LTS) so both `uvx` and `npx` entries work.

### 6.8 `agent.py` — the loop

```python
async def run(history: list[dict], *, backend, parser, registry, settings) -> AsyncIterator[Event]
```

- Implements §5 exactly. It is the most important file for learning, so it gets thorough comments explaining *why* each step exists. Target: ≤ 200 lines.
- **Guardrails:**
  - `agent.max_iterations` (default 6): stops with an `error` event when reached.
  - `tools.timeout_s` (default 20) per call.
  - **Parse failures:** if the output has `parse_errors` and no valid calls, append a `tool`-role (or user-role, if the template requires it) message explaining the error and loop again. This counts against `max_iterations`.
  - **Repeat guard:** an identical tool call (same name + arguments) made 3 times in a turn ends the turn with an `error` event, since small models loop.
- Multiple tool calls in one reply run **sequentially**, in order (simpler to read and debug; parallelism isn't worth it here).
- `agent.py` must not import `transformers`, `mcp`, or any concrete tool.

### 6.9 `server.py` — HTTP API

FastAPI + uvicorn, listening on `0.0.0.0:8000` inside the container and published only to `127.0.0.1:8000` on the host.

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{status, model, backend, model_loaded}`. The GUI polls it at startup (model load can take a while). |
| `GET /tools` | Registered tools: name, description, source, plus MCP server statuses |
| `POST /chat` `{session_id, message, enable_thinking?}` | Streams `application/x-ndjson`, one event per line |
| `POST /reset` `{session_id}` | Clears that session's history |

- **Sessions:** in-memory `dict[session_id, list[message]]`. The GUI generates a UUID per "conversation", and history is lost on restart (a non-goal).
- **Sync/async bridge:** model generation runs in a worker thread; token chunks pass to the async event stream via `asyncio.Queue` with `loop.call_soon_threadsafe`, or by wrapping the blocking streamer in `run_in_threadpool`. *(Mechanism is the builder's call.)*
- If the client disconnects mid-turn, generation finishes (it can't be interrupted cheaply), but no further tool calls are started. *(A stop button is a nice-to-have, §13.)*

**Event schema** (NDJSON; every event has `type` and `ts`):

```json
{"type":"token","text":"..."}
{"type":"thinking","text":"..."}
{"type":"tool_call","id":"c1","name":"web_search","arguments":{"query":"..."},"source":"python"}
{"type":"tool_result","id":"c1","content":"...","is_error":false,"duration_ms":812}
{"type":"answer","text":"..."}
{"type":"error","message":"..."}
{"type":"done","iterations":2,"elapsed_s":41.3,"tokens_generated":512}
```

`token` events let the GUI show live output. `thinking`/`answer` are the parsed, authoritative versions sent once per model call.

### 6.10 `gui/app.py` — tkinter client (runs on the Mac)

- Standard library tkinter + `requests` (or `httpx`). Started with `python gui/app.py` from a small host venv (`gui/requirements.txt`).
- **Layout:**
  - **Top bar:** connection/model status ("loading model…", "ready: Qwen3-1.7B"), "Thinking" checkbox, "New chat" button, "Tools" button (opens a list from `GET /tools`).
  - **Middle:** a scrolling read-only transcript (`tk.Text` with tags), with a distinct style per event:
    - user message: bold
    - thinking: grey italic, collapsible or dimmed
    - tool call: monospace, accent color, `→ web_search({"query": "..."})`
    - tool result: monospace, indented, first ~15 lines with a "…" marker, red if error
    - answer: normal text
    - error: red
  - **Live area:** while the model generates, streamed `token` text shows in a dimmed "generating…" block that's replaced by the parsed events when they arrive.
  - **Bottom:** a multi-line input, Send button (Cmd+Enter), and elapsed-time indicator.
- **Threading:** network reading happens on a background thread that pushes events onto a `queue.Queue`. The tk main loop polls it with `after(50, …)`. Tk widgets are never touched from the background thread.
- Server URL is configurable (`AGENT_URL` env var, default `http://127.0.0.1:8000`), which makes pointing the GUI at a cloud deployment later trivial.
- *Visual styling details are the builder's call*, as long as the six event kinds are clearly distinguishable.

### 6.11 Configuration

`config.yaml` (mounted into the container), loaded and validated by `config.py` (pydantic models). Env vars override with the `AGENT__SECTION__KEY` pattern.

```yaml
model:
  path: /models/Qwen3-1.7B
  family: qwen3            # selects parser
  backend: transformers    # transformers | unsloth
  dtype: bfloat16
  num_threads: null        # null = all available
generation:
  max_new_tokens: 1024
  temperature: 0.6
  enable_thinking: true    # default; GUI can override per request
agent:
  max_iterations: 6
  system_prompt_file: prompts/system.md
tools:
  timeout_s: 20
  max_result_chars: 4000
  disabled: []
mcp_servers: { ... see §6.7 ... }
server:
  host: 0.0.0.0
  port: 8000
logging:
  level: INFO
  trace_dir: /data/traces   # null disables
```

### 6.12 Logging and traces

- Standard `logging` to stdout (visible in `docker compose logs`).
- **Traces:** each turn writes a JSONL file to `trace_dir` with the full prompt as rendered by the chat template, the raw model output, the parsed output, and the tool calls/results. This is the main learning and debugging tool: it shows exactly what the model saw. The `/data` volume makes traces visible on the host.

## 7. Docker

### 7.1 Image
- `Dockerfile`: `python:3.12-slim` base. Install **CPU-only PyTorch** from the PyTorch CPU wheel index (avoids multi-GB CUDA wheels), then `requirements.txt`. Include `uv` and Node.js LTS for `uvx`/`npx`-launched MCP servers. Run as a non-root user.
- Target platform: `linux/arm64` (native on Apple Silicon). Should also build for `linux/amd64` for the later cloud move.
- `CMD ["uvicorn", "agent_bot.server:app", "--host", "0.0.0.0", "--port", "8000"]`.

### 7.2 `docker-compose.yml`
```yaml
services:
  agent:
    build: .
    ports: ["127.0.0.1:8000:8000"]
    volumes:
      - ./models:/models:ro
      - ./config.yaml:/app/config.yaml:ro
      - ./prompts:/app/prompts:ro
      - ./agent_bot/tools.py:/app/agent_bot/tools.py:ro   # add tools without rebuilding
      - ./mcp_servers:/app/mcp_servers:ro
      - ./data:/data
    env_file: [.env]                # MCP secrets; git-ignored, .env.example committed
    environment:
      - HF_HUB_OFFLINE=1          # weights come from the mount; never download at runtime
    deploy: { resources: { limits: { memory: 8g } } }
```

### 7.3 Host requirements
- **The user's machine:** an M2 Pro (12 cores, 16 GB RAM) running **Docker Desktop**. Rancher Desktop is also installed but not in use, and its `~/.rd/bin/docker` may come first on `PATH`. The README tells the user to check `docker context ls` and use the Docker Desktop context (`desktop-linux`). Memory is raised in *Settings → Resources*.
- The container VM needs **at least 8 GB memory** (12 GB for Qwen3-4B, which is tight on a 16 GB Mac with other apps open).
- Python 3.10+ with tkinter on the host for the GUI. Homebrew Python needs `brew install python-tk`.
- Restarting the container (`docker compose restart`) picks up changes to `tools.py`, `config.yaml`, prompts and `mcp_servers/` without a rebuild.

### 7.4 Reality check on speed
On CPU, expect roughly a few to ~15 tokens/second for a 1.7B model in bf16, depending on the Mac. One tool-using turn (2 model calls, with thinking) may take 30–90 s. That's acceptable for a proof of concept. Turning thinking off is the first speed lever. If it's still too slow, use the cloud GPU path (§11). *(These numbers are estimates; the builder records real ones in the README.)*

## 8. Repository layout

```
agent_bot/
├── plan.md
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            # server deps (torch installed separately in Dockerfile)
├── config.yaml
├── prompts/
│   └── system.md
├── agent_bot/                  # server package
│   ├── __init__.py
│   ├── server.py
│   ├── agent.py
│   ├── inference.py
│   ├── parsing.py
│   ├── registry.py
│   ├── tools.py
│   ├── mcp_tools.py
│   ├── config.py
│   └── events.py               # event dataclasses + to_json
├── mcp_servers/
│   └── notes_server.py
├── gui/
│   ├── app.py
│   └── requirements.txt
├── scripts/
│   └── download_model.py
├── models/                     # git-ignored; downloaded weights
├── data/                       # git-ignored; notes, traces
└── tests/
    ├── test_parsing.py
    ├── test_registry.py
    ├── test_tools.py
    ├── test_agent_loop.py
    └── test_mcp_tools.py
```

## 9. Testing

The whole stack must be testable **without loading a real model**:

- **`FakeBackend`** in tests: returns scripted raw outputs in sequence (e.g. first a `<tool_call>` for `get_current_time`, then a plain answer). This lets `test_agent_loop.py` cover:
  - a no-tool answer
  - a single tool call
  - multiple calls in one reply
  - a tool error fed back to the model
  - malformed JSON → retry
  - `max_iterations` reached
  - the repeat guard
  - the exact event sequence emitted
- **`test_parsing.py`**: Qwen3 outputs with and without thinking, multiple tool calls, broken JSON, arguments given as a string, no tags.
- **`test_registry.py`**: schema generation from a decorated function, argument validation, timeout, exception → `is_error`, truncation, duplicate names.
- **`test_tools.py`**: calculator safety (rejects `__import__`, attribute access, huge exponents) and time tool. Network tools are tested with mocked HTTP.
- **`test_mcp_tools.py`**: spawns the bundled `notes_server` over stdio, checks discovery and a round-trip call.
- **One slow, opt-in smoke test** (`pytest -m slow`) loads the real model and asks "What time is it in Tokyo?", expecting a `get_current_time` call.
- The GUI is tested manually (checklist in the README).

### 9.1 Reliability target (the proof-of-concept bar)

The owner wants a reliability target on **5 fixed prompts**, each solvable with the available tools. *(The specific prompts and pass checks below are Claude's proposal.)*

| # | Prompt | Must call | Answer check |
|---|---|---|---|
| 1 | "What's the weather in Istanbul tomorrow?" | `get_weather` with location containing "Istanbul", `days ≥ 2` | mentions a temperature (number + °C/°F) |
| 2 | "Search the web for the latest stable Python release and tell me its version number." | `web_search` (optionally `fetch_page`) | contains a version like `3.\d+(\.\d+)?` that appears in a tool result |
| 3 | "Save a note that my dentist appointment is Friday at 10am, then list all my notes." | `add_note` then `list_notes` (MCP) | mentions "dentist" |
| 4 | "What is 17.5% of 2,340?" | `calculator` | contains `409.5` |
| 5 | "What time is it in Tokyo right now?" | `get_current_time` with `timezone="Asia/Tokyo"` | contains a time within ±2 min of actual Tokyo time |

- **Eval script:** `scripts/eval.py` runs the prompts against the real model, through the same `agent.run` used by the server, in a fresh session each time. It runs inside the container (`docker compose exec agent python -m scripts.eval`). The prompts and checks live in `evals/prompts.yaml`, so they can be edited without code changes. Prompt 3 runs against an empty notes file.
- **Runs:** each prompt runs **3 times** at the configured temperature.
- **Pass rule:** a run passes when the required tool(s) were called with matching arguments, there were no unrecovered errors, it finished within `max_iterations`, and the answer check passes.
- **Target:** at least **4 of 5 prompts pass in at least 2 of 3 runs**. The stretch goal is 5/5.
- **Report:** a table per prompt/run (pass/fail, which check failed, iterations, seconds). It's printed and written to `/data/evals/<timestamp>.md`, with traces linked.
- **If the target isn't met**, try these levers in this order and record each result in the eval report history:
  1. system prompt and tool description wording
  2. thinking on vs off
  3. lower temperature
  4. switch to `Qwen3-4B`

  The final configuration that meets the target becomes the default in `config.yaml`.
- Prompts 1 and 2 depend on live web services. A failure caused by a network error, rather than the model, is marked `infra` and re-run, not counted.

## 10. Build order (milestones)

Each milestone ends in something runnable.

1. **Skeleton:** repo layout, config loading, `download_model.py`, model downloaded. ✅ when weights are in `./models/Qwen3-1.7B`.
2. **Inference alone:** `TransformersBackend` + a CLI script that streams a reply to a prompt on the host or in Docker. ✅ when it prints a streamed answer, including a `<think>` block.
3. **Parsing + registry + Python tools**, with unit tests. ✅ when tests pass.
4. **Agent loop** with `FakeBackend` tests, then a CLI (`python -m agent_bot.cli "weather in Istanbul"`) printing events against the real model. ✅ when the real model calls `get_weather` and answers.
5. **Server:** FastAPI with NDJSON streaming. ✅ when `curl -N` shows the event stream.
6. **Docker:** Dockerfile + compose. ✅ when the step-5 `curl` works against the container.
7. **GUI.** ✅ when a full turn renders all event kinds.
8. **MCP:** `mcp_tools.py` + `notes_server`, wired into the registry. ✅ when "save a note that I like tea" then "what notes do I have?" works from the GUI.
9. **Reliability eval** (§9.1): `scripts/eval.py` + `evals/prompts.yaml`, then tuning. ✅ when ≥ 4/5 prompts pass in ≥ 2/3 runs.
10. **Polish:** traces, README (setup, how to add a tool, how to add an MCP server, how to swap models, measured speed), manual GUI checklist.

## 11. Later: moving to a cloud GPU with Unsloth (designed for, not built now)

- Implement `UnslothBackend`: `FastLanguageModel.from_pretrained(model.path, load_in_4bit=…)`, `FastLanguageModel.for_inference(model)`, then the same `apply_chat_template` + streamer flow as `TransformersBackend` (Unsloth returns `transformers` objects, so most code is shared; factor it into a base class).
- Add a `Dockerfile.gpu` (CUDA base image, `unsloth`) and a compose override with GPU reservation.
- The GUI points at the cloud URL via `AGENT_URL`. At that point the server needs auth (an API key header at minimum) and must not be published on `0.0.0.0` without it.
- Nothing in `agent.py`, `parsing.py`, `registry.py`, the tools or the GUI should change.

## 12. Decisions log

**Made by the user:**
- Hugging Face model with locally downloaded weights; an inference backend runs it.
- Separate `inference.py`, `tools.py`, MCP module; tools both as Python functions and via MCP.
- tkinter GUI showing tool calls, thinking and response.
- Runs in Docker on the Mac as a proof of concept; may move to a cloud runtime later.
- Learning project, customizable by adding tools and swapping models.
- Hand-written agent loop; `transformers` on CPU now, Unsloth on a cloud GPU later.
- Docker Desktop (not Rancher Desktop) is the container runtime.
- A reliability target on 5 fixed prompts answerable with the available tools, rather than a one-off demo.

**Made by Claude (check these):**
- GUI runs on the host, talking to the container over HTTP with NDJSON streaming (tkinter-in-Docker is impractical on a Mac).
- Added `agent.py` (the loop), `parsing.py`, `registry.py`, `server.py`, `config.py` to the original file list.
- `mcp.py` renamed to `mcp_tools.py` (name clash with the `mcp` package).
- Default model `Qwen/Qwen3-1.7B` in bf16; `Qwen3-4B` as the upgrade.
- Web search via DuckDuckGo (`ddgs`), no API key; dedicated Open-Meteo weather tool.
- Starter tools: web_search, fetch_page, get_weather, get_current_time, calculator.
- Bundled example MCP server: notes (FastMCP), plus a disabled third-party `mcp-server-fetch` entry.
- MCP scope: stdio (`python`/`uvx`/`npx`, so Node is in the image) and Streamable HTTP; secrets via `.env` with `${VAR}` substitution. Servers needing host files need a manual volume mount.
- `tools.py`, `mcp_servers/`, config and prompts are bind-mounted so customizing needs a restart, not a rebuild.
- Default stays Qwen3-1.7B even though this Mac (16 GB) could run Qwen3-4B, to leave memory headroom. Switching is a config change.
- Weights mounted as a volume, not baked into the image; `HF_HUB_OFFLINE=1` at runtime.
- In-memory sessions; thinking stripped from history; sequential tool execution; max 6 iterations; 20 s tool timeout; 4000-char result cap.
- FastAPI + uvicorn, pydantic config, pytest with a fake backend.
- The 5 eval prompts and their checks (§9.1), 3 runs each, and a pass bar of ≥ 4/5 prompts in ≥ 2/3 runs; the order of tuning levers.

## 13. Left to the builder

- Exact system prompt wording.
- GUI visual styling, fonts and colors (within §6.10).
- Sync/async bridging mechanism in the server.
- Whether to use the `transformers` built-in schema/response helpers or hand-rolled equivalents (contracts in §6.4–6.6 must hold).
- HTML-to-text library for `fetch_page`.
- **Nice-to-haves if time allows:** stop button (cancel between iterations), model switcher in the GUI via a `/models` endpoint, token/sec display, a second parser (e.g. for Gemma) to prove model swapping across families, a smolagents comparison script.
