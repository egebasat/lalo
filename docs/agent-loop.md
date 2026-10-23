# How `agent.py` works

`agent_bot/agent.py` is a small, hand-written ReAct loop: **generate → parse →
run tools → repeat** until the model answers without calling a tool.

It only talks to three interfaces, so it knows nothing about a specific model,
MCP or concrete tools:

| Interface | Job | Implementations |
|---|---|---|
| `Backend` (`inference.py`) | render the prompt and stream generated text | `MLXBackend`, `TransformersBackend` |
| `Parser` (`parsing.py`) | split raw text into thinking / tool calls / content | `Qwen3Parser` |
| `ToolRegistry` (`registry.py`) | list tool schemas and run calls safely | Python tools + MCP tools |

## Entry point

```python
async for event in run(history, backend=..., parser=..., registry=...,
                       settings=RunSettings(...), tracer=...):
    ...
```

- `history` is the conversation (list of chat messages). `run` **appends** the
  assistant and tool messages of this turn to it, so the caller keeps it for the
  next turn.
- `RunSettings` holds the system prompt, thinking on/off, token limit,
  temperature, `max_iterations` and the tool timeout.
- Output is a stream of `Event`s (`events.py`): `Token`, `Thinking`,
  `ToolCallEvent`, `ToolResultEvent`, `Answer`, `Error`, and always a final `Done`
  with iteration count, elapsed time and tokens generated. The CLI prints them;
  a GUI can send them as JSON lines (`Event.to_json()`).

## One iteration

```
           ┌──────────────────────────────────────────────┐
           ▼                                              │
 system prompt + history + tool schemas                   │
           │  backend.generate()  → Token events          │
           ▼                                              │
       raw text  ── parser.parse() ──►  thinking / tool_calls / content
           │                                              │
   no tool calls ──► Answer, stop                         │
   broken markup ──► Error + correction message ──────────┤
   tool calls    ──► registry.call() each one             │
                     → ToolCall/ToolResult events         │
                     → append {"role": "tool"} message ───┘
```

1. **Build messages.** The system prompt is prepended on every call and never
   stored in `history`, so history stays pure conversation.
2. **Generate.** `backend.generate()` is a blocking iterator; `stream_in_thread`
   runs it on a worker thread and feeds chunks back through an `asyncio.Queue`, so
   the event loop (and any HTTP server on it) stays responsive. Each chunk is
   yielded as a `Token`.
3. **Parse.** Thinking is emitted as a `Thinking` event but not saved to history
   (it would only waste context). The assistant message, with any `tool_calls`, is
   appended.
4. **Decide.**
   - *Broken tool-call markup:* emit an `Error`, add a short user message telling
     the model what was wrong, and loop again — small models usually fix it. If
     the model simply ran out of tokens mid-`<think>`, end with an explanatory
     `Answer` instead of retrying.
   - *No tool calls:* the visible content is the `Answer`; the turn ends.
   - *Tool calls:* run them **one at a time** (predictable order, light on a local
     machine). Each gets an id (`c1`, `c2`, …), a `ToolCallEvent`, the
     `registry.call()` result as a `ToolResultEvent`, and a `tool` message in
     history. Then loop back to 1 so the model can read the results.

## Safety limits

- **`max_iterations`** — after that many generate steps without an answer, the
  turn stops with an `Error`.
- **Repeat guard** — the same tool with the same arguments a 3rd time
  (`MAX_IDENTICAL_CALLS`) stops the turn; small models sometimes loop.
- **Tool errors never crash the loop** — the registry returns them as
  `is_error` results the model can react to. Only a model/backend exception
  escapes `_loop`; `run` turns it into an `Error` event.
- **`Done` always comes last**, even after an error, with the stats collected so
  far (`_Stats`).

## Tracing

Each step is recorded through a `Tracer` (`trace.py`): the fully rendered
prompt, raw output, parsed output, and every tool call and result. With
`logging.trace_dir` set, the CLI writes them to a JSONL file per run, which is
the first place to look when the model does something odd. `NullTracer` (the
default) records nothing.
