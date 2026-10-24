# agent_bot — Task Tracker

The detailed steps for each task are in [docs/superpowers/plans/2026-10-22-agent-bot.md](docs/superpowers/plans/2026-10-22-agent-bot.md), and the design is in [plan.md](plan.md) (revision 2: MLX on the Mac GPU, Docker as the tool sandbox).

Status: ✅ done · ⏳ next · ⬜ not started

| # | Task | Done when | Status | Commits |
|---|---|---|---|---|
| 1 | Skeleton, config loading, model download | Weights are in `models/Qwen3-1.7B`, config tests pass | ✅ | `c5972c8` |
| 2 | Inference backend (CPU, `transformers`) | Streams a reply with a `<think>` block; generation errors are raised, not hung | ✅ | `38e3b69`, `f82e170` |
| 3 | Qwen3 output parser | Thinking, tool calls and text are split out; cut-off or broken output is reported, never dropped | ✅ | `d75cda1`, `8497532` |
| 4 | Tool registry | Argument checks, timeouts, truncation and errors work, and `call` never raises | ✅ | `bceadd4` |
| 5 | The built-in tools | `web_search`, `fetch_page`, `get_current_time` and a safe `calculator` work and are tested offline | ✅ | `ceef164`, `a34c240` |
| 6 | Revision-2 amendments (config, registry, requirements) | `mlx` is the default backend, the sandbox config exists, `ToolError` and `replace_origin` work | ✅ | `52ab355` |
| 7 | MLX backend on the Apple GPU | `scripts/generate.py` streams from MLX at tens of tok/s (measured 40–47) | ✅ | `965f24b` |
| 8 | Events and traces | Event types serialize to NDJSON; each turn writes a JSONL trace | ✅ | `afced78` |
| 9 | Agent loop + CLI | `python -m agent_bot.cli "…"` — the real model calls a tool and answers | ✅ | `bcf55f6` |
| 10 | Agent server on the host | `curl -N` shows the event stream from `/chat` | ✅ | `3a2bc14` |
| 11 | tkinter GUI | A full turn shows all event kinds (thinking, tool call, tool result, answer, error) | ✅ | `0d2e39e` |
| 12 | MCP client + notes server | "Save a note that I like tea" and then "what notes do I have?" works | ✅ | `fd5f240`, `42c4956` |
| 13 | Tool sandbox (tool server + Docker, headless Chromium) | GUI turns use container tools; after `docker compose restart`, a newly added tool shows up on the next turn; `fetch_page` reads JavaScript-only pages | ⬜ | |
| 14 | Reliability eval + tuning | At least 4 of 5 eval prompts pass in at least 2 of 3 runs | ⬜ | |
| 15 | README, smoke test, polish | Setup and customization docs, measured speeds, and a manual GUI checklist | ⬜ | |
| 16 | Approval gate for send tools + Mac apps MCP server (host) | A GUI turn creates an Apple Note; `send_imessage` runs only after the user clicks Approve, and Decline means nothing is sent | ⬜ | |
| 17 | Document ingestion pipeline | Files dropped into `data/inbox/` (PDF, Markdown, text, HTML) are parsed, chunked and indexed with their source and page; a `search_documents` tool returns cited chunks; re-ingesting an unchanged file is a no-op and a changed file replaces its old chunks | ⬜ | |
| 18 | Unified persistent memory | One store under `data/` holds notes, remembered facts from conversations and ingested documents; it survives an agent server restart; a single `recall` tool searches all of it with source labels; "Remember I like tea", restart, then "what do I like?" works | ⬜ | |
| 19 | GUI key bindings | Ctrl+Q quits, Ctrl+N starts a new session, Ctrl+L clears the transcript view, Esc stops the running turn, and Ctrl+/ shows a shortcuts list; on macOS each also works with Cmd; bindings live in one table and fire only once per press | ⬜ | |
| 20 | Timer-based event triggers in the server | Scheduled tasks (prompt, daily time or interval, local time zone) are saved under `data/` and reloaded when the server starts; the server sits idle until a timer fires and then starts a turn with the task's prompt; a task due while the server was down runs once at startup, not once per missed day; tested with a fake clock | ⬜ | |
| 21 | Scheduled tasks for the agent | `schedule_task`, `list_scheduled_tasks` and `cancel_scheduled_task` tools work; "check the weather every day at 9am" registers a task, and when it fires the agent runs it and the result shows in the GUI and the trace; a turn the timer starts while another turn is running waits its turn and doesn't interrupt it | ⬜ | |
| 22 | Collapsible events in the conversation view | Thinking blocks, tool calls and tool results show as one-line headers (e.g. "▸ Thinking · 312 tokens", "▸ calculator(…) · 0.2 s") that expand and collapse on click; they start collapsed, while answers and errors always show in full; a "show all / hide all" toggle and a remembered default work; long streaming turns don't flicker or jump the scroll position | ⬜ | |
| 23 | *(Experimental)* Rust backend study | A short write-up in `docs/` with a profile of a real turn that shows where the time goes (prefill, decode, parsing, tool calls, HTTP/event streaming); a prototype of the hottest non-GPU path in Rust (e.g. via PyO3, or `mlx-rs`/`candle` for inference) benchmarked against the current Python version on the same prompts; a go/no-go recommendation with the measured gain and the maintenance cost | ⬜ | |
| 24 | Filesystem tools on the host (`ls`, `cat`, `touch`, `mkdir`) | The model can list a directory, read a text file, create an empty file and create a directory on the Mac; they are Python tools, not a shell, so no other command can run; every path is resolved (`~`, `..`, symlinks) and must fall inside the allowed roots in `config.yaml`, otherwise the tool returns an error; `cat` caps output and rejects binary files; `touch`/`mkdir` never overwrite existing content; "make a folder `~/agent_workspace/trip` and list it" works in a GUI turn | ⬜ | |

## Measurements so far

| What | Result |
|---|---|
| Qwen3-1.7B, CPU, bf16 | 0.33 tok/s |
| Qwen3-1.7B, CPU, float32 | 4.6 tok/s |
| Qwen3-1.7B, MLX on the GPU | 38 tok/s on the first call, 47 tok/s once warmed up |
| MLX, 2k-token prompt + ~350 generated: peak RAM / speed | 1.7B bf16 4.24 GB / 43 tok/s; 4B bf16 8.89 GB / 20 tok/s; 4B 8-bit 5.30 GB / 35 tok/s |
| Prompt cache, Qwen3-4B 8-bit, ~3k-token prompt: time to first token on the next iteration | 6.70 s without the cache, 0.24 s with it (2,992 tokens reused) |
| Agent CLI, Qwen3-4B 8-bit, "What time is it in Tokyo?" | correct tool call and answer, 2 iterations, 11.6 s |
| Agent server, real model, "17.5% of 2340" | calculator called, 409.5 correct, 2 iterations, 12.2 s, 488 tokens |
| Tool choice (5 prompts, CPU, thinking off, 1 run) | 5/5 correct tool; the answer to "17.5% of 2,340" was wrong (40,950 instead of 409.5) |
