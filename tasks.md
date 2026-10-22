# agent_bot — Task Tracker

The detailed steps for each task are in [docs/superpowers/plans/2026-10-22-agent-bot.md](docs/superpowers/plans/2026-10-22-agent-bot.md), and the design is in [plan.md](plan.md) (revision 2: MLX on the Mac GPU, Docker as the tool sandbox).

Status: ✅ done · ⏳ next · ⬜ not started

| # | Task | Done when | Status | Commits |
|---|---|---|---|---|
| 1 | Skeleton, config loading, model download | Weights are in `models/Qwen3-1.7B`, config tests pass | ✅ | `c5972c8` |
| 2 | Inference backend (CPU, `transformers`) | Streams a reply with a `<think>` block; generation errors are raised, not hung | ✅ | `38e3b69`, `f82e170` |
| 3 | Qwen3 output parser | Thinking, tool calls and text are split out; cut-off or broken output is reported, never dropped | ✅ | `d75cda1`, `8497532` |
| 4 | Tool registry | Argument checks, timeouts, truncation and errors work, and `call` never raises | ✅ | `bceadd4` |
| 5 | The 5 built-in tools | `web_search`, `fetch_page`, `get_weather`, `get_current_time` and a safe `calculator` work and are tested offline | ✅ | `ceef164`, `a34c240` |
| 6 | Revision-2 amendments (config, registry, requirements) | `mlx` is the default backend, the sandbox config exists, `ToolError` and `replace_origin` work | ✅ | `52ab355` |
| 7 | MLX backend on the Apple GPU | `scripts/generate.py` streams from MLX at tens of tok/s (measured 40–47) | ✅ | `965f24b` |
| 8 | Events and traces | Event types serialize to NDJSON; each turn writes a JSONL trace | ✅ | `afced78` |
| 9 | Agent loop + CLI | `python -m agent_bot.cli "…"` — the real model calls `get_weather` and answers | ⏳ | |
| 10 | Agent server on the host | `curl -N` shows the event stream from `/chat` | ⬜ | |
| 11 | tkinter GUI | A full turn shows all event kinds (thinking, tool call, tool result, answer, error) | ⬜ | |
| 12 | MCP client + notes server | "Save a note that I like tea" and then "what notes do I have?" works | ⬜ | |
| 13 | Tool sandbox (tool server + Docker) | GUI turns use container tools; after `docker compose restart`, a newly added tool shows up on the next turn | ⬜ | |
| 14 | Reliability eval + tuning | At least 4 of 5 eval prompts pass in at least 2 of 3 runs | ⬜ | |
| 15 | README, smoke test, polish | Setup and customization docs, measured speeds, and a manual GUI checklist | ⬜ | |

## Measurements so far

| What | Result |
|---|---|
| Qwen3-1.7B, CPU, bf16 | 0.33 tok/s |
| Qwen3-1.7B, CPU, float32 | 4.6 tok/s |
| Qwen3-1.7B, MLX on the GPU | 38 tok/s on the first call, 47 tok/s once warmed up |
| Tool choice (5 prompts, CPU, thinking off, 1 run) | 5/5 correct tool; the answer to "17.5% of 2,340" was wrong (40,950 instead of 409.5) |
