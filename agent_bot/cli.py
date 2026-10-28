"""Run one agent turn from the terminal and print every event as it happens.

    python -m agent_bot.cli "What time is it in Tokyo?" [--no-thinking] [--local-tools]
        [--model-path P] [--backend mlx|transformers|unsloth]

Tools come from the Docker sandbox, like in the server, unless --local-tools (or
agent.local_python_tools) runs the Python tools in this process instead.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from contextlib import AsyncExitStack
from datetime import date
from pathlib import Path

from agent_bot.agent import RunSettings, run
from agent_bot.config import load_config
from agent_bot.events import Error, Event
from agent_bot.inference import create_backend
from agent_bot.mac_tools import maybe_register_mac_tools
from agent_bot.mcp_tools import connect_mcp_servers, failure_messages, tool_timeout, turn_tools
from agent_bot.parsing import get_parser
from agent_bot.registry import ToolRegistry
from agent_bot.tools import register_python_tools
from agent_bot.trace import JsonlTracer, NullTracer

_TTY = sys.stdout.isatty()


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _TTY else text


def setup_logging(level: str) -> None:
    """Our own modules log at `level` (logging.level in config.yaml) to stderr.

    The root stays at WARNING so libraries such as httpx don't print every request.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("agent_bot").setLevel(level)


def _print_event(event: Event, state: dict) -> None:
    if event.type == "token":
        # Raw model output streams inline; every other event starts on a fresh line.
        print(_dim(event.text), end="", flush=True)
        state["mid_line"] = True
        return
    if state.pop("mid_line", False):
        print()
    if event.type == "thinking":
        print(_dim(f"[thinking] {event.text}"))
    elif event.type == "tool_call":
        args = json.dumps(event.arguments, ensure_ascii=False)
        print(f"→ tool_call {event.name}({args}) [{event.source}]")
    elif event.type == "tool_result":
        lines = event.content.splitlines() or [""]
        more = f"\n    … ({len(lines) - 5} more lines)" if len(lines) > 5 else ""
        tag = "error" if event.is_error else f"{event.duration_ms} ms"
        print(f"← tool_result ({tag})\n    " + "\n    ".join(lines[:5]) + more)
    elif event.type == "answer":
        print(f"[answer] {event.text}")
    elif event.type == "error":
        print(f"[error] {event.message}")
    elif event.type == "done":
        print(f"[done] {event.iterations} iterations, {event.elapsed_s}s, "
              f"{event.tokens_generated} tokens")


async def _main(args: argparse.Namespace) -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    if args.model_path:
        cfg.model.path = args.model_path
    if args.backend:
        cfg.model.backend = args.backend

    print(_dim(f"Loading {cfg.model.path} ({cfg.model.backend})…"), flush=True)
    backend = create_backend(cfg.model, cfg.generation)
    registry = ToolRegistry(cfg.tools.max_result_chars, cfg.tools.disabled)
    local = args.local_tools or cfg.agent.local_python_tools
    if local:
        register_python_tools(registry)
    maybe_register_mac_tools(registry, cfg)
    # The sandbox (any url server) is skipped when its tools run in this process instead.
    per_turn = {} if local else cfg.mcp_servers

    template = Path(cfg.agent.system_prompt_file).read_text(encoding="utf-8")
    settings = RunSettings(
        system_prompt=template.replace("{today}", date.today().isoformat()),
        enable_thinking=cfg.generation.enable_thinking and not args.no_thinking,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        max_iterations=cfg.agent.max_iterations,
        tool_timeout_s=cfg.tools.timeout_s if local else tool_timeout(cfg),
    )
    tracer = (JsonlTracer(Path(cfg.logging.trace_dir), uuid.uuid4().hex)
              if cfg.logging.trace_dir else NullTracer())

    history = [{"role": "user", "content": args.prompt}]
    state: dict = {}
    async with AsyncExitStack() as stack:
        stdio = {n: c for n, c in cfg.mcp_servers.items() if c.command}
        await connect_mcp_servers(stdio, registry, stack)
        async with turn_tools(per_turn, registry) as statuses:
            for message in failure_messages(per_turn, statuses):
                _print_event(Error(message), state)
            async for event in run(history, backend=backend,
                                   parser=get_parser(backend.model_family),
                                   registry=registry, settings=settings, tracer=tracer):
                _print_event(event, state)
    if isinstance(tracer, JsonlTracer) and tracer.path:
        print(_dim(f"trace: {tracer.path}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one agent turn in the terminal.")
    parser.add_argument("prompt")
    parser.add_argument("--no-thinking", action="store_true", help="disable Qwen3 thinking mode")
    parser.add_argument("--local-tools", action="store_true",
                        help="run the Python tools in this process instead of the Docker sandbox")
    parser.add_argument("--model-path", help="override model.path from config.yaml")
    parser.add_argument("--backend", choices=["mlx", "transformers", "unsloth"],
                        help="override model.backend from config.yaml")
    return parser


def main() -> None:
    asyncio.run(_main(build_parser().parse_args()))


if __name__ == "__main__":
    main()
