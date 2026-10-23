"""The agent's HTTP API on the host: one turn per POST /chat, streamed as NDJSON events.

    uvicorn agent_bot.server:app --host 127.0.0.1 --port 8000
    python -m agent_bot.server          # same, host/port from config.yaml

The model loads in a background thread at startup, so /health answers while it loads.
One turn runs at a time (a single local model can't serve two); a second /chat gets 409.
Python tools run in-process only when agent.local_python_tools is true; the sandboxed
tools are wired in by Task 13.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_bot.agent import RunSettings, run
from agent_bot.config import AppConfig, load_config
from agent_bot.inference import Backend, create_backend
from agent_bot.parsing import get_parser
from agent_bot.registry import ToolRegistry
from agent_bot.tools import register_python_tools
from agent_bot.trace import JsonlTracer, NullTracer

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)
    enable_thinking: bool | None = None  # None = config default


class ResetRequest(BaseModel):
    session_id: str


@dataclass
class _State:
    cfg: AppConfig | None = None
    backend: Backend | None = None
    load_error: str | None = None
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    prompt_template: str = ""
    sessions: dict[str, list[dict]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _TurnResponse(StreamingResponse):
    """Releases the turn lock however the response ends: finished, failed or disconnected.

    Doing it here rather than only in the body generator also covers a client that
    leaves before the first byte, when the generator never starts and its `finally`
    would never run.
    """

    def __init__(self, content: AsyncIterator[str], release: Callable[[], None], **kw):
        super().__init__(content, **kw)
        self._release = release

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.body_iterator.aclose()  # stops the agent loop and its worker thread
            self._release()


def create_app(config: AppConfig | None = None, backend: Backend | None = None) -> FastAPI:
    state = _State(backend=backend)

    async def load_model(cfg: AppConfig) -> None:
        try:
            state.backend = await asyncio.to_thread(create_backend, cfg.model, cfg.generation)
            log.info("model loaded: %s (%s)", state.backend.model_name, cfg.model.backend)
        except Exception as exc:
            state.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model failed to load")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = state.cfg = config or load_config()
        state.registry = ToolRegistry(cfg.tools.max_result_chars, cfg.tools.disabled)
        if cfg.agent.local_python_tools:
            register_python_tools(state.registry)
        state.prompt_template = Path(cfg.agent.system_prompt_file).read_text(encoding="utf-8")
        loading = None
        if state.backend is None:
            loading = asyncio.create_task(load_model(cfg))
        yield
        if loading is not None:
            loading.cancel()  # the loader thread can't be interrupted; this just drops the task

    app = FastAPI(title="agent_bot", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        cfg = state.cfg
        loaded = state.backend is not None
        return {
            "status": "ok",
            "model": state.backend.model_name if loaded else Path(cfg.model.path).name,
            "backend": cfg.model.backend,
            "model_loaded": loaded,
        }

    @app.get("/tools")
    async def tools() -> dict:
        return {
            "tools": [{"name": t.name, "description": t.description, "source": t.source}
                      for t in state.registry.tools()],
            "mcp_servers": [],  # sandbox and host MCP servers arrive in Task 13
        }

    @app.post("/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        if state.backend is None:
            detail = (f"model failed to load: {state.load_error}" if state.load_error
                      else "model is still loading")
            raise HTTPException(503, detail)
        if state.lock.locked():
            raise HTTPException(409, "busy: another request is running")
        await state.lock.acquire()  # free, so this never waits
        try:
            response = _start_turn(state, req)
        except BaseException:
            state.lock.release()
            raise
        return response

    @app.post("/reset")
    async def reset(req: ResetRequest) -> dict:
        state.sessions.pop(req.session_id, None)
        return {"ok": True}

    return app


def _start_turn(state: _State, req: ChatRequest) -> _TurnResponse:
    cfg = state.cfg
    history = state.sessions.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})
    settings = RunSettings(
        system_prompt=state.prompt_template.replace("{today}", date.today().isoformat()),
        enable_thinking=(cfg.generation.enable_thinking if req.enable_thinking is None
                         else req.enable_thinking),
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        max_iterations=cfg.agent.max_iterations,
        tool_timeout_s=cfg.tools.timeout_s,  # +5 for sandbox tools (Task 13)
    )
    tracer = (JsonlTracer(Path(cfg.logging.trace_dir), f"{req.session_id}-{uuid.uuid4().hex}")
              if cfg.logging.trace_dir else NullTracer())
    backend = state.backend

    async def body() -> AsyncIterator[str]:
        events = run(history, backend=backend, parser=get_parser(backend.model_family),
                     registry=state.registry, settings=settings, tracer=tracer)
        async with aclosing(events):
            async for event in events:
                yield event.to_json() + "\n"

    return _TurnResponse(body(), release=state.lock.release, media_type="application/x-ndjson")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _cfg = load_config()
    logging.basicConfig(level=_cfg.logging.level)
    uvicorn.run(create_app(_cfg), host=_cfg.server.host, port=_cfg.server.port)
