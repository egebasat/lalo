import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from agent_bot import server
from agent_bot.config import AgentConfig, AppConfig, LoggingConfig, ModelConfig
from agent_bot.server import create_app
from tests.fakes import FakeBackend


@pytest.fixture
def make_config(tmp_path):
    prompt = tmp_path / "system.md"
    prompt.write_text("You are a test assistant. Today is {today}.")

    def make(local_python_tools: bool = True) -> AppConfig:
        return AppConfig(
            model=ModelConfig(path="models/fake"),
            agent=AgentConfig(system_prompt_file=str(prompt), local_python_tools=local_python_tools),
            logging=LoggingConfig(trace_dir=str(tmp_path / "traces")),
        )

    return make


def chat(client, message="hi", session_id="s1", **extra):
    resp = client.post("/chat", json={"session_id": session_id, "message": message, **extra})
    assert resp.status_code == 200, resp.text
    return [json.loads(line) for line in resp.text.splitlines() if line]


def test_health_reports_model(make_config):
    with TestClient(create_app(make_config(), backend=FakeBackend([]))) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"] == "fake"
    assert body["backend"] == "mlx"


def test_tools_lists_python_tools_when_local(make_config):
    with TestClient(create_app(make_config(), backend=FakeBackend([]))) as client:
        body = client.get("/tools").json()
    names = [t["name"] for t in body["tools"]]
    assert "calculator" in names
    assert all(t["source"] == "python" for t in body["tools"])
    assert body["mcp_servers"] == []


def test_no_tools_when_not_local(make_config):
    app = create_app(make_config(local_python_tools=False), backend=FakeBackend([]))
    with TestClient(app) as client:
        assert client.get("/tools").json()["tools"] == []


def test_chat_streams_ndjson(make_config, tmp_path):
    with TestClient(create_app(make_config(), backend=FakeBackend(["Hello there!"]))) as client:
        resp = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in resp.text.splitlines() if line]
    assert events[0]["type"] == "token"
    assert events[-1]["type"] == "done"
    assert {"type": "answer", "text": "Hello there!"}.items() <= next(
        e for e in events if e["type"] == "answer").items()
    assert list((tmp_path / "traces").glob("*.jsonl")), "no trace written"


def test_session_history_persists_and_resets(make_config):
    backend = FakeBackend(["First.", "Second.", "Third."])
    with TestClient(create_app(make_config(), backend=backend)) as client:
        chat(client, "one")
        chat(client, "two")
        assert client.post("/reset", json={"session_id": "s1"}).json() == {"ok": True}
        chat(client, "three")

    def user_and_assistant(call):
        return [(m["role"], m["content"]) for m in call["messages"] if m["role"] != "system"]

    assert user_and_assistant(backend.calls[1]) == [
        ("user", "one"), ("assistant", "First."), ("user", "two")]
    assert user_and_assistant(backend.calls[2]) == [("user", "three")]


def test_sessions_are_separate(make_config):
    backend = FakeBackend(["A.", "B."])
    with TestClient(create_app(make_config(), backend=backend)) as client:
        chat(client, "one", session_id="a")
        chat(client, "two", session_id="b")
    assert [m["content"] for m in backend.calls[1]["messages"] if m["role"] != "system"] == ["two"]


def test_enable_thinking_override_reaches_backend(make_config):
    backend = FakeBackend(["Ok.", "Ok."])
    with TestClient(create_app(make_config(), backend=backend)) as client:
        chat(client, enable_thinking=False)
        chat(client)  # omitted -> config default (true)
    assert [c["enable_thinking"] for c in backend.calls] == [False, True]


def test_concurrent_chat_returns_409(make_config):
    block = threading.Event()
    backend = FakeBackend(["First.", "Second."], block=block)
    with TestClient(create_app(make_config(), backend=backend)) as client:
        first: dict = {}
        worker = threading.Thread(target=lambda: first.update(events=chat(client, "one")))
        worker.start()
        deadline = time.monotonic() + 5
        while not backend.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert backend.calls, "first request never reached the backend"

        second = client.post("/chat", json={"session_id": "s2", "message": "two"})
        assert second.status_code == 409
        assert second.json() == {"detail": "busy: another request is running"}

        block.set()
        worker.join(5)
    assert first["events"][-1]["type"] == "done"


class _SlowFirstBackend(FakeBackend):
    """The first generate() yields one chunk and then stalls until released."""

    def __init__(self, outputs):
        super().__init__(outputs)
        self.release = threading.Event()

    def generate(self, messages, tools, **kw):
        if not self.calls:
            self.calls.append({"messages": messages, "enable_thinking": kw["enable_thinking"]})
            yield "Hel"
            self.release.wait(5)
            yield "lo."
            return
        yield from super().generate(messages, tools, **kw)


def test_lock_released_after_disconnect(make_config):
    backend = _SlowFirstBackend(["Second."])
    try:
        with TestClient(create_app(make_config(), backend=backend)) as client:
            with client.stream("POST", "/chat", json={"session_id": "s1", "message": "one"}) as resp:
                first_line = next(resp.iter_lines())
                assert json.loads(first_line)["type"] == "token"
            # closed mid-turn: the client disconnected

            start = time.monotonic()
            status = 409
            while status == 409 and time.monotonic() - start < 2:
                second = client.post("/chat", json={"session_id": "s2", "message": "two"})
                status = second.status_code
            assert status == 200
    finally:
        backend.release.set()


def test_chat_503_while_loading(make_config, monkeypatch):
    loaded = threading.Event()

    def slow_create_backend(model_cfg, gen_cfg):
        loaded.wait(5)
        return FakeBackend([])

    monkeypatch.setattr(server, "create_backend", slow_create_backend)
    try:
        with TestClient(create_app(make_config())) as client:
            assert client.get("/health").json()["model_loaded"] is False
            resp = client.post("/chat", json={"session_id": "s1", "message": "hi"})
            assert resp.status_code == 503
            assert resp.json() == {"detail": "model is still loading"}
            loaded.set()
    finally:
        loaded.set()


def test_empty_message_422(make_config):
    with TestClient(create_app(make_config(), backend=FakeBackend([]))) as client:
        assert client.post("/chat", json={"session_id": "s1", "message": ""}).status_code == 422
