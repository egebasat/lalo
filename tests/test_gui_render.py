import ast
from pathlib import Path

import pytest

pytest.importorskip("tkinter")

from gui.app import iter_events, render_event  # noqa: E402

APP = Path(__file__).parent.parent / "gui" / "app.py"


def test_render_tool_call():
    event = {"type": "tool_call", "id": "c1", "name": "web_search",
             "arguments": {"query": "x"}, "source": "sandbox/python"}
    [(text, tag)] = render_event(event)
    assert text.strip() == '→ web_search({"query": "x"})  [sandbox/python]'
    assert tag == "tool_call"


def test_render_long_tool_result_truncated():
    content = "\n".join(f"line {i}" for i in range(40))
    event = {"type": "tool_result", "id": "c1", "content": content,
             "is_error": False, "duration_ms": 12}
    [(text, tag)] = render_event(event)
    lines = text.rstrip("\n").splitlines()
    assert tag == "tool_result"
    assert lines[:15] == [f"    line {i}" for i in range(15)]
    assert lines[15].strip() == "…"
    assert len(lines) == 16


def test_render_error_result_tag():
    event = {"type": "tool_result", "id": "c1", "content": "Tool 'x' failed",
             "is_error": True, "duration_ms": 3}
    [(_, tag)] = render_event(event)
    assert tag == "tool_error"


def test_render_done_and_token():
    [(text, tag)] = render_event({"type": "done", "iterations": 2, "elapsed_s": 3.4,
                                  "tokens_generated": 120})
    assert tag == "meta"
    assert "2" in text and "3.4" in text and "120" in text
    assert render_event({"type": "token", "text": "hi"}) == []


def test_iter_events_bad_line():
    events = list(iter_events([b'{"type":"answer","text":"hi"}', b'', b'oops']))
    assert len(events) == 2
    assert events[0] == {"type": "answer", "text": "hi"}
    assert events[1]["type"] == "error"
    assert "Bad line from server" in events[1]["message"]


def test_gui_imports_nothing_from_agent_bot():
    imported = set()
    for node in ast.walk(ast.parse(APP.read_text())):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [m for m in imported if m == "agent_bot" or m.startswith("agent_bot.")]


def test_rejected_chat_yields_one_error_and_finishes(monkeypatch):
    import queue
    from types import SimpleNamespace

    import gui.app as gui_app

    class Rejected:
        status_code = 409
        text = '{"detail": "busy: another request is running"}'

        def json(self):
            return {"detail": "busy: another request is running"}

        def iter_lines(self):
            return iter([self.text.encode()])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(gui_app.requests, "post", lambda *a, **kw: Rejected())
    fake_app = SimpleNamespace(url="http://x", inbox=queue.Queue())
    gui_app.App._chat(fake_app, {"session_id": "s", "message": "hi"}, 7)
    items = []
    while not fake_app.inbox.empty():
        items.append(fake_app.inbox.get_nowait())
    assert items == [
        ("event", (7, {"type": "error", "message": "HTTP 409: busy: another request is running"})),
        ("finished", (7, True)),
    ]
