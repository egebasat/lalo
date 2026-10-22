import json

from agent_bot.events import Answer, Done, ToolCallEvent, ToolResultEvent
from agent_bot.trace import JsonlTracer, NullTracer


def test_event_json_shape():
    d = json.loads(ToolCallEvent("c1", "calc", {"expression": "1+1"}, "python").to_json())
    assert d["type"] == "tool_call"
    assert isinstance(d["ts"], float)
    assert d["id"] == "c1" and d["name"] == "calc"
    assert d["arguments"] == {"expression": "1+1"} and d["source"] == "python"


def test_other_event_shapes():
    d = json.loads(ToolResultEvent("c1", "x", False, 12).to_json())
    assert d["type"] == "tool_result" and d["is_error"] is False and d["duration_ms"] == 12
    d = json.loads(Done(2, 41.3, 512).to_json())
    assert d["type"] == "done" and d["iterations"] == 2 and d["tokens_generated"] == 512


def test_unicode_preserved():
    assert "İstanbul 18°C" in Answer("İstanbul 18°C").to_json()


def test_jsonl_tracer_writes_lines(tmp_path):
    t = JsonlTracer(tmp_path / "traces", "abcdef123456")
    t.record("prompt", {"text": "İ"})
    t.record("tool_call", {"arguments": {"a": [1, 2]}})
    lines = t.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["kind"] for l in lines] == ["prompt", "tool_call"]
    assert "İ" in lines[0] and "abcdef12" in t.path.name


def test_tracer_creates_nothing_until_record(tmp_path):
    t = JsonlTracer(tmp_path / "traces", "abc")
    assert t.path is None
    assert not (tmp_path / "traces").exists()


def test_null_tracer():
    NullTracer().record("x", {})
