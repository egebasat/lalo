import pytest

from agent_bot.parsing import Qwen3Parser, get_parser

p = Qwen3Parser()


def _call(payload: str) -> str:
    return f"<tool_call>\n{payload}\n</tool_call>"


def test_plain_answer():
    out = p.parse("Hello")
    assert out.thinking is None
    assert out.tool_calls == []
    assert out.content == "Hello"
    assert out.parse_errors == []


def test_thinking_and_answer():
    out = p.parse("<think>\nhmm\n</think>\n\nHi")
    assert out.thinking == "hmm"
    assert out.content == "Hi"


def test_empty_thinking_is_none():
    out = p.parse("<think>\n\n</think>\n\nHi")
    assert out.thinking is None


def test_closing_tag_only():
    out = p.parse("hmm</think>Hi")
    assert out.thinking == "hmm"
    assert out.content == "Hi"


def test_unclosed_think():
    out = p.parse("<think>long thought")
    assert out.thinking == "long thought"
    assert out.content == ""
    assert len(out.parse_errors) == 1
    assert "unclosed <think>" in out.parse_errors[0]


def test_single_tool_call():
    out = p.parse(
        _call('{"name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}}')
    )
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "get_current_time"
    assert out.tool_calls[0].arguments == {"timezone": "Asia/Tokyo"}
    assert out.content == ""


def test_multiple_tool_calls():
    raw = _call('{"name": "a", "arguments": {"x": 1}}') + "\n" + _call(
        '{"name": "b", "arguments": {}}'
    )
    out = p.parse(raw)
    assert [c.name for c in out.tool_calls] == ["a", "b"]
    assert out.tool_calls[0].arguments == {"x": 1}


def test_arguments_as_string():
    out = p.parse(_call('{"name": "search", "arguments": "{\\"query\\": \\"x\\"}"}'))
    assert out.tool_calls[0].arguments == {"query": "x"}


def test_code_fence_inside_tag():
    out = p.parse(_call('```json\n{"name": "a", "arguments": {"k": "v"}}\n```'))
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].arguments == {"k": "v"}


def test_missing_arguments_defaults_empty():
    out = p.parse(_call('{"name": "list_notes"}'))
    assert out.tool_calls[0].name == "list_notes"
    assert out.tool_calls[0].arguments == {}


def test_broken_json():
    raw = _call('{"name": "a", "arguments": {}}') + _call('{"name": "b", "argu')
    out = p.parse(raw)
    assert len(out.tool_calls) == 1
    assert len(out.parse_errors) == 1
    assert out.parse_errors[0].startswith("invalid JSON in <tool_call> #2: ")


def test_missing_name():
    out = p.parse(_call('{"arguments": {}}'))
    assert out.tool_calls == []
    assert len(out.parse_errors) == 1
    assert "missing 'name'" in out.parse_errors[0]


def test_unknown_family():
    with pytest.raises(ValueError, match="No parser for model family 'gemma'"):
        get_parser("gemma")
