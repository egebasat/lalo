import pytest

from agent_bot.registry import ToolRegistry
from agent_bot.tools import (
    calculator,
    fetch_page,
    get_current_time,
    register_python_tools,
    web_search,
)

def test_calculator_basic():
    assert calculator("0.175 * 2340") == "409.5"
    assert calculator("2**10") == "1024"


def test_calculator_rejects_import():
    with pytest.raises(ValueError):
        calculator("__import__('os')")


def test_calculator_rejects_attribute():
    with pytest.raises(ValueError):
        calculator("(1).__class__")


def test_calculator_rejects_huge_exponent():
    with pytest.raises(ValueError):
        calculator("9**999999")


def test_calculator_thousands_separator():
    with pytest.raises(ValueError, match="thousands separators"):
        calculator("0.175 * 2,340")


def test_calculator_rejects_huge_round_digits():
    with pytest.raises(ValueError, match="round"):
        calculator("round(1, -99999999)")


def test_calculator_rejects_nested_exponents():
    for expr in ("(((9**100)**100)**100)", "(((((9**100)**100)**100)**100)**100)"):
        with pytest.raises(ValueError, match="too large"):
            calculator(expr)


def test_calculator_overflow_is_value_error():
    for expr in ("1e308*10", "1e200**100"):
        with pytest.raises(ValueError):
            calculator(expr)


def test_calculator_bad_call_is_value_error():
    for expr in ("round(5, 1.5)", "abs()"):
        with pytest.raises(ValueError):
            calculator(expr)


def test_time_tokyo():
    out = get_current_time("Asia/Tokyo")
    assert "Asia/Tokyo" in out
    assert "+09:00" in out


def test_time_unknown_zone():
    with pytest.raises(ValueError, match="Unknown timezone"):
        get_current_time("Mars/Base")


def test_web_search_formats_results(monkeypatch):
    rows = [{"title": "Example Title", "href": "https://example.com/a", "body": "Some body"}]
    monkeypatch.setattr("ddgs.ddgs.DDGS.text", lambda self, q, max_results=5: rows)
    out = web_search("example")
    assert "Example Title" in out and "https://example.com/a" in out
    monkeypatch.setattr("ddgs.ddgs.DDGS.text", lambda self, q, max_results=5: [])
    assert "No results for" in web_search("example")


def test_fetch_page_rejects_non_http():
    with pytest.raises(ValueError):
        fetch_page("file:///etc/passwd")


def test_register_python_tools():
    reg = ToolRegistry()
    register_python_tools(reg)
    assert {t.name for t in reg.tools()} == {
        "web_search", "fetch_page", "get_current_time", "calculator",
    }
    assert all(t.source == "python" for t in reg.tools())
