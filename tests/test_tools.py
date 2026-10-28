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


ARTICLE = ("<html><body><article><h1>Careers at Example</h1>"
           + "".join(f"<p>Paragraph {i}: we build useful software with kind, curious people "
                     f"and we are hiring engineers in several cities.</p>" for i in range(6))
           + "</article></body></html>")
JS_SHELL = '<html><body><div id="root"></div><script src="/app.js"></script></body></html>'


def test_fetch_page_uses_browser_when_js_shell(monkeypatch):
    import respx
    import agent_bot.tools as tools

    rendered = []
    monkeypatch.setattr(tools, "_render_with_browser",
                        lambda url, timeout_s=15: rendered.append(url) or ARTICLE)
    with respx.mock:
        respx.get("https://example.com/jobs").respond(200, html=JS_SHELL)
        text = fetch_page("https://example.com/jobs")
    assert rendered == ["https://example.com/jobs"]
    assert "hiring engineers" in text


def test_fetch_page_skips_browser_when_text_found(monkeypatch):
    import respx
    import agent_bot.tools as tools

    def no_browser(url, timeout_s=15):
        raise AssertionError("browser should not be used")

    monkeypatch.setattr(tools, "_render_with_browser", no_browser)
    with respx.mock:
        respx.get("https://example.com/post").respond(200, html=ARTICLE)
        assert "hiring engineers" in fetch_page("https://example.com/post")


def test_fetch_page_without_playwright_keeps_error(monkeypatch):
    import respx
    import agent_bot.tools as tools

    def missing(url, timeout_s=15):
        raise ImportError("No module named 'playwright'")

    monkeypatch.setattr(tools, "_render_with_browser", missing)
    with respx.mock:
        respx.get("https://example.com/jobs").respond(200, html=JS_SHELL)
        with pytest.raises(ValueError, match="Could not extract readable text"):
            fetch_page("https://example.com/jobs")
