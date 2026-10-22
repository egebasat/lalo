import httpx
import pytest
import respx

from agent_bot.registry import ToolRegistry
from agent_bot.tools import (
    calculator,
    fetch_page,
    get_current_time,
    get_weather,
    register_python_tools,
    web_search,
)

GEO = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"

HIT = {"results": [{"name": "Istanbul", "country": "Turkey", "latitude": 41.0, "longitude": 29.0}]}
FORECAST_JSON = {
    "current": {"temperature_2m": 18.5, "weather_code": 3, "wind_speed_10m": 12.0},
    "daily": {
        "time": ["2026-10-23", "2026-10-24"],
        "temperature_2m_max": [20.0, 21.0],
        "temperature_2m_min": [14.0, 15.0],
        "precipitation_probability_max": [10, 60],
        "weather_code": [1, 61],
    },
}


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


@respx.mock
def test_weather_formats_days():
    respx.get(GEO).mock(return_value=httpx.Response(200, json=HIT))
    respx.get(FORECAST).mock(return_value=httpx.Response(200, json=FORECAST_JSON))
    out = get_weather("Istanbul", days=2)
    assert "Istanbul" in out
    assert "2026-10-23" in out and "2026-10-24" in out
    assert "°C" in out


@respx.mock
def test_weather_retries_without_country():
    route = respx.get(GEO).mock(
        side_effect=[
            httpx.Response(200, json={"results": []}),
            httpx.Response(200, json=HIT),
        ]
    )
    respx.get(FORECAST).mock(return_value=httpx.Response(200, json=FORECAST_JSON))
    out = get_weather("Istanbul, Turkey")
    assert "Istanbul" in out
    assert route.calls[1].request.url.params["name"] == "Istanbul"


def test_weather_rejects_days_out_of_range():
    with pytest.raises(ValueError):
        get_weather("Istanbul", days=0)


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
        "web_search", "fetch_page", "get_weather", "get_current_time", "calculator",
    }
    assert all(t.source == "python" for t in reg.tools())
