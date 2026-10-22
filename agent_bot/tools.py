"""Built-in Python tools the model can call."""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from agent_bot.registry import ToolRegistry, function_tool, tool

_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _wmo(code: int) -> str:
    return _WMO.get(code, f"weather code {code}")


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information, news or facts. Returns titles, URLs and snippets.

    Args:
        query: Search keywords, e.g. 'python 3.13 release date'.
        max_results: Number of results to return, 1-10.
    """
    from ddgs import DDGS

    n = max(1, min(10, max_results))
    results = DDGS().text(query, max_results=n)
    if not results:
        return f"No results for '{query}'."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')}")
    return "\n".join(lines)


@tool
def fetch_page(url: str, max_chars: int = 4000) -> str:
    """Download a web page and return its readable text. Use after web_search to read a result.

    Args:
        url: Full http or https URL, e.g. 'https://example.com/article'.
        max_chars: Maximum number of characters to return.
    """
    import trafilatura

    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Only http and https URLs are supported")
    resp = httpx.get(
        url, follow_redirects=True, timeout=15, headers={"User-Agent": "agent_bot/0.1"}
    )
    resp.raise_for_status()
    text = trafilatura.extract(resp.text)
    if text is None:
        raise ValueError(f"Could not extract readable text from {url}")
    return text[:max_chars]


def _geocode(name: str) -> dict | None:
    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return results[0] if results else None


@tool
def get_weather(location: str, days: int = 1) -> str:
    """Get the current weather and a daily forecast for a city.

    Args:
        location: City name, e.g. 'Istanbul' or 'Paris'.
        days: Number of days to forecast, 1-16. Use 2 or more to include tomorrow.
    """
    if not 1 <= days <= 16:
        raise ValueError("days must be between 1 and 16")
    place = _geocode(location)
    if place is None and "," in location:
        place = _geocode(location.split(",", 1)[0].strip())
    if place is None:
        raise ValueError(f"Location '{location}' not found")

    resp = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,weather_code",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    cur, daily = data["current"], data["daily"]

    lines = [
        f"Weather for {place['name']}, {place.get('country', '')}",
        f"Now: {cur['temperature_2m']}°C, {_wmo(cur['weather_code'])}, "
        f"wind {cur['wind_speed_10m']} km/h",
    ]
    for i, date in enumerate(daily["time"]):
        lines.append(
            f"{date}: {daily['temperature_2m_min'][i]}°C to {daily['temperature_2m_max'][i]}°C, "
            f"rain {daily['precipitation_probability_max'][i]}%, "
            f"{_wmo(daily['weather_code'][i])}"
        )
    return "\n".join(lines)


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in a timezone.

    Args:
        timezone: IANA timezone name such as 'Asia/Tokyo' or 'Europe/Istanbul'.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise ValueError(
            f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/Istanbul'."
        )
    now = datetime.now(zone)
    return f"{now.isoformat(timespec='seconds')} ({now.strftime('%A')}), {timezone}"


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round}


def _eval(node: ast.AST):
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large (max 100)")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Tuple):
        raise ValueError("Write numbers without thousands separators, e.g. 2340 not 2,340")
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use it for any math instead of computing yourself.

    Args:
        expression: Plain arithmetic like '0.175 * 2340'. Supports + - * / // % ** and sqrt(), abs(), round(). No commas in numbers.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e.msg}")
    try:
        result = _eval(tree.body)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    if isinstance(result, float):
        result = round(result, 10)
        if result == int(result):
            result = int(result)
    return str(result)


def register_python_tools(registry: ToolRegistry) -> None:
    """Register every @tool function defined in this module."""
    for obj in list(globals().values()):
        if callable(obj) and getattr(obj, "__agent_tool__", False):
            registry.register(function_tool(obj, source="python"))
