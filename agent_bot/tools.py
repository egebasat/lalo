"""Built-in Python tools the model can call."""
from __future__ import annotations

import ast
import logging
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from agent_bot.registry import ToolRegistry, function_tool, tool

log = logging.getLogger(__name__)

MIN_PAGE_TEXT = 200  # less than this from the plain HTML usually means a JavaScript-only page


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
    if text is None or len(text) < MIN_PAGE_TEXT:
        # Probably rendered by JavaScript: let headless Chromium run it and try again.
        try:
            rendered = trafilatura.extract(_render_with_browser(url))
        except ImportError:  # no Playwright (e.g. on the host): keep what we have
            rendered = None
        except Exception as exc:
            log.warning("browser render of %s failed: %s: %s", url, type(exc).__name__, exc)
            rendered = None
        if rendered and len(rendered) > len(text or ""):
            text = rendered
    if not text:
        raise ValueError(f"Could not extract readable text from {url}")
    return text[:max_chars]


def _render_with_browser(url: str, timeout_s: float = 15) -> str:
    """The page's HTML after its JavaScript has run, from headless Chromium.

    The sync Playwright API is fine here: sync tools run in asyncio.to_thread, a thread
    with no event loop. Playwright is only installed in the sandbox image.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_s * 1000)
            return page.content()
        finally:
            browser.close()


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


_MAX_BITS = 4000
_TOO_LARGE = "Result too large"


def _check(value):
    """Reject results that are huge or non-finite."""
    if isinstance(value, int) and value.bit_length() > _MAX_BITS:
        raise ValueError(_TOO_LARGE)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(_TOO_LARGE)
    return value


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _eval(node: ast.AST):
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return _check(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("Exponent too large (max 100)")
            if _is_int(left) and _is_int(right) and left.bit_length() * abs(right) > _MAX_BITS:
                raise ValueError(_TOO_LARGE)
        return _check(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _check(_UNARY_OPS[type(node.op)](_eval(node.operand)))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        args = [_eval(a) for a in node.args]
        if node.func.id == "round" and len(args) == 2 and not (
                _is_int(args[1]) and abs(args[1]) <= 15):
            raise ValueError("round() digits must be a whole number between -15 and 15")
        return _check(_FUNCS[node.func.id](*args))
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
    except OverflowError:
        raise ValueError(_TOO_LARGE)
    except TypeError as exc:
        raise ValueError(f"Invalid expression: {exc}")
    if isinstance(result, float):
        result = round(result, 10)
        if math.isfinite(result) and result == int(result):
            result = int(result)
    return str(result)


def register_python_tools(registry: ToolRegistry) -> None:
    """Register every @tool function defined in this module."""
    for obj in list(globals().values()):
        if callable(obj) and getattr(obj, "__agent_tool__", False):
            registry.register(function_tool(obj, source="python"))
