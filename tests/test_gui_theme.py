import re
from pathlib import Path

import pytest

from gui.theme import DARK

GUI = Path(__file__).parent.parent / "gui"

# Roles meant to be read; the rest (meta, live, hr) are deliberately dim.
READ_ROLES = ["fg", "user", "muted", "tool_call", "tool_result", "error", "link", "quote"]
DIM_ROLES = ["meta", "live", "hr"]


def luminance(hex_color: str) -> float:
    """WCAG relative luminance of #rrggbb."""
    def channel(c: int) -> float:
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_backgrounds_are_dark():
    for role in ("bg", "surface", "code_bg", "code_block_bg"):
        assert luminance(DARK[role]) < 0.05, role


@pytest.mark.parametrize("role", READ_ROLES)
def test_text_colors_readable_on_dark(role):
    assert contrast(DARK[role], DARK["bg"]) >= 4.5, role


@pytest.mark.parametrize("role", DIM_ROLES)
def test_dim_colors_still_visible(role):
    assert contrast(DARK[role], DARK["bg"]) >= 2.5, role


def test_code_readable_on_its_own_background():
    assert contrast(DARK["code_fg"], DARK["code_bg"]) >= 4.5
    assert contrast(DARK["code_fg"], DARK["code_block_bg"]) >= 4.5


def test_no_hardcoded_colors_outside_theme():
    for name in ("app.py", "renderer.py"):
        source = (GUI / name).read_text()
        assert not re.findall(r"""["']#[0-9a-fA-F]{3,6}["']""", source), name


@pytest.fixture
def root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    yield root
    root.destroy()


def test_renderer_tags_use_palette(root):
    import tkinter as tk

    from gui.renderer import configure_tags
    text = tk.Text(root)
    configure_tags(text, "Helvetica", 13, "Menlo", DARK)
    assert text.tag_cget("md_code", "background") == DARK["code_bg"]
    assert text.tag_cget("md_link", "foreground") == DARK["link"]


def test_app_is_dark(root):
    from gui.app import App
    app = App(root, url="http://127.0.0.1:9")
    assert app.transcript.cget("background") == DARK["bg"]
    assert app.transcript.cget("foreground") == DARK["fg"]
    assert app.input.cget("background") == DARK["surface"]
    assert app.transcript.tag_cget("tool_call", "foreground") == DARK["tool_call"]


def test_accent_readable_on_buttons():
    assert contrast(DARK["accent"], DARK["button_bg"]) >= 4.5
    assert contrast(DARK["accent"], DARK["button_hover"]) >= 4.5


def test_buttons_are_orange_on_theme(root):
    from gui.app import App
    app = App(root, url="http://127.0.0.1:9")
    assert set(app.buttons) == {"New chat", "Tools", "Send"}
    for button in app.buttons.values():
        assert button.cget("foreground") == DARK["accent"]
        assert button.cget("background") == DARK["button_bg"]


def test_button_click_runs_command(root):
    from gui.app import App
    app = App(root, url="http://127.0.0.1:9")
    app.input.insert("1.0", "hello")
    sent = []
    app._send = lambda: sent.append(True)  # the button must call the current handler
    button = app.buttons["Send"]
    root.update()
    button.event_generate("<ButtonPress-1>")
    button.event_generate("<ButtonRelease-1>", x=1, y=1)
    root.update()
    assert sent == [True]
