"""The GUI's one color palette: dark. Every color in gui/ comes from here.

Text roles meet WCAG AA contrast (4.5:1) on `bg`; meta/live/hr are dim on purpose.
tests/test_gui_theme.py checks both.
"""

DARK = {
    # surfaces
    "bg": "#1e2126",             # transcript and window
    "surface": "#2a2e35",        # input box
    "select_bg": "#3e4a5e",
    "code_bg": "#2c3038",        # inline `code`
    "code_block_bg": "#262a31",  # fenced blocks
    # text
    "fg": "#e2e5e9",
    "user": "#ffffff",
    "muted": "#9aa1ab",          # thinking, elapsed timer
    "meta": "#6f7782",           # per-turn stats line
    "live": "#7c8490",           # streaming tokens
    "tool_call": "#79b8ff",
    "tool_result": "#85d996",
    "error": "#ff8a80",
    "code_fg": "#e8d4a2",
    "link": "#79b8ff",
    "quote": "#a9b1bb",
    "hr": "#5d6573",
    # buttons (drawn as labels: macOS native buttons ignore background colors)
    "accent": "#ffa04d",         # button text
    "button_bg": "#343a43",
    "button_hover": "#3f4651",
}
