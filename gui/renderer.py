"""Markdown + LaTeX rendering for answers in the tkinter transcript.

Two halves:
  - `parse_markdown(text)` is pure: it turns an answer into `Span`s (text plus style
    names) and `Math` items, so it can be tested without a display.
  - `insert_markdown(widget, text, ...)` applies that to a `tk.Text`: spans become tags,
    and each formula becomes an inline image drawn by matplotlib's mathtext.

Math is cut out before Markdown parsing and put back afterwards. Otherwise Markdown
would eat the backslash in `\\[` and read `a_1 * b_2` as emphasis.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from functools import lru_cache

from markdown_it import MarkdownIt


@dataclass(frozen=True)
class Span:
    text: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Math:
    latex: str
    display: bool


Segment = Span | Math

_MD = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])

# One pass over the raw text. Code comes first so that `$x$` inside backticks or a
# fence stays code; then display math, then inline math. A single-dollar pair must hug
# its content, or be padded by exactly one space on both sides (Qwen3 writes `$ C $`),
# and its closing `$` must not be followed by a digit. A price's `$` is never padded
# ("$5"), so "$5 and $10" and "Pay $5. Then $x$" don't pair the wrong dollars.
_MATH_OR_CODE = re.compile(
    r"(?P<fence>^[ \t]*(?P<fq>```|~~~).*?^[ \t]*(?P=fq)[ \t]*$)"
    r"|(?P<code>(?P<bt>`+).+?(?P=bt))"
    r"|\$\$(?P<d1>.+?)\$\$"
    r"|\\\[(?P<d2>.+?)\\\]"
    r"|\\\((?P<i1>.+?)\\\)"
    r"|(?<![\\$\w])\$(?: (?P<i2>[^\s$](?:[^$\n]*?[^\s$])?) |(?P<i3>[^\s$](?:[^$\n]*?[^\s$])?))\$(?!\d)",
    re.S | re.M,
)
_PLACEHOLDER = re.compile("\ue000(\\d+)\ue001")
_STYLE_ORDER = ("strong", "em", "strike", "link", "code")


def _protect_math(text: str) -> tuple[str, list[tuple[Math, str]]]:
    """Swap each formula for a placeholder Markdown leaves alone; keep (Math, source)."""
    formulas: list[tuple[Math, str]] = []

    def swap(m: re.Match) -> str:
        if m["fence"] is not None or m["code"] is not None:
            return m[0]
        display = m["d1"] is not None or m["d2"] is not None
        latex = next(g for g in (m["d1"], m["d2"], m["i1"], m["i2"], m["i3"]) if g is not None)
        formulas.append((Math(latex.strip(), display), m[0]))
        return f"\ue000{len(formulas) - 1}\ue001"

    return _MATH_OR_CODE.sub(swap, text), formulas


class _Builder:
    def __init__(self, formulas: list[tuple[Math, str]]):
        self.formulas = formulas
        self.out: list[Segment] = []
        self.lists: list[list] = []  # one [ordered, next_number] per open list
        self.block: str | None = None  # h1..h3 or None
        self.quote = 0
        self.styles: list[str] = []
        self.table: list[list[str]] | None = None

    # --- helpers ------------------------------------------------------------------

    def restore(self, text: str) -> str:
        """Put formulas back as source text (for code, where they must stay literal)."""
        return _PLACEHOLDER.sub(lambda m: self.formulas[int(m[1])][1], text)

    def block_tags(self) -> tuple[str, ...]:
        tags = []
        if self.block:
            tags.append(self.block)
        if self.lists:
            tags.append(f"li{len(self.lists)}")
        if self.quote:
            tags.append("quote")
        return tuple(tags)

    def emit(self, text: str, extra: tuple[str, ...] = ()) -> None:
        if not text:
            return
        styles = sorted(set(self.styles) | set(extra), key=_STYLE_ORDER.index)
        tags = self.block_tags() + tuple(styles)
        pos = 0
        for m in _PLACEHOLDER.finditer(text):
            if m.start() > pos:
                self.out.append(Span(text[pos:m.start()], tags))
            self.out.append(self.formulas[int(m[1])][0])
            pos = m.end()
        if pos < len(text):
            self.out.append(Span(text[pos:], tags))

    def trailing_newlines(self) -> int:
        count = 0
        for seg in reversed(self.out):
            if not isinstance(seg, Span):
                return count
            stripped = seg.text.rstrip("\n")
            count += len(seg.text) - len(stripped)
            if stripped:
                return count
        return -1  # nothing emitted yet

    def ensure_newlines(self, n: int) -> None:
        have = self.trailing_newlines()
        if have >= 0 and have < n:
            self.out.append(Span("\n" * (n - have)))

    def start_block(self) -> None:
        """Blank line between top-level blocks; one line break inside lists."""
        self.ensure_newlines(1 if self.lists else 2)

    # --- token walk ---------------------------------------------------------------

    def walk(self, tokens) -> list[Segment]:
        for tok in tokens:
            handler = getattr(self, f"on_{tok.type}", None)
            if handler:
                handler(tok)
        while self.out and isinstance(self.out[-1], Span) and not self.out[-1].text.strip():
            self.out.pop()
        return self.out

    def on_heading_open(self, tok) -> None:
        self.start_block()
        self.block = f"h{min(int(tok.tag[1]), 3)}"

    def on_heading_close(self, tok) -> None:
        self.block = None
        self.ensure_newlines(1)

    def on_paragraph_open(self, tok) -> None:
        if not tok.hidden and not self._just_after_marker():
            self.start_block()

    def on_paragraph_close(self, tok) -> None:
        self.ensure_newlines(1)

    def _just_after_marker(self) -> bool:
        return bool(self.out) and getattr(self.out[-1], "text", "").endswith(("• ", ". "))

    def on_bullet_list_open(self, tok) -> None:
        self.start_block()
        self.lists.append([False, 1])

    def on_ordered_list_open(self, tok) -> None:
        self.start_block()
        self.lists.append([True, int(tok.attrGet("start") or 1)])

    def on_bullet_list_close(self, tok) -> None:
        self.lists.pop()
        self.ensure_newlines(1)

    on_ordered_list_close = on_bullet_list_close

    def on_list_item_open(self, tok) -> None:
        self.ensure_newlines(1)
        current = self.lists[-1]
        marker = f"{current[1]}. " if current[0] else "• "
        current[1] += 1
        self.out.append(Span(marker, self.block_tags()))

    def on_blockquote_open(self, tok) -> None:
        self.start_block()
        self.quote += 1

    def on_blockquote_close(self, tok) -> None:
        self.quote -= 1

    def on_fence(self, tok) -> None:
        self.start_block()
        self.out.append(Span(self.restore(tok.content), ("code_block",)))

    on_code_block = on_fence

    def on_hr(self, tok) -> None:
        self.start_block()
        self.out.append(Span("―" * 24 + "\n", ("hr",)))

    def on_table_open(self, tok) -> None:
        self.start_block()
        self.table = []

    def on_tr_open(self, tok) -> None:
        self.table.append([])

    def on_table_close(self, tok) -> None:
        rows, self.table = self.table, None
        widths = [max(len(row[i]) for row in rows if i < len(row))
                  for i in range(max(map(len, rows)))]
        lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip()
                 for row in rows]
        lines.insert(1, "  ".join("─" * w for w in widths))
        self.out.append(Span("\n".join(lines) + "\n", ("table",)))

    def on_inline(self, tok) -> None:
        if self.table is not None:  # table cells are plain text, formulas as source
            self.table[-1].append(self.restore("".join(
                c.content for c in tok.children or [] if c.type in ("text", "code_inline"))))
            return
        for child in tok.children or []:
            kind = child.type
            if kind in ("text", "html_inline"):
                self.emit(child.content)
            elif kind in ("softbreak", "hardbreak"):
                self.emit("\n")
            elif kind == "code_inline":
                self.emit(self.restore(child.content), ("code",))
            elif kind == "image":
                self.emit(child.content or "[image]")
            elif kind.endswith("_open") and kind[:-5] in ("strong", "em", "s", "link"):
                self.styles.append({"s": "strike"}.get(kind[:-5], kind[:-5]))
            elif kind.endswith("_close") and kind[:-6] in ("strong", "em", "s", "link"):
                self.styles.pop()


def parse_markdown(text: str) -> list[Segment]:
    protected, formulas = _protect_math(text)
    return _Builder(formulas).walk(_MD.parse(protected))


# --- LaTeX -> image -------------------------------------------------------------------

_ENVIRONMENTS = re.compile(
    r"\\(?:begin|end)\{(?:aligned|align\*?|gathered|gather\*?|split|equation\*?)\}")
_REWRITES = [(r"\displaystyle", ""), (r"\textstyle", ""), (r"\textbf{", r"\mathbf{"),
             (r"\textit{", r"\mathit{"), (r"\tfrac", r"\frac")]


def _unwrap_boxed(latex: str) -> tuple[str, bool]:
    start = latex.find(r"\boxed{")
    if start < 0:
        return latex, False
    depth, i = 1, start + len(r"\boxed{")
    while i < len(latex) and depth:
        depth += {"{": 1, "}": -1}.get(latex[i], 0)
        i += 1
    if depth:
        return latex, False  # unbalanced; leave it for mathtext to reject
    inner = latex[start + len(r"\boxed{"):i - 1]
    return latex[:start] + inner + latex[i:], True


def prepare_latex(latex: str) -> tuple[list[str], bool]:
    """Rewrite what models write into what mathtext supports: (lines, boxed)."""
    for old, new in _REWRITES:
        latex = latex.replace(old, new)
    latex, boxed = _unwrap_boxed(latex)
    latex = _ENVIRONMENTS.sub("", latex)
    lines = [re.sub(r"\s+", " ", line.replace("&", "")).strip()
             for line in latex.split(r"\\")]
    return [line for line in lines if line], boxed


@lru_cache(maxsize=256)
def render_math(latex: str, display: bool, color: str, fontsize: float = 13,
                dpi: float = 100) -> bytes | None:
    """PNG bytes for a formula, or None if matplotlib is missing or can't parse it."""
    lines, boxed = prepare_latex(latex)
    if not lines:
        return None
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return None
    fig = Figure()
    FigureCanvasAgg(fig)
    box = dict(boxstyle="square,pad=0.3", fill=False, edgecolor=color, linewidth=0.8)
    fig.text(0, 0, "\n".join(f"${line}$" for line in lines),
             fontsize=fontsize * (1.4 if display else 1.15), color=color,
             multialignment="center", bbox=box if boxed else None)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                    bbox_inches="tight", pad_inches=0.08 if boxed else 0.02)
    except Exception:  # mathtext raises ValueError for LaTeX it can't parse
        return None
    return buf.getvalue()


# --- Tk side --------------------------------------------------------------------------

def configure_tags(widget, family: str, size: int, mono: str) -> None:
    """Create the md_* tags. Call after the widget's other tags so these fonts win."""
    styles = {
        "md_b": {"font": (family, size, "bold")},
        "md_i": {"font": (family, size, "italic")},
        "md_bi": {"font": (family, size, "bold italic")},
        "md_code": {"font": (mono, size - 1), "background": "#eeeeee", "foreground": "#333333"},
        "md_code_block": {"font": (mono, size - 1), "background": "#f4f4f4",
                          "foreground": "#333333", "lmargin1": 12, "lmargin2": 12},
        "md_table": {"font": (mono, size - 1), "lmargin1": 8, "lmargin2": 8},
        "md_h1": {"font": (family, size + 6, "bold"), "spacing1": 6, "spacing3": 2},
        "md_h2": {"font": (family, size + 4, "bold"), "spacing1": 4, "spacing3": 2},
        "md_h3": {"font": (family, size + 2, "bold"), "spacing1": 2},
        "md_strike": {"overstrike": True},
        "md_link": {"foreground": "#1f5fbf", "underline": True},
        "md_quote": {"foreground": "#666666", "lmargin1": 16, "lmargin2": 16},
        "md_hr": {"foreground": "#bbbbbb"},
        "md_display": {"justify": "center", "spacing1": 4, "spacing3": 4},
    }
    for depth in range(1, 7):
        indent = 8 + 18 * (depth - 1)
        styles[f"md_li{depth}"] = {"lmargin1": indent, "lmargin2": indent + 16}
    for tag, style in styles.items():
        widget.tag_configure(tag, **style)


def _tk_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    """Map parser style names onto md_* tags; at most one of them sets the font."""
    names = set(tags)
    out = []
    font = next((f for f in ("code_block", "table", "code", "h1", "h2", "h3") if f in names), None)
    if font is None and {"strong", "em"} <= names:
        font = "bi"
    elif font is None and names & {"strong", "em"}:
        font = "b" if "strong" in names else "i"
    if font:
        out.append(f"md_{font}")
    out += [f"md_{t}" for t in tags if t.startswith("li") or t in ("strike", "link", "quote", "hr")]
    return tuple(out)


def _hex_color(widget) -> str:
    """The widget's text color as #rrggbb (Tk on macOS reports names like systemTextColor)."""
    r, g, b = widget.winfo_rgb(widget.cget("foreground"))
    return f"#{r >> 8:02x}{g >> 8:02x}{b >> 8:02x}"


def insert_markdown(widget, text: str, base_tag: str, images: list) -> None:
    """Insert rendered Markdown at the end of `widget` (which must be state=normal).

    `images` keeps the PhotoImages alive; Tk drops an image once Python forgets it.
    """
    import tkinter as tk

    color = _hex_color(widget)
    dpi = float(widget.tk.call("tk", "scaling")) * 72
    break_after = False  # a display formula must end its line
    for seg in parse_markdown(text):
        if isinstance(seg, Span):
            if break_after and not seg.text.startswith("\n"):
                widget.insert("end", "\n", base_tag)
            break_after = False
            widget.insert("end", seg.text, (base_tag, *_tk_tags(seg.tags)))
            continue
        png = render_math(seg.latex, seg.display, color, dpi=dpi)
        if png is None:  # unsupported LaTeX: show the source rather than lose it
            source = f"\\[{seg.latex}\\]" if seg.display else f"\\({seg.latex}\\)"
            widget.insert("end", source, (base_tag, "md_code"))
            continue
        image = tk.PhotoImage(data=base64.b64encode(png))
        images.append(image)
        if seg.display:
            if widget.get("end-2c") != "\n":
                widget.insert("end", "\n", base_tag)
            widget.image_create("end", image=image, align="center")
            widget.tag_add("md_display", "end-2c")
            break_after = True
        else:
            widget.image_create("end", image=image, align="center")
