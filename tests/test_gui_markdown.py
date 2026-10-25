import ast
from pathlib import Path

import pytest

from gui.renderer import Math, Span, parse_markdown, prepare_latex

RENDERER = Path(__file__).parent.parent / "gui" / "renderer.py"


def spans(segments) -> list[tuple[str, tuple]]:
    """Text spans with their style tags, newlines dropped, for compact asserts."""
    return [(s.text, s.tags) for s in segments if isinstance(s, Span) and s.text.strip()]


def text_of(segments) -> str:
    return "".join(s.text if isinstance(s, Span) else f"<{s.latex}>" for s in segments)


def maths(segments) -> list[Math]:
    return [s for s in segments if isinstance(s, Math)]


def test_bold_and_italic():
    assert spans(parse_markdown("Hello **world** and *you*")) == [
        ("Hello ", ()), ("world", ("strong",)), (" and ", ()), ("you", ("em",))]


def test_nested_bold_italic():
    assert ("both", ("strong", "em")) in spans(parse_markdown("***both***"))


def test_inline_code_and_heading():
    out = spans(parse_markdown("# Title\n\nUse `pip install`"))
    assert ("Title", ("h1",)) in out
    assert ("pip install", ("code",)) in out


def test_bullet_list_gets_bullets_and_depth():
    segs = parse_markdown("- one\n- two\n  - inner")
    text = text_of(segs)
    assert "• one" in text and "• two" in text and "• inner" in text
    depths = {s.tags for s in segs if isinstance(s, Span) and "inner" in s.text}
    assert depths == {("li2",)}


def test_ordered_list_numbers():
    text = text_of(parse_markdown("1. first\n2. second"))
    assert "1. first" in text and "2. second" in text


def test_paragraphs_are_separated():
    assert "One.\n\nTwo." in text_of(parse_markdown("One.\n\nTwo."))


def test_code_block_kept_verbatim():
    segs = parse_markdown("```python\nx = a*b*c\n```")
    assert ("x = a*b*c\n", ("code_block",)) in [(s.text, s.tags) for s in segs
                                                  if isinstance(s, Span)]


def test_table_rendered_as_aligned_monospace():
    segs = parse_markdown("| City | Pop |\n|---|---|\n| Istanbul | 15.7M |\n| Ankara | 5.8M |")
    table = "".join(s.text for s in segs if isinstance(s, Span) and s.tags == ("table",))
    lines = [line for line in table.splitlines() if line]
    assert lines[0].startswith("City") and "Pop" in lines[0]
    assert lines[2].startswith("Istanbul") and lines[3].startswith("Ankara")
    assert lines[0].index("Pop") == lines[2].index("15.7M")  # columns line up


@pytest.mark.parametrize("source", [r"\(x^2\)", r"$x^2$"])
def test_inline_math(source):
    [m] = maths(parse_markdown(f"so {source} grows"))
    assert m == Math("x^2", display=False)


@pytest.mark.parametrize("source", [r"\[\frac{a}{b}\]", r"$$\frac{a}{b}$$"])
def test_display_math(source):
    [m] = maths(parse_markdown(f"The ratio:\n\n{source}\n\nDone."))
    assert m == Math(r"\frac{a}{b}", display=True)


def test_inline_dollar_math_with_inner_spaces():
    # Real Qwen3 output (trace 20261023-123205): it pads inline formulas with spaces.
    text = r"Let $ C $ be a curve bounding $ D $, and $ \mathbf{F} = (P, Q) $ a field."
    assert [m.latex for m in maths(parse_markdown(text))] == [
        "C", "D", r"\mathbf{F} = (P, Q)"]


@pytest.mark.parametrize("text, expected", [
    ("Pay $5. Then $x$ is small.", ["x"]),
    ("The fee is $5 and $ x $ is the rate.", ["x"]),
])
def test_price_before_formula_on_same_line(text, expected):
    assert [m.latex for m in maths(parse_markdown(text))] == expected


@pytest.mark.parametrize("text", ["$  x  $ wide", "$ x$ lopsided", "$x $ lopsided"])
def test_only_one_matching_space_of_padding(text):
    assert maths(parse_markdown(text)) == []


def test_markdown_does_not_mangle_math():
    [m] = maths(parse_markdown(r"\( a_1 * b_2 * c_3 \)"))
    assert m.latex == "a_1 * b_2 * c_3"


@pytest.mark.parametrize("text", [
    "It costs $5 and $10 today.",
    "Prices: $5, $10, and $20.",
    "Between $1,000 and $2,000 per month.",
    "It went from $3.50 to $4.25 in a year.",
    "Only $5 left.",
])
def test_dollar_amounts_are_not_math(text):
    segs = parse_markdown(text)
    assert maths(segs) == []
    assert text in text_of(segs)


def test_math_inside_code_is_left_alone():
    segs = parse_markdown("Type `$x$` literally")
    assert maths(segs) == []
    assert ("$x$", ("code",)) in spans(segs)


def test_math_inside_bold_keeps_working():
    [m] = maths(parse_markdown(r"**Answer: \(409.5\)**"))
    assert m.latex == "409.5"


def test_prepare_latex_boxed_and_unsupported_commands():
    lines, boxed = prepare_latex(r"\displaystyle \boxed{\textbf{42}}")
    assert boxed is True
    assert lines == [r"\mathbf{42}"]


def test_prepare_latex_splits_aligned_lines():
    lines, boxed = prepare_latex(r"\begin{aligned} x &= 1 \\ y &= 2 \end{aligned}")
    assert boxed is False
    assert lines == ["x = 1", "y = 2"]


def test_renderer_imports_nothing_from_agent_bot():
    imported = set()
    for node in ast.walk(ast.parse(RENDERER.read_text())):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [m for m in imported if m == "agent_bot" or m.startswith("agent_bot.")]


class TestRenderMath:
    @pytest.fixture(autouse=True)
    def _needs_matplotlib(self):
        pytest.importorskip("matplotlib")

    def test_renders_png(self):
        from gui.renderer import render_math
        png = render_math(r"\frac{1}{2}", display=True, color="#000000")
        assert png is not None and png.startswith(b"\x89PNG")

    def test_boxed_renders(self):
        from gui.renderer import render_math
        assert render_math(r"\boxed{409.5}", display=True, color="#000000") is not None

    def test_invalid_latex_returns_none(self):
        from gui.renderer import render_math
        assert render_math(r"\frac{", display=False, color="#000000") is None

    def test_boxed_frame_is_not_cropped(self):
        import io

        from matplotlib.image import imread

        from gui.renderer import render_math
        alpha = imread(io.BytesIO(render_math(r"\boxed{x = 3}", display=True,
                                              color="#000000")))[:, :, 3]
        height = alpha.shape[0]
        # the frame's left edge: a column near the border that is opaque most of the way down
        edge_columns = [(alpha[:, c] > 0.5).sum() for c in range(min(6, alpha.shape[1]))]
        assert max(edge_columns) > height * 0.6


def test_display_math_has_one_blank_line_each_side():
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("matplotlib")
    from gui.renderer import configure_tags, insert_markdown
    from gui.theme import DARK
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    try:
        text = tk.Text(root)
        configure_tags(text, "Helvetica", 13, "Menlo", DARK)
        images: list = []
        insert_markdown(text, "Before:\n\n\\[x^2\\]\n\nAfter.", "answer", images)
        flat = "".join("<img>" if kind == "image" else value
                       for kind, value, _ in text.dump("1.0", "end-1c", text=True, image=True))
        assert flat == "Before:\n\n<img>\n\nAfter."
        assert len(images) == 1
    finally:
        root.destroy()


# Models often "show" a report by wrapping it in a ```markdown fence (trace
# 20261023-133723); the fence is presentation, so its content is rendered.
@pytest.mark.parametrize("info", ["markdown", "md", "Markdown"])
def test_markdown_fence_is_rendered_not_shown_as_code(info):
    text = f"Here's the report:\n\n```{info}\n# Title\n\n- **bold** item with $ x $\n```\n\nDone."
    segs = parse_markdown(text)
    assert not [s for s in segs if isinstance(s, Span) and "code_block" in s.tags]
    assert ("Title", ("h1",)) in spans(segs)
    assert ("bold", ("li1", "strong")) in spans(segs)
    assert [m.latex for m in maths(segs)] == ["x"]
    assert "Done." in text_of(segs)


def test_code_fences_inside_markdown_fence_stay_code():
    text = "```markdown\n## Example\n\n```python\nprint(1)\n```\n\nAfter the code.\n```"
    segs = parse_markdown(text)
    assert ("print(1)\n", ("code_block",)) in [(s.text, s.tags) for s in segs
                                                 if isinstance(s, Span)]
    assert ("Example", ("h2",)) in spans(segs)
    assert "After the code." in text_of(segs)


def test_unclosed_markdown_fence_is_still_rendered():
    segs = parse_markdown("```markdown\n# Cut off\n\nSome **text**")
    assert ("Cut off", ("h1",)) in spans(segs)
    assert ("text", ("strong",)) in spans(segs)


def test_other_fences_stay_code():
    segs = parse_markdown("```python\n# not a heading\n```")
    assert [s.tags for s in segs if isinstance(s, Span)] == [("code_block",)]


def test_link_span_carries_its_url():
    [link] = [s for s in parse_markdown("See [the docs](https://example.com/a) now")
              if "link" in s.tags]
    assert (link.text, link.href) == ("the docs", "https://example.com/a")


def test_styled_text_inside_a_link_keeps_the_url():
    segs = parse_markdown("[**bold** text](https://example.com)")
    assert {s.href for s in segs if isinstance(s, Span) and s.text.strip()} == {
        "https://example.com"}


def test_bare_url_becomes_a_link():
    pytest.importorskip("linkify_it")
    [link] = [s for s in parse_markdown("Go to https://example.com/x today") if "link" in s.tags]
    assert (link.text, link.href) == ("https://example.com/x", "https://example.com/x")


def test_clicking_a_link_opens_it(monkeypatch):
    tk = pytest.importorskip("tkinter")
    from gui import renderer
    from gui.theme import DARK
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    opened = []
    monkeypatch.setattr(renderer.webbrowser, "open", opened.append)
    try:
        text = tk.Text(root)
        renderer.configure_tags(text, "Helvetica", 13, "Menlo", DARK)
        renderer.insert_markdown(text, "[one](https://a.example) and [two](https://b.example)",
                                 "answer", [])
        for word in ("one", "two"):
            start = text.search(word, "1.0")
            [tag] = [t for t in text.tag_names(start) if t.startswith("href")]
            renderer.open_link(text, tag)
        assert opened == ["https://a.example", "https://b.example"]
    finally:
        root.destroy()
