import pytest


@pytest.fixture
def app():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    from gui.app import App
    app = App(root, url="http://127.0.0.1:9")
    app._append([(f"line {i}\n", "answer") for i in range(200)])
    root.update()  # the window has no size until now, so start at the bottom explicitly
    app.transcript.see("end")
    root.update()
    yield app
    root.destroy()


def _scroll_up(app):
    app.transcript.yview_moveto(0.2)
    app.root.update()
    return app.transcript.yview()


def _at_bottom(app):
    app.root.update()
    return app.transcript.yview()[1] >= 0.999


def test_follows_new_output_when_at_bottom(app):
    assert _at_bottom(app)
    app._show_event({"type": "token", "text": "hello"})
    app._show_event({"type": "tool_call", "name": "calculator", "arguments": {}, "source": "python"})
    assert _at_bottom(app)


def test_tokens_do_not_move_a_scrolled_up_view(app):
    view = _scroll_up(app)
    for _ in range(50):
        app._show_event({"type": "token", "text": "tok "})
    app.root.update()
    assert app.transcript.yview()[0] == pytest.approx(view[0], abs=0.01)


def test_events_and_answer_do_not_move_a_scrolled_up_view(app):
    view = _scroll_up(app)
    app._show_event({"type": "token", "text": "tok "})
    app._show_event({"type": "thinking", "text": "hmm"})
    app._show_event({"type": "answer", "text": "**done**"})
    app._show_event({"type": "done", "iterations": 1, "elapsed_s": 1.0, "tokens_generated": 5})
    app.root.update()
    assert app.transcript.yview()[0] == pytest.approx(view[0], abs=0.01)


def test_sending_a_message_scrolls_to_the_bottom(app):
    _scroll_up(app)
    app._chat = lambda body: None  # no network
    app.input.insert("1.0", "hello")
    app._send()
    assert _at_bottom(app)
