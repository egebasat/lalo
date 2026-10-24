"""Prompts sent while a turn runs wait in the GUI and go out one by one, in order."""
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
    app.requests = []  # (turn, message) for each /chat the GUI starts
    app._chat = lambda body, turn: app.requests.append((turn, body["message"]))
    yield app
    root.destroy()


def send(app, text):
    app.input.insert("1.0", text)
    app._send()


def pump(app, *items):
    for item in items:
        app.inbox.put(item)
    app._poll()


def transcript(app):
    return app.transcript.get("1.0", "end-1c")


def test_prompts_sent_while_busy_wait_their_turn(app):
    send(app, "first")
    send(app, "second")
    send(app, "third")
    assert [m for _, m in app.requests] == ["first"]
    assert app.queued.get() == "2 queued"

    pump(app, ("finished", (app.requests[0][0], True)))
    assert [m for _, m in app.requests] == ["first", "second"]
    assert app.queued.get() == "1 queued"

    pump(app, ("finished", (app.requests[1][0], True)))
    assert [m for _, m in app.requests] == ["first", "second", "third"]
    assert app.queued.get() == ""


def test_user_line_appears_when_the_prompt_is_sent(app):
    send(app, "first")
    turn = app.requests[0][0]
    pump(app, ("event", (turn, {"type": "token", "text": "partial"})))
    send(app, "second")
    assert "You: second" not in transcript(app)

    pump(app, ("event", (turn, {"type": "answer", "text": "one"})),
         ("finished", (turn, True)))
    text = transcript(app)
    assert text.index("You: first") < text.index("one") < text.index("You: second")


def test_thinking_setting_is_captured_when_queued(app):
    bodies = []
    app._chat = lambda body, turn: bodies.append(body)
    send(app, "first")
    app.thinking.set(False)
    send(app, "second")
    app.thinking.set(True)
    pump(app, ("finished", (1, True)))
    assert bodies[1]["enable_thinking"] is False


def test_new_chat_drops_the_queue(app):
    send(app, "first")
    send(app, "second")
    app._new_chat()
    assert app.queued.get() == ""
    pump(app, ("finished", (app.requests[0][0], True)))
    assert [m for _, m in app.requests] == ["first"]


def test_stray_error_does_not_clear_live_output(app):
    send(app, "first")
    turn = app.requests[0][0]
    pump(app, ("event", (turn, {"type": "token", "text": "streaming text"})),
         ("event", (999, {"type": "error", "message": "HTTP 409: busy"})),
         ("finished", (999, True)))
    text = transcript(app)
    assert "streaming text" in text
    assert "HTTP 409: busy" in text
    assert app.turn == turn  # the stray request's end doesn't end the real turn


def test_error_event_keeps_the_queue_going(app):
    send(app, "first")
    send(app, "second")
    turn = app.requests[0][0]
    pump(app, ("event", (turn, {"type": "error", "message": "boom"})),
         ("finished", (turn, True)))
    assert [m for _, m in app.requests] == ["first", "second"]


def test_connection_error_stops_the_queue_and_keeps_prompts(app):
    send(app, "first")
    send(app, "second")
    turn = app.requests[0][0]
    pump(app, ("event", (turn, {"type": "error", "message": "Connection error: refused"})),
         ("finished", (turn, False)))
    assert [m for _, m in app.requests] == ["first"]
    assert "1 queued prompt not sent" in transcript(app)

    send(app, "third")  # the next Send retries, oldest prompt first
    assert [m for _, m in app.requests] == ["first", "second"]
    assert app.queued.get() == "1 queued"
