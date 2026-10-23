"""tkinter chat window for the agent server: shows every event of a turn as it streams.

    python gui/app.py            # talks to $AGENT_URL, default http://127.0.0.1:8000

Talks to the server only over HTTP (imports nothing from agent_bot), so it runs from
any Python with tkinter and `requests`. Network calls run on background threads and
hand their results to the Tk main loop through a queue; widgets are touched only on
the main thread, which polls that queue every 50 ms.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
import uuid
from collections.abc import Iterable, Iterator
from tkinter import scrolledtext

import requests

try:
    from gui import renderer
except ImportError:  # run as a script (python gui/app.py): gui/ itself is on sys.path
    import renderer

AGENT_URL = os.environ.get("AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_RESULT_LINES = 15
POLL_MS = 50
HEALTH_EVERY_S = 2


def iter_events(lines: Iterable[bytes | str]) -> Iterator[dict]:
    """Parse an NDJSON stream; a line that isn't a JSON object becomes an error event."""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if isinstance(event, dict):
            yield event
        else:
            yield {"type": "error", "message": f"Bad line from server: {line[:200]}"}


def render_event(event: dict) -> list[tuple[str, str]]:
    """Turn one event into (text, tag) segments for the transcript. Tokens render nothing:
    they go to the live "generating…" region instead."""
    kind = event.get("type")
    if kind == "token":
        return []
    if kind == "thinking":
        return [(f"{event.get('text', '').strip()}\n", "thinking")]
    if kind == "tool_call":
        args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        return [(f"→ {event.get('name')}({args})  [{event.get('source')}]\n", "tool_call")]
    if kind == "tool_result":
        lines = event.get("content", "").splitlines() or [""]
        shown = lines[:MAX_RESULT_LINES] + (["…"] if len(lines) > MAX_RESULT_LINES else [])
        tag = "tool_error" if event.get("is_error") else "tool_result"
        return [("".join(f"    {line}\n" for line in shown), tag)]
    if kind == "answer":
        return [(f"{event.get('text', '')}\n", "answer")]
    if kind == "error":
        return [(f"⚠ {event.get('message', '')}\n", "error")]
    if kind == "done":
        return [(f"[{event.get('iterations')} iterations · {event.get('elapsed_s')} s · "
                 f"{event.get('tokens_generated')} tokens]\n\n", "meta")]
    return [(f"[unknown event: {kind}]\n", "meta")]


def render_user(text: str) -> list[tuple[str, str]]:
    return [(f"You: {text}\n", "user")]


def _http_error(resp: requests.Response) -> dict:
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return {"type": "error", "message": f"HTTP {resp.status_code}: {detail}"}


class App:
    def __init__(self, root: tk.Tk, url: str = AGENT_URL):
        self.root = root
        self.url = url
        self.session_id = uuid.uuid4().hex
        self.inbox: queue.Queue = queue.Queue()  # (kind, payload) from worker threads
        self.in_flight = 0  # /chat requests still open; the timer runs while > 0
        self.busy_since = 0.0
        self.images: list = []  # formula images in the transcript; Tk needs them referenced

        root.title("agent_bot")
        root.geometry("900x700")
        self._build()
        threading.Thread(target=self._watch_health, daemon=True).start()
        root.after(POLL_MS, self._poll)

    # --- layout -------------------------------------------------------------------

    def _build(self) -> None:
        top = tk.Frame(self.root, padx=8, pady=6)
        top.pack(fill="x")
        self.status = tk.StringVar(value="connecting…")
        tk.Label(top, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True)
        self.thinking = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="Thinking", variable=self.thinking).pack(side="left")
        tk.Button(top, text="New chat", command=self._new_chat).pack(side="left", padx=4)
        tk.Button(top, text="Tools", command=self._open_tools).pack(side="left")

        self.transcript = scrolledtext.ScrolledText(self.root, wrap="word", state="disabled",
                                                    padx=8, pady=8, font=("Helvetica", 13))
        self.transcript.pack(fill="both", expand=True, padx=8)
        mono = ("Menlo", 12)
        styles = {
            "user": {"font": ("Helvetica", 13, "bold"), "spacing1": 10},
            "thinking": {"foreground": "#888888", "font": ("Helvetica", 12, "italic")},
            "tool_call": {"foreground": "#1f5fbf", "font": mono},
            "tool_result": {"foreground": "#2e7d32", "font": mono},
            "tool_error": {"foreground": "#c62828", "font": mono},
            "answer": {"spacing1": 4},
            "error": {"foreground": "#c62828", "font": ("Helvetica", 13, "bold")},
            "meta": {"foreground": "#999999", "font": ("Helvetica", 11)},
            "live": {"foreground": "#aaaaaa", "font": mono},
        }
        for tag, style in styles.items():
            self.transcript.tag_configure(tag, **style)
        renderer.configure_tags(self.transcript, "Helvetica", 13, "Menlo")

        bottom = tk.Frame(self.root, padx=8, pady=8)
        bottom.pack(fill="x")
        self.input = tk.Text(bottom, height=3, wrap="word", font=("Helvetica", 13))
        self.input.pack(side="left", fill="x", expand=True)
        side = tk.Frame(bottom)
        side.pack(side="left", padx=(8, 0))
        tk.Button(side, text="Send", command=self._send).pack()
        self.elapsed = tk.StringVar(value="")
        tk.Label(side, textvariable=self.elapsed, foreground="#888888").pack()
        for key in ("<Command-Return>", "<Control-Return>"):
            self.input.bind(key, lambda _e: (self._send(), "break")[1])
        self.input.focus_set()

    # --- transcript ---------------------------------------------------------------

    def _append(self, segments: list[tuple[str, str]]) -> None:
        self.transcript.configure(state="normal")
        for text, tag in segments:
            self.transcript.insert("end", text, tag)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _clear_live(self) -> None:
        ranges = self.transcript.tag_ranges("live")
        if ranges:
            self.transcript.configure(state="normal")
            self.transcript.delete(ranges[0], ranges[-1])
            self.transcript.configure(state="disabled")

    def _show_event(self, event: dict) -> None:
        if event.get("type") == "token":
            if not self.transcript.tag_ranges("live"):
                self._append([("generating… ", "live")])
            self._append([(event.get("text", ""), "live")])
            return
        self._clear_live()
        if event.get("type") == "answer":
            self._append_answer(event.get("text", ""))
        else:
            self._append(render_event(event))

    def _append_answer(self, text: str) -> None:
        """Answers get Markdown and LaTeX rendering; every other event stays plain."""
        self.transcript.configure(state="normal")
        renderer.insert_markdown(self.transcript, text, "answer", self.images)
        self.transcript.insert("end", "\n", "answer")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    # --- actions (main thread) ----------------------------------------------------

    def _send(self) -> None:
        message = self.input.get("1.0", "end").strip()
        if not message:
            return
        self.input.delete("1.0", "end")
        self._append(render_user(message))
        # Sending while a turn runs is allowed on purpose: the server answers 409 and
        # that shows up as an error, which is clearer than a silently greyed-out button.
        if self.in_flight == 0:
            self.busy_since = time.monotonic()
        self.in_flight += 1
        body = {"session_id": self.session_id, "message": message,
                "enable_thinking": self.thinking.get()}
        threading.Thread(target=self._chat, args=(body,), daemon=True).start()

    def _new_chat(self) -> None:
        old, self.session_id = self.session_id, uuid.uuid4().hex
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self.images.clear()
        threading.Thread(target=self._post_quietly, args=("/reset", {"session_id": old}),
                         daemon=True).start()

    def _open_tools(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Tools")
        window.geometry("600x450")
        text = scrolledtext.ScrolledText(window, wrap="word", padx=8, pady=8)
        text.pack(fill="both", expand=True)
        text.insert("end", "loading…")
        text.configure(state="disabled")
        threading.Thread(target=self._fetch_tools, args=(text,), daemon=True).start()

    # --- workers (background threads; never touch widgets) ------------------------

    def _chat(self, body: dict) -> None:
        try:
            with requests.post(f"{self.url}/chat", json=body, stream=True,
                               timeout=(5, None)) as resp:
                if resp.status_code != 200:
                    self.inbox.put(("event", _http_error(resp)))
                else:
                    for event in iter_events(resp.iter_lines()):
                        self.inbox.put(("event", event))
        except requests.RequestException as exc:
            self.inbox.put(("event", {"type": "error", "message": f"Connection error: {exc}"}))
        self.inbox.put(("finished", None))

    def _watch_health(self) -> None:
        while True:
            try:
                health = requests.get(f"{self.url}/health", timeout=3).json()
                if health.get("model_loaded"):
                    self.inbox.put(("status", f"ready — {health['model']} ({health['backend']})"))
                    return
                self.inbox.put(("status", "loading model…"))
            except (requests.RequestException, ValueError):
                self.inbox.put(("status", "server unreachable — retrying"))
            time.sleep(HEALTH_EVERY_S)

    def _fetch_tools(self, widget: tk.Text) -> None:
        try:
            body = requests.get(f"{self.url}/tools", timeout=30).json()
            lines = [f"• {t['name']}  [{t['source']}]\n    {t['description']}" for t in body["tools"]]
            lines = lines or ["(no tools)"]
            servers = [f"• {s['name']}: {s['status']} ({s['tool_count']} tools)"
                       + (f" — {s['error']}" if s.get("error") else "")
                       for s in body.get("mcp_servers", [])]
            text = "Tools\n" + "\n".join(lines)
            text += "\n\nMCP servers\n" + ("\n".join(servers) or "(none)")
        except (requests.RequestException, ValueError, KeyError) as exc:
            text = f"Could not load tools: {exc}"
        self.inbox.put(("tools", (widget, text)))

    def _post_quietly(self, path: str, body: dict) -> None:
        try:
            requests.post(f"{self.url}{path}", json=body, timeout=5)
        except requests.RequestException:
            pass  # the server forgets sessions on restart anyway

    # --- the queue pump (main thread) ---------------------------------------------

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.inbox.get_nowait()
                if kind == "event":
                    self._show_event(payload)
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "finished":
                    self.in_flight -= 1
                elif kind == "tools":
                    widget, text = payload
                    if widget.winfo_exists():
                        widget.configure(state="normal")
                        widget.delete("1.0", "end")
                        widget.insert("end", text)
                        widget.configure(state="disabled")
        except queue.Empty:
            pass
        if self.in_flight:
            self.elapsed.set(f"{time.monotonic() - self.busy_since:.1f} s")
        self.root.after(POLL_MS, self._poll)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
