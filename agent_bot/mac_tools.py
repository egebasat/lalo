"""Apple Notes and Mail tools for the host, run as in-process AppleScript.

Scripts are constants that read all text from `on run argv`; nothing the user or the
model writes is ever spliced into script source. They run through NSAppleScript
(PyObjC), compiled once and called with a `run` Apple event, which is several times
faster than starting `osascript` per call. Foundation is imported only when a script
runs, so this module imports anywhere (the Linux sandbox, CI).

Notes tools only ever touch one folder, pinned by its id in data/mac_tools.json.
"""
from __future__ import annotations

import html
import json
import logging
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path

from agent_bot.config import MacToolsConfig
from agent_bot.registry import ToolError

log = logging.getLogger(__name__)

DEFAULT_STATE = Path("data/mac_tools.json")
FS = "\x1f"  # between fields of one record
RS = "\x1e"  # between records

SCRIPTS: dict[str, str] = {
    "notes_folder_exists": """on run argv
    tell application "Notes" to return (exists folder id (item 1 of argv)) as text
end run""",
    "notes_folder_names": """on run argv
    tell application "Notes" to set folderNames to name of folders of default account
    set AppleScript's text item delimiters to (character id 30)
    set out to folderNames as text
    set AppleScript's text item delimiters to ""
    return out
end run""",
    "notes_create_folder": """on run argv
    tell application "Notes"
        set f to make new folder at default account with properties {name:(item 1 of argv)}
        return id of f
    end tell
end run""",
}

# Handlers appended to the scripts that return notes. A record is: id, name, year, month,
# day, seconds since midnight, plaintext; joined by FS, records by RS. Dates go as numbers
# because AppleScript's date-to-text depends on the user's locale.
_RECORDS = """

on clean(t)
    set AppleScript's text item delimiters to {character id 30, character id 31}
    set parts to text items of t
    set AppleScript's text item delimiters to ""
    return parts as text
end clean

on packNotes(idList, nameList, dateList, textList)
    set FS to character id 31
    set out to {}
    repeat with i from 1 to count of idList
        set d to item i of dateList
        set end of out to (item i of idList) & FS & my clean(item i of nameList) & FS & (year of d as text) & FS & ((month of d as integer) as text) & FS & (day of d as text) & FS & (time of d as text) & FS & my clean(item i of textList)
    end repeat
    set AppleScript's text item delimiters to (character id 30)
    set joined to out as text
    set AppleScript's text item delimiters to ""
    return joined
end packNotes"""

# Every script that takes a note id looks it up *inside* the pinned folder
# (`note id X of folder id F`), which fails with -1728 for a note anywhere else. That
# lookup is the folder boundary. Reading `name` first forces it before any write.
SCRIPTS.update({
    "notes_create": """on run argv
    tell application "Notes"
        set n to make new note at folder id (item 1 of argv) with properties {body:(item 2 of argv)}
        return (id of n) & (character id 31) & my clean(name of n)
    end tell
end run""" + _RECORDS,
    "notes_list": """on run argv
    tell application "Notes"
        set f to folder id (item 1 of argv)
        set idList to id of notes of f
        set nameList to name of notes of f
        set dateList to modification date of notes of f
        set textList to plaintext of notes of f
    end tell
    return my packNotes(idList, nameList, dateList, textList)
end run""" + _RECORDS,
    "notes_search": """on run argv
    set q to item 2 of argv
    tell application "Notes"
        set f to folder id (item 1 of argv)
        set idList to id of (notes of f whose name contains q or plaintext contains q)
        set nameList to name of (notes of f whose name contains q or plaintext contains q)
        set dateList to modification date of (notes of f whose name contains q or plaintext contains q)
        set textList to plaintext of (notes of f whose name contains q or plaintext contains q)
    end tell
    return my packNotes(idList, nameList, dateList, textList)
end run""" + _RECORDS,
    "notes_read": """on run argv
    tell application "Notes"
        set n to note id (item 1 of argv) of folder id (item 2 of argv)
        set noteName to name of n
        set r to my packNotes({id of n}, {noteName}, {modification date of n}, {plaintext of n})
    end tell
    return r
end run""" + _RECORDS,
    "notes_set_body": """on run argv
    tell application "Notes"
        set n to note id (item 1 of argv) of folder id (item 2 of argv)
        set noteName to name of n
        set body of n to (item 3 of argv)
        return noteName
    end tell
end run""",
    "notes_append": """on run argv
    tell application "Notes"
        set n to note id (item 1 of argv) of folder id (item 2 of argv)
        set noteName to name of n
        set body of n to (body of n) & (item 3 of argv)
        return noteName
    end tell
end run""",
})

# Scripts whose first argument is a note id looked up inside the pinned folder.
_NOTE_ID_SCRIPTS = {"notes_read", "notes_set_body", "notes_append"}


class AppleScriptError(Exception):
    def __init__(self, number: int, message: str):
        super().__init__(f"{number}: {message}")
        self.number = number
        self.message = message


_run_lock = threading.Lock()  # NSAppleScript isn't documented as thread-safe
_compiled: dict[str, object] = {}


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode(), "big")


def _raise_from(err) -> None:
    err = dict(err or {})
    raise AppleScriptError(int(err.get("NSAppleScriptErrorNumber", -1)),
                           str(err.get("NSAppleScriptErrorMessage", "unknown error")))


def run_source(source: str, *argv: str) -> str:
    """Run an `on run argv` script in this process; argv arrives as a list of strings."""
    from Foundation import NSAppleEventDescriptor, NSAppleScript

    with _run_lock:
        script = _compiled.get(source)
        if script is None:
            script = NSAppleScript.alloc().initWithSource_(source)
            ok, err = script.compileAndReturnError_(None)
            if not ok:
                _raise_from(err)
            _compiled[source] = script
        args = NSAppleEventDescriptor.listDescriptor()
        for i, arg in enumerate(argv, 1):
            args.insertDescriptor_atIndex_(NSAppleEventDescriptor.descriptorWithString_(arg), i)
        event = NSAppleEventDescriptor.appleEventWithEventClass_eventID_targetDescriptor_returnID_transactionID_(
            _fourcc("aevt"), _fourcc("oapp"), NSAppleEventDescriptor.nullDescriptor(), -1, 0)
        event.setParamDescriptor_forKeyword_(args, _fourcc("----"))
        result, err = script.executeAppleEvent_error_(event, None)
        if result is None:
            _raise_from(err)
        return result.stringValue() or ""


def run_script(name: str, *argv: str) -> str:
    return run_source(SCRIPTS[name], *argv)


def _call(name: str, *argv: str) -> str:
    """run_script, with AppleScript errors turned into messages the model can act on."""
    app = "Mail" if name.startswith("mail_") else "Notes"
    try:
        return run_script(name, *argv)
    except AppleScriptError as e:
        if e.number == -1743:
            raise ToolError(
                f"macOS blocked control of {app}. Allow it in System Settings → Privacy & "
                "Security → Automation for the app that started the agent server, then try again."
            ) from None
        if e.number == -1728 and name in _NOTE_ID_SCRIPTS:
            raise ToolError(
                f"No note with id {argv[0]} in the Agent folder. Use list_apple_notes or "
                "search_apple_notes to find the right id."
            ) from None
        raise ToolError(f"{app} error {e.number}: {e.message}") from None


_cfg = MacToolsConfig()
_state_path = DEFAULT_STATE
_folder_id: str | None = None
_resolve_lock = threading.Lock()  # so two first calls can't both create a folder


def configure(cfg: MacToolsConfig, state_path: Path = DEFAULT_STATE) -> None:
    global _cfg, _state_path, _folder_id
    _cfg, _state_path, _folder_id = cfg, Path(state_path), None


def notes_folder_id() -> str:
    """The pinned folder's id: the stored one if it still exists, else a new folder.

    An existing folder is never adopted, even one with the configured name, because it
    could hold the owner's own notes.
    """
    global _folder_id
    with _resolve_lock:
        if _folder_id:
            return _folder_id
        stored = None
        if _state_path.exists():
            stored = json.loads(_state_path.read_text(encoding="utf-8")).get("notes_folder_id")
        if stored and _call("notes_folder_exists", stored) == "true":
            _folder_id = stored
            return stored
        if stored:
            log.warning("Notes folder %s is gone; creating a new one", stored)
        taken = set(_call("notes_folder_names").split(RS))
        name = _cfg.notes_folder if _cfg.notes_folder not in taken else f"{_cfg.notes_folder} (agent_bot)"
        _folder_id = _call("notes_create_folder", name)
        _state_path.parent.mkdir(parents=True, exist_ok=True)
        _state_path.write_text(json.dumps({"notes_folder_id": _folder_id}), encoding="utf-8")
        return _folder_id


# --- Notes tools -------------------------------------------------------------------

_NOTE_ID = re.compile(r"^x-coredata://.+/ICNote/p\d+$")
PREVIEW_CHARS = 200


def _require(param: str, value: str) -> None:
    if not value.strip():
        raise ToolError(f"{param} must not be empty.")


def _check_id(note_id: str) -> None:
    if not _NOTE_ID.match(note_id):
        raise ToolError("That isn't a Notes id; use the id from list_apple_notes or search_apple_notes.")


def _note_html(title: str | None, text: str) -> str:
    """Note body HTML: the title as a heading (Notes names the note after it), one div per line."""
    head = f"<h1>{html.escape(title)}</h1>" if title is not None else ""
    lines = text.rstrip("\n").split("\n")
    return head + "".join(f"<div>{html.escape(line)}</div>" if line else "<div><br></div>"
                          for line in lines)


def _parse(raw: str) -> list[dict]:
    notes = []
    for record in filter(None, raw.split(RS)):
        note_id, name, y, mo, d, secs, text = record.split(FS)
        modified = datetime(int(y), int(mo), int(d)) + timedelta(seconds=int(secs))
        notes.append({"id": note_id, "title": name, "modified": modified,
                      "text": _without_title(name, text)})
    return notes


def _without_title(name: str, text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].strip() == name.strip():
        lines = lines[1:]
    return "\n".join(lines).strip("\n")


def _iso(modified: datetime) -> str:
    return modified.isoformat(timespec="minutes")


def _summaries(raw: str, max_results: int) -> list[str]:
    notes = sorted(_parse(raw), key=lambda n: n["modified"], reverse=True)
    n = max(1, min(50, max_results))
    return [f"{x['id']} | {x['title']} | {_iso(x['modified'])} | "
            f"{x['text'].replace(chr(10), ' ')[:PREVIEW_CHARS]}" for x in notes[:n]]


def create_apple_note(title: str, body: str) -> str:
    """Create a new note in your own folder in Apple Notes. Returns the note's id.

    Only notes in your own Notes folder exist for these tools; the user's other notes
    are not reachable.

    Args:
        title: Short title, e.g. 'Packing list'.
        body: The note's text; use one line per item for lists.
    """
    _require("title", title)
    _require("body", body)
    out = _call("notes_create", notes_folder_id(), _note_html(title, body))
    note_id, name = out.split(FS)
    return f'Created note "{name}" (id: {note_id}).'


def list_apple_notes(max_results: int = 20) -> str:
    """List the notes in your own Apple Notes folder, most recently changed first.

    Each line is: id | title | last modified | start of the text.

    Args:
        max_results: How many notes to return, 1-50.
    """
    lines = _summaries(_call("notes_list", notes_folder_id()), max_results)
    return "\n".join(lines) if lines else "The Agent folder has no notes yet."


def search_apple_notes(query: str, max_results: int = 10) -> str:
    """Find notes in your own Apple Notes folder whose title or text contains the query.

    Use this to find a note's id before reading or changing it.

    Args:
        query: Word or phrase to look for (case-insensitive), e.g. 'packing'.
        max_results: How many notes to return, 1-50.
    """
    _require("query", query)
    lines = _summaries(_call("notes_search", notes_folder_id(), query), max_results)
    return "\n".join(lines) if lines else f"No notes in the Agent folder match '{query}'."


def read_apple_note(note_id: str) -> str:
    """Read the full text of one note from your own Apple Notes folder.

    Args:
        note_id: The id from list_apple_notes or search_apple_notes.
    """
    _check_id(note_id)
    note = _parse(_call("notes_read", note_id, notes_folder_id()))[0]
    return f"Title: {note['title']}\nModified: {_iso(note['modified'])}\n\n{note['text']}"


def update_apple_note(note_id: str, body: str) -> str:
    """Replace the whole text of a note in your own Apple Notes folder; the title stays.

    To add items to a note, use append_to_apple_note instead.

    Args:
        note_id: The id from list_apple_notes or search_apple_notes.
        body: The new full text, without the title.
    """
    _check_id(note_id)
    _require("body", body)
    folder = notes_folder_id()
    title = _parse(_call("notes_read", note_id, folder))[0]["title"]
    name = _call("notes_set_body", note_id, folder, _note_html(title, body))
    return f'Updated note "{name}".'


def append_to_apple_note(note_id: str, text: str) -> str:
    """Add lines to the end of a note in your own Apple Notes folder.

    Prefer this for "add X to my list": it keeps everything already in the note.

    Args:
        note_id: The id from list_apple_notes or search_apple_notes.
        text: The text to add; one line per item.
    """
    _check_id(note_id)
    _require("text", text)
    name = _call("notes_append", note_id, notes_folder_id(), _note_html(None, text))
    return f'Added to note "{name}".'


NOTES_TOOLS = [create_apple_note, list_apple_notes, search_apple_notes,
               read_apple_note, update_apple_note, append_to_apple_note]
