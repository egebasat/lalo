"""Apple Notes and Mail tools for the host, run as in-process AppleScript.

Scripts are constants that read all text from `on run argv`; nothing the user or the
model writes is ever spliced into script source. They run through NSAppleScript
(PyObjC), compiled once and called with a `run` Apple event, which is several times
faster than starting `osascript` per call. Foundation is imported only when a script
runs, so this module imports anywhere (the Linux sandbox, CI).

Notes tools only ever touch one folder, pinned by its id in data/mac_tools.json.
"""
from __future__ import annotations

import json
import logging
import threading
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
