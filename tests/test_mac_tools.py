import json
import re
import subprocess
import sys

import pytest

from agent_bot import mac_tools
from agent_bot.config import MacToolsConfig
from agent_bot.mac_tools import RS, SCRIPTS, AppleScriptError
from agent_bot.registry import ToolError


class FakeRunner:
    """Stands in for run_script: records calls, answers from `responses`."""

    def __init__(self):
        self.responses: dict[str, str | Exception] = {}
        self.calls: list[tuple[str, tuple]] = []

    def __call__(self, name: str, *argv: str) -> str:
        self.calls.append((name, argv))
        answer = self.responses.get(name, "")
        if isinstance(answer, Exception):
            raise answer
        return answer

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def state(tmp_path):
    return tmp_path / "mac.json"


@pytest.fixture
def fake(monkeypatch, state):
    runner = FakeRunner()
    monkeypatch.setattr(mac_tools, "run_script", runner)
    mac_tools.configure(MacToolsConfig(enabled=True), state)
    return runner


def test_import_without_pyobjc():
    code = "import sys; sys.modules['Foundation'] = None; import agent_bot.mac_tools"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_scripts_take_text_only_from_argv():
    assert SCRIPTS
    for name, src in SCRIPTS.items():
        assert src.strip().startswith("on run argv"), name
        # no Python format fields ({title}, %s); AppleScript's {} and {noteName} are fine
        assert not re.search(r"\{[a-z_][a-z0-9_]*\}|%s", src), name


def test_scripts_compile():
    """Compiling never contacts Notes or Mail; it catches AppleScript syntax errors."""
    foundation = pytest.importorskip("Foundation")
    for name, src in SCRIPTS.items():
        ok, err = foundation.NSAppleScript.alloc().initWithSource_(src).compileAndReturnError_(None)
        assert ok, f"{name}: {dict(err).get('NSAppleScriptErrorMessage')}"


def test_resolve_uses_stored_id(fake, state):
    state.write_text(json.dumps({"notes_folder_id": "F1"}))
    fake.responses["notes_folder_exists"] = "true"
    assert mac_tools.notes_folder_id() == "F1"
    assert "notes_create_folder" not in fake.names()


def test_resolve_creates_when_stored_id_gone(fake, state):
    state.write_text(json.dumps({"notes_folder_id": "F1"}))
    fake.responses.update(notes_folder_exists="false", notes_folder_names="",
                          notes_create_folder="F2")
    assert mac_tools.notes_folder_id() == "F2"
    assert ("notes_create_folder", ("Agent",)) in fake.calls
    assert json.loads(state.read_text()) == {"notes_folder_id": "F2"}


def test_resolve_creates_suffixed_folder_when_name_taken(fake, state):
    fake.responses.update(notes_folder_names="Notes" + RS + "Agent", notes_create_folder="F3")
    assert mac_tools.notes_folder_id() == "F3"
    assert ("notes_create_folder", ("Agent (agent_bot)",)) in fake.calls
    assert "notes_folder_exists" not in fake.names()  # no stored id to check


def test_resolve_cached(fake, state):
    state.write_text(json.dumps({"notes_folder_id": "F1"}))
    fake.responses["notes_folder_exists"] = "true"
    mac_tools.notes_folder_id()
    mac_tools.notes_folder_id()
    assert fake.names().count("notes_folder_exists") == 1


def test_automation_denied_message(fake):
    fake.responses["notes_folder_names"] = AppleScriptError(-1743, "not allowed")
    with pytest.raises(ToolError, match="Privacy & Security → Automation") as e:
        mac_tools._call("notes_folder_names")
    assert "Notes" in str(e.value)
    fake.responses["mail_compose"] = AppleScriptError(-1743, "not allowed")
    with pytest.raises(ToolError, match="Mail"):
        mac_tools._call("mail_compose", "s", "b", "a@x.com")


def test_other_errors_name_app_and_number(fake):
    fake.responses["notes_folder_names"] = AppleScriptError(-600, "not running")
    with pytest.raises(ToolError, match=r"Notes error -600: not running"):
        mac_tools._call("notes_folder_names")


# --- Notes tools -------------------------------------------------------------------

from agent_bot.mac_tools import (  # noqa: E402
    FS,
    _note_html,
    append_to_apple_note,
    create_apple_note,
    list_apple_notes,
    read_apple_note,
    search_apple_notes,
    update_apple_note,
)

NID = "x-coredata://1241ABA8-8908-4088-B5E3-899236DC070C/ICNote/p905"
NID2 = "x-coredata://1241ABA8-8908-4088-B5E3-899236DC070C/ICNote/p906"


def rec(note_id, name, y, mo, d, secs, text):
    return FS.join([note_id, name, str(y), str(mo), str(d), str(secs), text])


@pytest.fixture
def notes(fake, state):
    """fake, with the pinned folder already resolved to 'F'."""
    state.write_text(json.dumps({"notes_folder_id": "F"}))
    fake.responses["notes_folder_exists"] = "true"
    return fake


def test_create_note_text_only_in_argv(notes):
    before = dict(SCRIPTS)
    notes.responses["notes_create"] = NID + FS + 'He said "hi"'
    out = create_apple_note('He said "hi"', '"; do shell script "rm -rf ~"\n<b>x</b>')
    assert SCRIPTS == before
    name, argv = notes.calls[-1]
    assert name == "notes_create" and argv[0] == "F"
    assert "He said &quot;hi&quot;" in argv[1] and "&lt;b&gt;" in argv[1]
    assert out == f'Created note "He said "hi"" (id: {NID}).'


def test_body_html_blank_lines_and_escaping():
    assert _note_html("T", "a & b\n\n<c>\n") == (
        "<h1>T</h1><div>a &amp; b</div><div><br></div><div>&lt;c&gt;</div>")
    assert _note_html(None, "x") == "<div>x</div>"


def test_list_formats_and_sorts(notes):
    notes.responses["notes_list"] = RS.join([
        rec(NID, "Old", 2026, 1, 2, 3600, "Old\nfirst line\nsecond"),
        rec(NID2, "New", 2026, 10, 28, 53587, "New\nbody"),
    ])
    lines = list_apple_notes().splitlines()
    assert lines == [f"{NID2} | New | 2026-10-28T14:53 | body",
                     f"{NID} | Old | 2026-01-02T01:00 | first line second"]
    assert notes.calls[-1] == ("notes_list", ("F",))


def test_list_parses_hostile_text(notes):
    text = "Dentist\nFriday\t10am, Nişantaşı\n\nbring card"
    notes.responses["notes_list"] = rec(NID, "Dentist", 2026, 10, 28, 0, text)
    assert list_apple_notes() == f"{NID} | Dentist | 2026-10-28T00:00 | Friday\t10am, Nişantaşı  bring card"


def test_list_and_search_empty(notes):
    assert list_apple_notes() == "The Agent folder has no notes yet."
    assert search_apple_notes("zzz") == "No notes in the Agent folder match 'zzz'."
    assert notes.calls[-1] == ("notes_search", ("F", "zzz"))


def test_max_results_clamped(notes):
    notes.responses["notes_list"] = RS.join(
        rec(f"{NID}{i}", f"N{i}", 2026, 1, 1, i, f"N{i}") for i in range(60))
    assert len(list_apple_notes(500).splitlines()) == 50
    assert len(list_apple_notes(0).splitlines()) == 1


def test_preview_capped_at_200_chars(notes):
    notes.responses["notes_list"] = rec(NID, "T", 2026, 1, 1, 0, "T\n" + "x" * 500)
    assert list_apple_notes().endswith(" | " + "x" * 200)


def test_id_scripts_pass_note_then_folder(notes):
    notes.responses.update(notes_read=rec(NID, "Packing", 2026, 1, 1, 0, "Packing\npassport"),
                           notes_append="Packing", notes_set_body="Packing")
    read_apple_note(NID)
    assert notes.calls[-1] == ("notes_read", (NID, "F"))
    assert append_to_apple_note(NID, "sunscreen") == 'Added to note "Packing".'
    assert notes.calls[-1] == ("notes_append", (NID, "F", "<div>sunscreen</div>"))
    assert update_apple_note(NID, "charger") == 'Updated note "Packing".'
    assert notes.calls[-2] == ("notes_read", (NID, "F"))
    assert notes.calls[-1] == ("notes_set_body", (NID, "F", _note_html("Packing", "charger")))


def test_bad_note_id_rejected_without_script(notes):
    with pytest.raises(ToolError, match="isn't a Notes id"):
        read_apple_note("123")
    assert "notes_read" not in notes.names()


def test_empty_text_rejected(notes):
    with pytest.raises(ToolError, match="title must not be empty"):
        create_apple_note("  ", "x")
    with pytest.raises(ToolError, match="query must not be empty"):
        search_apple_notes("")
    with pytest.raises(ToolError, match="text must not be empty"):
        append_to_apple_note(NID, "\n")
    assert "notes_create" not in notes.names()


def test_note_outside_folder_message(notes):
    notes.responses["notes_read"] = AppleScriptError(-1728, "Can't get note id")
    with pytest.raises(ToolError, match="in the Agent folder") as e:
        read_apple_note(NID)
    assert "list_apple_notes" in str(e.value)


def test_read_format(notes):
    notes.responses["notes_read"] = rec(NID, "T", 2026, 10, 28, 53587, "T\nline 1\nline 2\n")
    assert read_apple_note(NID) == "Title: T\nModified: 2026-10-28T14:53\n\nline 1\nline 2"


# --- Registration ------------------------------------------------------------------

from agent_bot.config import AppConfig, ModelConfig  # noqa: E402
from agent_bot.registry import ToolRegistry  # noqa: E402

NOTES_TOOL_NAMES = ["create_apple_note", "list_apple_notes", "search_apple_notes",
                    "read_apple_note", "update_apple_note", "append_to_apple_note"]


def test_register_mac_tools(fake, state):
    reg = ToolRegistry()
    mac_tools.register_mac_tools(reg, MacToolsConfig(enabled=True), state)
    assert [t.name for t in reg.tools()] == NOTES_TOOL_NAMES
    assert all(t.source == "mac" and t.timeout_s == 60 for t in reg.tools())
    assert all("may still complete" in t.timeout_hint for t in reg.tools())
    assert fake.calls == []  # registering never launches Notes


def test_maybe_register_platform_and_flag(fake, monkeypatch):
    def attempt(platform, enabled):
        monkeypatch.setattr(sys, "platform", platform)
        cfg = AppConfig(model=ModelConfig(path="m"))
        cfg.mac_tools.enabled = enabled
        reg = ToolRegistry()
        return mac_tools.maybe_register_mac_tools(reg, cfg), len(reg.tools())

    assert attempt("linux", True) == (False, 0)
    assert attempt("darwin", False) == (False, 0)
    assert attempt("darwin", True) == (True, len(NOTES_TOOL_NAMES))
