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
        assert not re.search(r"\{\w*\}|%s", src), name  # no Python format fields


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
