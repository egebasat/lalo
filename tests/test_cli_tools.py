"""Where the CLI's tools come from: the sandbox by default, in-process with --local-tools."""
import argparse
import socket

import pytest

from agent_bot import cli
from agent_bot.config import AgentConfig, AppConfig, McpServerConfig, ModelConfig
from tests.fakes import FakeBackend

CALL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "6 * 7"}}\n</tool_call>'


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    prompt = tmp_path / "system.md"
    prompt.write_text("Test assistant. {today}")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    cfg = AppConfig(model=ModelConfig(path="models/fake"),
                    agent=AgentConfig(system_prompt_file=str(prompt)),
                    mcp_servers={"sandbox": McpServerConfig(url=f"http://127.0.0.1:{dead}/mcp")})
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "create_backend", lambda *a, **kw: FakeBackend([CALL, "It is 42."]))

    async def run(*flags: str) -> str:
        parser = cli.build_parser()
        await cli._main(parser.parse_args(["what is 6*7?", *flags]))

    return run


async def test_cli_uses_sandbox_by_default(run_cli, capsys):
    await run_cli()
    out = capsys.readouterr().out
    assert "Tool sandbox unreachable" in out
    assert "Unknown tool 'calculator'" in out


async def test_cli_local_tools_flag_runs_python_tools_in_process(run_cli, capsys):
    await run_cli("--local-tools")
    out = capsys.readouterr().out
    assert "Tool sandbox unreachable" not in out
    assert "calculator" in out and "42" in out
    assert "Unknown tool" not in out
