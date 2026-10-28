from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_bot.config import load_config

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def write_config(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(REPO_CONFIG.read_text())
    mutate(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_loads_defaults_from_yaml():
    cfg = load_config(REPO_CONFIG, env={})
    assert cfg.model.path == "models/Qwen3-4B-8bit"
    assert cfg.model.family == "qwen3"
    assert cfg.model.backend == "mlx"
    assert cfg.server.host == "127.0.0.1"
    assert cfg.mcp_servers["sandbox"].url == "http://127.0.0.1:8001/mcp"
    assert cfg.sandbox.port == 8001
    assert cfg.sandbox.mcp_servers["fetch"].enabled is False
    assert cfg.agent.local_python_tools is False  # tools run in the Docker sandbox
    assert cfg.agent.max_iterations == 6
    assert cfg.tools.timeout_s == 20
    assert cfg.tools.max_result_chars == 4000


def test_env_override_nested():
    cfg = load_config(
        REPO_CONFIG,
        env={"AGENT__MODEL__PATH": "/models/x", "AGENT__AGENT__MAX_ITERATIONS": "3"},
    )
    assert cfg.model.path == "/models/x"
    assert cfg.agent.max_iterations == 3


def test_mcp_entry_requires_command_or_url(tmp_path):
    path = write_config(tmp_path, lambda raw: raw["mcp_servers"].update(bad={"enabled": True}))
    with pytest.raises(ValidationError):
        load_config(path, env={})


def test_unknown_backend_rejected(tmp_path):
    path = write_config(tmp_path, lambda raw: raw["model"].update(backend="foo"))
    with pytest.raises(ValidationError):
        load_config(path, env={})


def test_env_override_local_tools():
    cfg = load_config(REPO_CONFIG, env={"AGENT__AGENT__LOCAL_PYTHON_TOOLS": "false"})
    assert cfg.agent.local_python_tools is False


def test_mac_tools_defaults():
    from agent_bot.config import AppConfig, ModelConfig

    mac = AppConfig(model=ModelConfig(path="m")).mac_tools
    assert (mac.enabled, mac.notes_folder, mac.timeout_s) == (False, "Agent", 60)
    assert load_config(REPO_CONFIG, env={}).mac_tools.enabled is True
