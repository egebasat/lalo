"""All settings in one place: config.yaml, validated by pydantic, with env overrides.

Any key can be overridden with an env var named AGENT__<SECTION>__<KEY>
(e.g. AGENT__MODEL__PATH=/models/Qwen3-4B). Pydantic coerces the string to
the field's type.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

ENV_PREFIX = "AGENT__"


class ModelConfig(BaseModel):
    path: str
    family: str = "qwen3"
    backend: Literal["mlx", "transformers", "unsloth"] = "mlx"
    dtype: Literal["bfloat16", "float16", "float32"] = "float32"
    num_threads: int | None = None
    prompt_cache_size: int = 4  # mlx only: KV caches kept for prefix reuse; 0 disables
    prompt_cache_gb: float = 2.0  # mlx only: memory cap for those caches


class GenerationConfig(BaseModel):
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True


class AgentConfig(BaseModel):
    max_iterations: int = 6
    system_prompt_file: str = "prompts/system.md"
    local_python_tools: bool = False


class ToolsConfig(BaseModel):
    timeout_s: float = 20
    max_result_chars: int = 4000
    disabled: list[str] = []


class McpServerConfig(BaseModel):
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    url: str | None = None
    headers: dict[str, str] = {}
    enabled: bool = True

    @model_validator(mode="after")
    def _exactly_one_transport(self) -> "McpServerConfig":
        if (self.command is None) == (self.url is None):
            raise ValueError("an MCP server needs exactly one of 'command' (stdio) or 'url' (HTTP)")
        return self


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class SandboxConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001
    python_tools: bool = True
    mcp_servers: dict[str, McpServerConfig] = {}


class MacToolsConfig(BaseModel):
    enabled: bool = False  # host only, macOS only
    notes_folder: str = "Agent"  # name used when the folder is first created
    timeout_s: float = 60  # per-call limit for Mac tools (cold app launch + permission prompt)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_dir: str | None = None


class AppConfig(BaseModel):
    model: ModelConfig
    generation: GenerationConfig = GenerationConfig()
    agent: AgentConfig = AgentConfig()
    tools: ToolsConfig = ToolsConfig()
    mcp_servers: dict[str, McpServerConfig] = {}
    sandbox: SandboxConfig = SandboxConfig()
    mac_tools: MacToolsConfig = MacToolsConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()


def _apply_env_overrides(raw: dict, env: Mapping[str, str]) -> None:
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        *sections, key = [part.lower() for part in name[len(ENV_PREFIX):].split("__")]
        node = raw
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = value


def load_config(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> AppConfig:
    env = os.environ if env is None else env
    path = Path(path or env.get("AGENT_CONFIG", "config.yaml"))
    raw = yaml.safe_load(path.read_text()) or {}
    _apply_env_overrides(raw, env)
    return AppConfig.model_validate(raw)
