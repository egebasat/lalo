import subprocess
import sys

import pytest

from agent_bot.config import ModelConfig
from agent_bot.inference import create_backend


def test_unsloth_backend_not_implemented():
    cfg = ModelConfig(path="/models/Qwen3-1.7B", backend="unsloth")
    with pytest.raises(NotImplementedError, match="unsloth"):
        create_backend(cfg)


def test_inference_import_is_light():
    code = "import sys, agent_bot.inference; sys.exit(1 if 'transformers' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
