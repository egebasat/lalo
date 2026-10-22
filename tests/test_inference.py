import subprocess
import sys
import threading

import pytest

from agent_bot.config import ModelConfig
from agent_bot.inference import TransformersBackend, create_backend


def test_unsloth_backend_not_implemented():
    cfg = ModelConfig(path="/models/Qwen3-1.7B", backend="unsloth")
    with pytest.raises(NotImplementedError, match="unsloth"):
        create_backend(cfg)


def test_inference_import_is_light():
    code = "import sys, agent_bot.inference; sys.exit(1 if 'transformers' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


VOCAB = {1: "hello", 2: "there", 3: "friend"}
PROMPT_IDS = [7, 8, 9]


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text, return_tensors=None):
        import torch
        from transformers import BatchEncoding

        return BatchEncoding({"input_ids": torch.tensor([PROMPT_IDS])})

    def decode(self, ids, **kwargs):
        return "".join(VOCAB[int(i)] + " " for i in ids)


class FakeModel:
    device = "cpu"

    def __init__(self, fail=False):
        self.fail = fail

    def generate(self, input_ids, streamer, **kwargs):
        import torch

        if self.fail:
            raise RuntimeError("boom")
        streamer.put(input_ids)  # the prompt; skipped by the streamer
        for token_id in (1, 2, 3):
            streamer.put(torch.tensor([token_id]))
        streamer.end()
        return torch.tensor([PROMPT_IDS + [1, 2, 3]])


def make_backend(model) -> TransformersBackend:
    backend = object.__new__(TransformersBackend)  # skip __init__: no real model
    backend._tokenizer = FakeTokenizer()
    backend._model = model
    backend._lock = threading.Lock()
    backend.last_token_count = 99  # stale value from an earlier call
    return backend


def generate(backend):
    return backend.generate(
        [{"role": "user", "content": "hi"}], [],
        enable_thinking=False, max_new_tokens=8, temperature=0.0,
    )


def test_generate_streams_chunks_and_counts_tokens():
    backend = make_backend(FakeModel())
    assert "".join(generate(backend)) == "hello there friend "
    assert backend.last_token_count == 3


def test_generate_error_is_raised_not_hung():
    backend = make_backend(FakeModel(fail=True))
    outcome = {}

    def consume():
        try:
            list(generate(backend))
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "generate() hung instead of raising"
    assert isinstance(outcome["error"], RuntimeError)
    assert backend.last_token_count == 0
