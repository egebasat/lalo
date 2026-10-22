import subprocess
import sys
import threading

import pytest

import types

from agent_bot.config import GenerationConfig, ModelConfig
from agent_bot.inference import MLXBackend, TransformersBackend, create_backend


def test_unsloth_backend_not_implemented():
    cfg = ModelConfig(path="/models/Qwen3-1.7B", backend="unsloth")
    with pytest.raises(NotImplementedError, match="unsloth"):
        create_backend(cfg)


def test_inference_import_is_light():
    code = (
        "import sys, agent_bot.inference; "
        "sys.exit(1 if {'transformers', 'mlx', 'mlx_lm'} & set(sys.modules) else 0)"
    )
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


def test_create_backend_selects_mlx(monkeypatch):
    seen = {}

    class Stub:
        def __init__(self, cfg, top_p=0.95, top_k=20):
            seen.update(cfg=cfg, top_p=top_p, top_k=top_k)

    monkeypatch.setattr("agent_bot.inference.MLXBackend", Stub)
    cfg = ModelConfig(path="x", backend="mlx")
    assert isinstance(create_backend(cfg, GenerationConfig(top_k=7)), Stub)
    assert seen["top_k"] == 7 and seen["cfg"] is cfg


class FakeMLX:
    """Installs a fake mlx_lm package that records the thread each call runs on."""

    def __init__(self, monkeypatch, fail_after_first=False):
        self.threads = []
        self.sampler_kwargs = None
        self.stream_kwargs = None
        self.fail_after_first = fail_after_first
        self.calls = 0
        outer = self

        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

        def load(path):
            outer.threads.append(threading.current_thread().name)
            return "model", Tok()

        def stream_generate(model, tokenizer, prompt, **kwargs):
            outer.threads.append(threading.current_thread().name)
            outer.stream_kwargs = kwargs
            outer.calls += 1
            for i, text in enumerate(["hello", "", " there", " friend"], start=1):
                if outer.fail_after_first and outer.calls == 1 and i == 2:
                    raise RuntimeError("boom")
                yield types.SimpleNamespace(text=text, generation_tokens=i)

        def make_sampler(**kw):
            outer.sampler_kwargs = kw
            return kw

        pkg = types.ModuleType("mlx_lm")
        pkg.load = load
        pkg.stream_generate = stream_generate
        pkg.__path__ = []
        sample_utils = types.ModuleType("mlx_lm.sample_utils")
        sample_utils.make_sampler = make_sampler
        pkg.sample_utils = sample_utils
        monkeypatch.setitem(sys.modules, "mlx_lm", pkg)
        monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)


def test_mlx_load_and_generate_share_one_thread(monkeypatch):
    fake = FakeMLX(monkeypatch)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"))
    assert list(generate(backend)) == ["hello", " there", " friend"]
    assert backend.last_token_count == 4
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("chunks", list(generate(backend))))
    t.start()
    t.join(timeout=5)
    assert out["chunks"] == ["hello", " there", " friend"]
    assert len(fake.threads) == 3 and len(set(fake.threads)) == 1
    assert fake.threads[0].startswith("mlx") and fake.threads[0] != "MainThread"


def test_mlx_generate_error_is_raised_not_hung(monkeypatch):
    FakeMLX(monkeypatch, fail_after_first=True)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"))
    outcome = {"chunks": []}

    def consume():
        try:
            for chunk in generate(backend):
                outcome["chunks"].append(chunk)
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "generate() hung instead of raising"
    assert outcome["chunks"] == ["hello"]
    assert isinstance(outcome["error"], RuntimeError)
    assert list(generate(backend)) == ["hello", " there", " friend"]  # still usable


def test_mlx_passes_sampling_settings(monkeypatch):
    fake = FakeMLX(monkeypatch)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"), top_p=0.8, top_k=7)
    list(backend.generate([{"role": "user", "content": "hi"}], [],
                          enable_thinking=False, max_new_tokens=8, temperature=0.3))
    assert fake.sampler_kwargs == {"temp": 0.3, "top_p": 0.8, "top_k": 7}
    assert fake.stream_kwargs["max_tokens"] == 8


MESSAGES = [{"role": "user", "content": "What time is it in Tokyo?"}]


@pytest.mark.slow
def test_mlx_streams_tool_call():
    from transformers.utils import get_json_schema

    from agent_bot.tools import get_current_time

    backend = MLXBackend(ModelConfig(path="models/Qwen3-1.7B", backend="mlx"))
    chunks = list(backend.generate(
        MESSAGES, [get_json_schema(get_current_time)],
        enable_thinking=True, max_new_tokens=400, temperature=0.6,
    ))
    text = "".join(chunks)
    assert len(chunks) > 1
    assert "<tool_call>" in text and "Asia/Tokyo" in text
    assert backend.last_token_count > 0


@pytest.mark.slow
def test_mlx_generate_on_worker_thread():
    backend = MLXBackend(ModelConfig(path="models/Qwen3-1.7B", backend="mlx"))
    out = {}

    def run():
        try:
            out["text"] = "".join(backend.generate(
                MESSAGES, [], enable_thinking=False, max_new_tokens=20, temperature=0.0))
        except BaseException as exc:
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=120)
    assert "error" not in out, out.get("error")
    assert out["text"].strip()
