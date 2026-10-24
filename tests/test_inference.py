import logging
import re
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


class FakeKV:
    """One layer's KV cache, reduced to the token ids it holds."""

    def __init__(self):
        self.tokens = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(n, len(self.tokens))
        del self.tokens[len(self.tokens) - n:]
        return n

    @property
    def nbytes(self):
        return len(self.tokens)


GENERATED = "WXYZ"  # the fake model's reply, one token per character


class FakeMLX:
    """Installs a fake mlx_lm package that records the thread each call runs on.

    Tokens are characters (`ord`), the chat template concatenates message contents,
    and the real LRUPromptCache manages FakeKV caches.
    """

    def __init__(self, monkeypatch, fail_after_first=False):
        import mlx_lm.models.cache as real_cache  # before the fake package shadows it

        self.threads = []
        self.sampler_kwargs = None
        self.stream_kwargs = None
        self.prompts = []  # the token ids each stream_generate call had to process
        self.fail_after_first = fail_after_first
        self.calls = 0
        outer = self

        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                return "".join(m["content"] for m in messages)

            def encode(self, text, add_special_tokens=True):
                return [ord(c) for c in text]

            def decode(self, ids):
                return "".join(chr(i) for i in ids)

        def load(path):
            outer.threads.append(threading.current_thread().name)
            return "model", Tok()

        def stream_generate(model, tokenizer, prompt, prompt_cache=None, **kwargs):
            outer.threads.append(threading.current_thread().name)
            outer.stream_kwargs = kwargs
            outer.prompts.append(list(prompt))
            outer.calls += 1
            assert len(prompt) > 0, "mlx_lm needs at least one token to process"
            kv = prompt_cache[0].tokens if prompt_cache else []
            kv.extend(prompt)
            for i, text in enumerate(["hello", "", " there", " friend"], start=1):
                if outer.fail_after_first and outer.calls == 1 and i == 2:
                    raise RuntimeError("boom")
                token = ord(GENERATED[i - 1])
                kv.append(token)  # like mlx_lm, each returned token is already in the cache
                yield types.SimpleNamespace(text=text, token=token, generation_tokens=i)

        cache = types.ModuleType("mlx_lm.models.cache")
        cache.LRUPromptCache = real_cache.LRUPromptCache
        cache.trim_prompt_cache = real_cache.trim_prompt_cache
        cache.can_trim_prompt_cache = real_cache.can_trim_prompt_cache
        cache.make_prompt_cache = lambda model: [FakeKV()]
        models = types.ModuleType("mlx_lm.models")
        models.__path__ = []
        models.cache = cache
        monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
        monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)

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


def mlx_generate(backend, *contents):
    messages = [{"role": "user", "content": c} for c in contents]
    return "".join(backend.generate(messages, [], enable_thinking=False,
                                    max_new_tokens=8, temperature=0.0))


def test_mlx_next_iteration_processes_only_the_new_tokens(monkeypatch):
    fake = FakeMLX(monkeypatch)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"))
    mlx_generate(backend, "system", "question")
    assert backend.last_cached_tokens == 0
    # The next agent iteration: same history plus the reply and a tool result.
    mlx_generate(backend, "system", "question", "WX", "tool result")
    shared = len("systemquestionWX")
    assert backend.last_cached_tokens == shared
    assert fake.prompts[1] == [ord(c) for c in "tool result"]


def test_mlx_identical_prompt_still_processes_one_token(monkeypatch):
    fake = FakeMLX(monkeypatch)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"))
    mlx_generate(backend, "hi")
    mlx_generate(backend, "hi", GENERATED)  # exactly the tokens already in the cache
    assert fake.prompts[1] == [ord("Z")]
    assert backend.last_cached_tokens == len("hiWXY")


def test_mlx_failed_generation_is_not_cached(monkeypatch):
    fake = FakeMLX(monkeypatch, fail_after_first=True)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"))
    with pytest.raises(RuntimeError):
        mlx_generate(backend, "hi")
    mlx_generate(backend, "hi")
    assert backend.last_cached_tokens == 0
    assert fake.prompts[1] == [ord("h"), ord("i")]


def test_mlx_logs_the_cache_boundary_before_generating(monkeypatch, caplog):
    FakeMLX(monkeypatch)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"))
    mlx_generate(backend, "system", "question")
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="agent_bot.inference"):
        mlx_generate(backend, "system", "question", "WX", "tool result")
    first = caplog.messages[0]
    assert "reused 16 of 27 tokens" in first
    assert "questionWX" in first and "tool result" in first  # the end of the reused part, then the new part


def test_mlx_logs_time_to_first_token(monkeypatch, caplog):
    FakeMLX(monkeypatch)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx"))
    with caplog.at_level(logging.DEBUG, logger="agent_bot.inference"):
        mlx_generate(backend, "hi")
    assert any(re.search(r"time to first token: \d+\.\d{2}s", m) for m in caplog.messages)


def test_mlx_prompt_cache_can_be_disabled(monkeypatch):
    fake = FakeMLX(monkeypatch)
    backend = MLXBackend(ModelConfig(path="models/x", backend="mlx", prompt_cache_size=0))
    mlx_generate(backend, "hi")
    mlx_generate(backend, "hi", "WX", "more")
    assert backend.last_cached_tokens == 0
    assert fake.prompts[1] == [ord(c) for c in "hiWXmore"]


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


@pytest.mark.slow
def test_mlx_cached_generation_matches_uncached():
    def turn(backend):
        history = [*MESSAGES]
        first = "".join(backend.generate(history, [], enable_thinking=False,
                                         max_new_tokens=40, temperature=0.0))
        history += [{"role": "assistant", "content": first},
                    {"role": "user", "content": "Now say it in one word."}]
        second = "".join(backend.generate(history, [], enable_thinking=False,
                                          max_new_tokens=40, temperature=0.0))
        return second, backend.last_cached_tokens

    cached = MLXBackend(ModelConfig(path="models/Qwen3-1.7B", backend="mlx"))
    plain = MLXBackend(ModelConfig(path="models/Qwen3-1.7B", backend="mlx", prompt_cache_size=0))
    cached_text, reused = turn(cached)
    plain_text, _ = turn(plain)
    assert reused > 0
    assert cached_text == plain_text
