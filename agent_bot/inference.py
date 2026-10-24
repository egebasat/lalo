"""Model inference backends. Heavy libraries (torch, transformers, mlx_lm) are imported lazily."""

import logging
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from agent_bot.config import GenerationConfig, ModelConfig

log = logging.getLogger(__name__)


class Backend(Protocol):
    last_token_count: int

    @property
    def model_family(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def render_prompt(self, messages: list[dict], tools: list[dict], *, enable_thinking: bool) -> str: ...

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[str]: ...


class TransformersBackend:
    def __init__(self, cfg: ModelConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if cfg.num_threads:
            torch.set_num_threads(cfg.num_threads)
        self._cfg = cfg
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.path)
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.path, dtype=getattr(torch, cfg.dtype), device_map="cpu"
        )
        self._lock = threading.Lock()
        self.last_token_count = 0

    @property
    def model_family(self) -> str:
        return self._cfg.family

    @property
    def model_name(self) -> str:
        return Path(self._cfg.path).name

    def render_prompt(self, messages: list[dict], tools: list[dict], *, enable_thinking: bool) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=False,
        )

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        prompt = self.render_prompt(messages, tools, enable_thinking=enable_thinking)
        with self._lock:
            self.last_token_count = 0
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            prompt_len = inputs["input_ids"].shape[1]
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
            )
            if temperature > 0:
                kwargs["temperature"] = temperature
            result: dict = {}

            def run() -> None:
                try:
                    result["ids"] = self._model.generate(**kwargs)
                except BaseException as exc:
                    result["error"] = exc
                    streamer.end()  # unblock the consumer; the error is re-raised below

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                yield from streamer
            finally:
                thread.join()
            if "error" in result:
                raise result["error"]
            self.last_token_count = result["ids"].shape[1] - prompt_len


class MLXBackend:
    """Runs on the Apple GPU via mlx-lm.

    MLX keeps GPU streams per thread, so one single-worker executor owns the model load
    and every generation; the caller's thread only reads from a queue.

    Prompt caching: each agent iteration's prompt extends the previous one, so the KV
    cache of a finished generation is kept in mlx_lm's LRUPromptCache, keyed by its
    token ids. The next call starts from the nearest cached prefix and only processes
    the tokens after it.
    """

    def __init__(self, cfg: ModelConfig, top_p: float = 0.95, top_k: int = 20):
        self._cfg = cfg
        self._top_p = top_p
        self._top_k = top_k
        self.last_token_count = 0
        self.last_cached_tokens = 0  # prompt tokens the last call reused from the cache
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._executor.submit(self._load).result()

    def _load(self) -> None:
        import mlx_lm  # lazy, and on the MLX thread
        from mlx_lm.models.cache import LRUPromptCache

        self.model, self.tokenizer = mlx_lm.load(self._cfg.path)
        self._prompt_cache = None
        if self._cfg.prompt_cache_size > 0:
            self._prompt_cache = LRUPromptCache(
                max_size=self._cfg.prompt_cache_size,
                max_bytes=int(self._cfg.prompt_cache_gb * 1e9),
            )

    def _cached_kv(self, tokens: list[int]) -> tuple[list, list[int]]:
        """The KV cache to start from and the tokens it still has to process."""
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

        kv, rest = None, tokens
        if self._prompt_cache is not None:
            kv, rest = self._prompt_cache.fetch_nearest_cache(self.model_name, tokens)
            if kv is not None and not rest:
                # An exact hit leaves nothing to process, but sampling needs the logits
                # of the last prompt token, so step back one.
                if can_trim_prompt_cache(kv):
                    trim_prompt_cache(kv, 1)
                    rest = tokens[-1:]
                else:
                    kv, rest = None, tokens
        return kv or make_prompt_cache(self.model), rest

    @property
    def model_family(self) -> str:
        return self._cfg.family

    @property
    def model_name(self) -> str:
        return Path(self._cfg.path).name

    def render_prompt(self, messages: list[dict], tools: list[dict], *, enable_thinking: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=False,
        )

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        start = time.perf_counter()
        prompt = self.render_prompt(messages, tools, enable_thinking=enable_thinking)
        chunks: queue.Queue = queue.Queue()
        done = object()  # sentinel: the producer finished cleanly

        def produce() -> None:
            try:
                import mlx_lm
                from mlx_lm.sample_utils import make_sampler

                self.last_token_count = 0
                self.last_cached_tokens = 0
                sampler = make_sampler(temp=temperature, top_p=self._top_p, top_k=self._top_k)
                # The chat template already added the special tokens.
                key = self.tokenizer.encode(prompt, add_special_tokens=False)
                kv, rest = self._cached_kv(key)
                self.last_cached_tokens = len(key) - len(rest)
                if log.isEnabledFor(logging.DEBUG):  # skip the decoding when nobody is looking
                    n = self.last_cached_tokens
                    log.debug("prompt cache: reused %d of %d tokens\n  ...%r\n  >>> new: %r...",
                              n, len(key), self.tokenizer.decode(key[max(0, n - 40):n]),
                              self.tokenizer.decode(rest[:80]))
                tokens = 0
                for response in mlx_lm.stream_generate(
                    self.model, self.tokenizer, rest, max_tokens=max_new_tokens,
                    sampler=sampler, prompt_cache=kv,
                ):
                    if tokens == 0:
                        log.debug("time to first token: %.2fs (%d new prompt tokens)",
                                  time.perf_counter() - start, len(rest))
                    tokens = response.generation_tokens
                    key.append(response.token)  # mlx_lm has already run it into the cache
                    if response.text:
                        chunks.put(response.text)
                self.last_token_count = tokens
                if self._prompt_cache is not None:  # only after a clean finish: kv matches key
                    self._prompt_cache.insert_cache(self.model_name, key, kv)
                chunks.put(done)
            except BaseException as exc:
                chunks.put(exc)  # re-raised in the caller; never leaves it blocked

        self._executor.submit(produce)
        while True:
            item = chunks.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def create_backend(cfg: ModelConfig, gen: GenerationConfig | None = None) -> Backend:
    if cfg.backend == "unsloth":
        raise NotImplementedError("unsloth backend not implemented yet — see plan §11")
    if cfg.backend == "mlx":
        gen = gen or GenerationConfig()
        return MLXBackend(cfg, gen.top_p, gen.top_k)
    return TransformersBackend(cfg)
