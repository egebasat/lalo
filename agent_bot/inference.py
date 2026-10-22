"""Model inference backends. Heavy libraries (torch, transformers) are imported lazily."""

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from agent_bot.config import ModelConfig


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
                result["ids"] = self._model.generate(**kwargs)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                yield from streamer
            finally:
                thread.join()
            self.last_token_count = result["ids"].shape[1] - prompt_len if "ids" in result else 0


def create_backend(cfg: ModelConfig) -> Backend:
    if cfg.backend == "unsloth":
        raise NotImplementedError("unsloth backend not implemented yet — see plan §11")
    return TransformersBackend(cfg)
