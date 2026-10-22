"""Turn raw model text into thinking, tool calls and visible content."""

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str


@dataclass
class ParsedOutput:
    thinking: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""
    parse_errors: list[str] = field(default_factory=list)


class Parser(Protocol):
    def parse(self, raw: str) -> ParsedOutput: ...


class Qwen3Parser:
    def parse(self, raw: str) -> ParsedOutput:
        errors: list[str] = []
        thinking, text = self._split_thinking(raw, errors)

        calls: list[ToolCall] = []
        for index, match in enumerate(_TOOL_CALL_RE.finditer(text), start=1):
            call = self._parse_call(match.group(1), index, errors)
            if call is not None:
                calls.append(call)
        content = _TOOL_CALL_RE.sub("", text)
        if "<tool_call>" in content:
            # Truncated generation: drop the fragment but report it.
            content = content.split("<tool_call>", 1)[0]
            errors.append("unclosed <tool_call> tag (output truncated?)")
        content = content.strip()

        return ParsedOutput(
            thinking=thinking or None,
            tool_calls=calls,
            content=content,
            parse_errors=errors,
        )

    @staticmethod
    def _split_thinking(raw: str, errors: list[str]) -> tuple[str, str]:
        """Return (thinking, remaining text) handling the three think variants."""
        matches = _THINK_RE.findall(raw)
        if matches:
            return "\n".join(m.strip() for m in matches).strip(), _THINK_RE.sub("", raw)
        if "</think>" in raw:
            # The opening tag lived in the prompt: output starts mid-thought.
            before, after = raw.split("</think>", 1)
            return before.strip(), after
        if "<think>" in raw:
            before, after = raw.split("<think>", 1)
            errors.append("unclosed <think> tag")
            return after.strip(), before
        return "", raw

    @staticmethod
    def _parse_call(body: str, index: int, errors: list[str]) -> ToolCall | None:
        inner = body.strip()
        fence = _FENCE_RE.match(inner)
        payload = fence.group(1).strip() if fence else inner
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            errors.append(f"invalid JSON in <tool_call> #{index}: {e}")
            return None
        if not isinstance(data, dict):
            errors.append(f"<tool_call> #{index} is not a JSON object")
            return None

        name = data.get("name")
        if not name or not isinstance(name, str):
            errors.append(f"<tool_call> #{index} is missing 'name'")
            return None

        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                errors.append(f"invalid JSON in arguments of <tool_call> #{index}: {e}")
                return None
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            errors.append(f"arguments of <tool_call> #{index} must be an object")
            return None
        return ToolCall(name=name, arguments=arguments, raw=body)


_PARSERS: dict[str, type] = {"qwen3": Qwen3Parser}


def get_parser(model_family: str) -> Parser:
    try:
        return _PARSERS[model_family]()
    except KeyError:
        raise ValueError(f"No parser for model family '{model_family}'") from None
