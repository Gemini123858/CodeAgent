from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    source: str = "api"

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            source="api" if self.source == other.source == "api" else "estimated",
        )


def estimate_text_tokens(text: str) -> int:
    """Small dependency-free estimate suitable for context thresholds."""
    ascii_chars = sum(1 for character in text if ord(character) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4 + non_ascii_chars))


def estimate_messages_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    payload = json.dumps(messages, ensure_ascii=False, default=str)
    total = estimate_text_tokens(payload) + len(messages) * 4
    if tools:
        total += estimate_text_tokens(
            json.dumps(tools, ensure_ascii=False, default=str)
        )
    return total


def usage_from_response(
    response: Any,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> TokenUsage:
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if isinstance(prompt, int) and isinstance(completion, int):
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total if isinstance(total, int) else prompt + completion,
            source="api",
        )

    message = response.choices[0].message
    completion_text = str(getattr(message, "content", "") or "")
    completion_text += str(getattr(message, "tool_calls", "") or "")
    estimated_prompt = estimate_messages_tokens(messages, tools)
    estimated_completion = estimate_text_tokens(completion_text)
    return TokenUsage(
        prompt_tokens=estimated_prompt,
        completion_tokens=estimated_completion,
        total_tokens=estimated_prompt + estimated_completion,
        source="estimated",
    )
