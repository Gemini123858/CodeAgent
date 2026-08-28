from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_STEPS = 12
DEFAULT_MAX_TOOL_CALLS = 32


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    max_steps: int = DEFAULT_MAX_STEPS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS


def load_settings() -> Settings:
    """Load the minimal configuration needed for one DeepSeek request."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "缺少 DEEPSEEK_API_KEY，请先在当前终端中设置 DeepSeek API Key。"
        )

    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
    raw_max_steps = os.getenv("CODING_AGENT_MAX_STEPS", str(DEFAULT_MAX_STEPS))
    raw_max_tool_calls = os.getenv(
        "CODING_AGENT_MAX_TOOL_CALLS",
        str(DEFAULT_MAX_TOOL_CALLS),
    )
    if not base_url:
        raise ConfigurationError("DEEPSEEK_BASE_URL 不能为空。")
    if not model:
        raise ConfigurationError("DEEPSEEK_MODEL 不能为空。")
    try:
        max_steps = int(raw_max_steps)
    except ValueError as exc:
        raise ConfigurationError("CODING_AGENT_MAX_STEPS 必须是整数。") from exc
    if max_steps <= 0:
        raise ConfigurationError("CODING_AGENT_MAX_STEPS 必须大于 0。")
    try:
        max_tool_calls = int(raw_max_tool_calls)
    except ValueError as exc:
        raise ConfigurationError("CODING_AGENT_MAX_TOOL_CALLS 必须是整数。") from exc
    if max_tool_calls <= 0:
        raise ConfigurationError("CODING_AGENT_MAX_TOOL_CALLS 必须大于 0。")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
    )
