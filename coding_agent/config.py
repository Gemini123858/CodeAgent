from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str


def load_settings() -> Settings:
    """Load the minimal configuration needed for one DeepSeek request."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "缺少 DEEPSEEK_API_KEY，请先在当前终端中设置 DeepSeek API Key。"
        )

    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
    if not base_url:
        raise ConfigurationError("DEEPSEEK_BASE_URL 不能为空。")
    if not model:
        raise ConfigurationError("DEEPSEEK_MODEL 不能为空。")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
