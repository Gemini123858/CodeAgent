from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


DEBUG_ENV_VAR = "CODING_AGENT_DEBUG"
TRUE_VALUES = frozenset({"1", "true", "yes", "on", "debug"})
SENSITIVE_KEY_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


def debug_enabled(command_line_flag: bool = False) -> bool:
    if command_line_flag:
        return True
    value = os.getenv(DEBUG_ENV_VAR, "").strip().lower()
    return value in TRUE_VALUES


@dataclass(frozen=True)
class DebugPrinter:
    enabled: bool = False
    max_chars: int = 12_000

    def log(self, label: str, payload: Any | None = None) -> None:
        if not self.enabled:
            return

        print(f"\n[DEBUG] {label}", file=sys.stderr)
        if payload is None:
            return

        normalized = self._normalize(payload)
        text = json.dumps(normalized, ensure_ascii=False, indent=2, default=str)
        text = re.sub(
            r"sk-[A-Za-z0-9_-]{12,}",
            "sk-<redacted>",
            text,
        )
        if len(text) > self.max_chars:
            marker = "\n... DEBUG 中间内容已截断 ...\n"
            side = max(1, (self.max_chars - len(marker)) // 2)
            text = text[:side] + marker + text[-side:]
        print(text, file=sys.stderr)

    def _normalize(self, value: Any, key: str = "") -> Any: # Normalize the value for debug output, redacting sensitive information and handling various data types.
        upper_key = key.upper()
        if any(marker in upper_key for marker in SENSITIVE_KEY_MARKERS):
            return "<redacted>"
        if key == "reasoning_content":
            length = len(value) if isinstance(value, str) else 0
            return f"<omitted: {length} chars>"
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        if isinstance(value, Mapping):
            return {
                str(item_key): self._normalize(item_value, str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)
