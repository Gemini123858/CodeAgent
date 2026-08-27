from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **details: Any) -> ToolResult:
        return cls(ok=True, content=content, details=details)

    @classmethod
    def failure(cls, content: str, **details: Any) -> ToolResult:
        return cls(ok=False, content=content, details=details)

    def render(self) -> str:
        lines = [self.content]
        if self.details:
            summary = ", ".join(
                f"{key}={value}" for key, value in self.details.items()
            )
            lines.append(f"[{summary}]")
        return "\n".join(line for line in lines if line)
