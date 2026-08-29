from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    summary: str
    reason: str
    risk_level: str
    fingerprint: str
    audit_advice: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ApprovalProvider(Protocol):
    def request(self, request: ApprovalRequest) -> bool:
        """Return True only when the user explicitly approves the action."""


class DenyApprovalProvider:
    """Safe default for non-interactive callers."""

    def request(self, request: ApprovalRequest) -> bool:
        del request
        return False


class CLIApprovalProvider:
    def __init__(
        self,
        *,
        input_func: Callable[[str], str] | None = None,
        output_func: Callable[[str], None] | None = None,
    ) -> None:
        self.input_func = input_func
        self.output_func = output_func or (
            lambda message: print(message, file=sys.stderr)
        )

    def request(self, request: ApprovalRequest) -> bool:
        lines = [
            "\n检测到需要确认的工具操作：",
            f"  操作：{request.summary}",
            f"  风险：{request.risk_level} - {request.reason}",
        ]
        if request.audit_advice:
            lines.append(f"  审计建议：{request.audit_advice}")
        self.output_func("\n".join(lines))
        reader = self.input_func or input
        try:
            answer = reader("是否允许执行？[y/N] ").strip().lower()
        except EOFError:
            self.output_func("未读取到用户确认，默认拒绝。")
            return False
        return answer in {"y", "yes"}

