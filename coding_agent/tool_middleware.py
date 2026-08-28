from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .chat_protocol import ChatProtocolError, parse_function_arguments
from .tools import ToolRegistry, ToolResult


@dataclass(frozen=True)
class ToolExecution:
    step: int
    tool_call_id: str
    tool_name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    result: ToolResult
    status: str
    duration_ms: float


class ToolExecutionMiddleware:
    """Normalize, execute and time every local tool call."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, step: int, call: Any) -> ToolExecution:
        started = perf_counter()
        tool_name = getattr(getattr(call, "function", None), "name", "<unknown>")
        raw_value = getattr(
            getattr(call, "function", None),
            "arguments",
            "{}",
        )
        raw_arguments = raw_value if isinstance(raw_value, str) else str(raw_value)
        arguments: dict[str, Any] | None = None

        try:
            if getattr(call, "type", None) != "function":
                result = ToolResult.failure(
                    f"不支持的工具调用类型：{getattr(call, 'type', None)}"
                )
            else:
                arguments = parse_function_arguments(raw_arguments)
                result = self.registry.execute(tool_name, arguments)
        except ChatProtocolError as exc:
            result = ToolResult.failure(str(exc))
        except Exception as exc:
            result = ToolResult.failure(
                f"工具中间层出现未预期错误：{type(exc).__name__}"
            )

        return ToolExecution(
            step=step,
            tool_call_id=str(getattr(call, "id", "<missing>")),
            tool_name=tool_name,
            raw_arguments=raw_arguments,
            arguments=arguments,
            result=result,
            status="success" if result.ok else "failed",
            duration_ms=(perf_counter() - started) * 1000,
        )

    def skipped(self, step: int, call: Any, reason: str) -> ToolExecution:
        return ToolExecution(
            step=step,
            tool_call_id=str(getattr(call, "id", "<missing>")),
            tool_name=str(
                getattr(getattr(call, "function", None), "name", "<unknown>")
            ),
            raw_arguments=str(
                getattr(getattr(call, "function", None), "arguments", "{}")
            ),
            arguments=None,
            result=ToolResult.failure(reason, skipped=True),
            status="skipped",
            duration_ms=0.0,
        )
