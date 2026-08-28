from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .tools import ToolResult


class ChatProtocolError(RuntimeError):
    """Raised when a model message cannot be converted to chat history."""


def parse_function_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ChatProtocolError(
            f"模型返回的工具参数不是合法 JSON：{exc.msg}"
        ) from exc
    if not isinstance(arguments, Mapping):
        raise ChatProtocolError("模型返回的工具参数必须是 JSON 对象。")
    return dict(arguments)


def assistant_message(message: Any) -> dict[str, Any]:
    tool_calls = []
    for call in message.tool_calls or []:
        if call.type != "function":
            raise ChatProtocolError(f"不支持的工具调用类型：{call.type}")
        tool_calls.append(
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )

    serialized: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    if tool_calls:
        serialized["tool_calls"] = tool_calls

    # DeepSeek thinking mode requires this field to be preserved after tool
    # calls. It is absent while thinking is disabled, but the adapter is ready
    # for a future mode switch.
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None:
        serialized["reasoning_content"] = reasoning_content
    return serialized


def tool_result_payload(result: ToolResult) -> dict[str, Any]:
    return {
        "status": "success" if result.ok else "error",
        "content": result.content,
        "details": result.details,
    }


def tool_message(tool_call_id: str, result: ToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(
            tool_result_payload(result),
            ensure_ascii=False,
            default=str,
        ),
    }
