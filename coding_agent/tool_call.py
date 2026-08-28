from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chat_protocol import (
    ChatProtocolError,
    assistant_message,
    parse_function_arguments,
    tool_message,
    tool_result_payload,
)
from .debug import DebugPrinter
from .llm_client import LLMClient
from .tools import ToolRegistry, ToolResult
from .tools.definitions import tool_definitions


SYSTEM_PROMPT = """你是一个本地编程助手。
本次测试必须且只能调用一个工具来获取或修改工作区信息。
工具执行结果返回后，请根据真实结果给出简洁回答，不要再次调用工具。
"""


class ToolCallError(RuntimeError):
    """Raised when the single-tool-call protocol cannot be completed."""


@dataclass(frozen=True)
class ToolCallOutcome:
    tool_name: str
    arguments: dict[str, Any]
    tool_result: ToolResult
    final_answer: str


class SingleToolCallRunner:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        debug: DebugPrinter | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.debug = debug or DebugPrinter()

    def run(self, prompt: str) -> ToolCallOutcome:
        tools = tool_definitions()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        self.debug.log(
            "request.1",
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "thinking": "disabled",
            },
        )
        first_response = self.llm.complete(
            messages=messages,
            tools=tools,
            tool_choice="required",
        )
        self.debug.log("response.1", first_response)

        first_message = first_response.choices[0].message
        tool_calls = first_message.tool_calls or []
        if len(tool_calls) != 1:
            raise ToolCallError(
                f"本阶段要求恰好一次工具调用，模型实际返回 {len(tool_calls)} 次。"
            )

        tool_call = tool_calls[0]
        if tool_call.type != "function":
            raise ToolCallError(f"不支持的工具调用类型：{tool_call.type}")

        tool_name = tool_call.function.name
        arguments = self._parse_arguments(tool_call.function.arguments)
        self.debug.log(
            "tool.call",
            {
                "id": tool_call.id,
                "name": tool_name,
                "raw_arguments": tool_call.function.arguments,
                "parsed_arguments": arguments,
            },
        )

        tool_result = self.registry.execute(tool_name, arguments)
        tool_payload = tool_result_payload(tool_result)
        self.debug.log("tool.result", tool_payload)

        messages.append(self._assistant_message(first_message))
        messages.append(tool_message(tool_call.id, tool_result))

        self.debug.log(
            "request.2",
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": "none",
                "thinking": "disabled",
            },
        )
        final_response = self.llm.complete(
            messages=messages,
            tools=tools,
            tool_choice="none",
        )
        self.debug.log("response.2", final_response)

        final_message = final_response.choices[0].message
        if final_message.tool_calls:
            raise ToolCallError("第二次响应仍包含工具调用，违反单次工具调用约束。")
        if not final_message.content:
            raise ToolCallError("第二次响应没有返回最终文本。")

        return ToolCallOutcome(
            tool_name=tool_name,
            arguments=arguments,
            tool_result=tool_result,
            final_answer=final_message.content,
        )

    @staticmethod
    def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
        try:
            return parse_function_arguments(raw_arguments)
        except ChatProtocolError as exc:
            raise ToolCallError(str(exc)) from exc

    @staticmethod
    def _assistant_message(message: Any) -> dict[str, Any]:
        try:
            return assistant_message(message)
        except ChatProtocolError as exc:
            raise ToolCallError(str(exc)) from exc
