from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .chat_protocol import (
    ChatProtocolError,
    assistant_message,
    parse_function_arguments,
    tool_message,
    tool_result_payload,
)
from .context import ConversationContext, FullHistoryContext
from .debug import DebugPrinter
from .llm_client import LLMClient
from .tools import ToolRegistry, ToolResult
from .tools.definitions import tool_definitions


SYSTEM_PROMPT = """你是一个本地编程智能体。请使用工具完成用户交给你的任务。
读取文件后再修改，不要猜测项目内容。修改后尽可能运行相关命令验证结果。
工具失败时请分析错误并尝试修正。确认任务完成后，不再调用工具，输出修改内容和验证结果。
"""

ContextFactory = Callable[[], ConversationContext]
ProgressCallback = Callable[[str], None]


class AgentError(RuntimeError):
    """Base class for task-level agent failures."""


class MaxStepsExceeded(AgentError):
    """Raised when the agent does not finish within the configured limit."""


@dataclass(frozen=True)
class ToolExecution:
    step: int
    tool_call_id: str
    tool_name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    result: ToolResult


@dataclass(frozen=True)
class AgentOutcome:
    final_answer: str
    steps: int
    tool_executions: tuple[ToolExecution, ...]
    context_strategy: str
    retained_messages: int


class AgentRunner:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 12,
        context_factory: ContextFactory | None = None,
        debug: DebugPrinter | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0。")
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.context_factory = context_factory or FullHistoryContext
        self.debug = debug or DebugPrinter()
        self.progress = progress

    def run(self, task: str) -> AgentOutcome:
        context = self.context_factory()
        context.extend(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]
        )
        tools = tool_definitions()
        executions: list[ToolExecution] = []

        for step in range(1, self.max_steps + 1):
            self._progress(f"[step {step}/{self.max_steps}] 请求模型...")
            request_messages = context.messages_for_request()
            self.debug.log(
                f"agent.request.{step}",
                {
                    "context_strategy": context.strategy_name,
                    "retained_messages": context.message_count,
                    "messages": request_messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "thinking": "disabled",
                },
            )

            response = self.llm.complete(
                messages=request_messages,
                tools=tools,
                tool_choice="auto",
            )
            self.debug.log(f"agent.response.{step}", response)

            choice = response.choices[0]
            message = choice.message
            try:
                context.append(assistant_message(message))
            except ChatProtocolError as exc:
                raise AgentError(str(exc)) from exc

            tool_calls = message.tool_calls or []
            if not tool_calls:
                if getattr(choice, "finish_reason", None) == "length":
                    raise AgentError("模型输出达到长度上限，任务未确认完成。")
                if not message.content:
                    raise AgentError("模型既没有调用工具，也没有返回最终文本。")
                return AgentOutcome(
                    final_answer=message.content,
                    steps=step,
                    tool_executions=tuple(executions),
                    context_strategy=context.strategy_name,
                    retained_messages=context.message_count,
                )

            for call in tool_calls:
                execution = self._execute_tool(step, call)
                executions.append(execution)
                context.append(tool_message(call.id, execution.result))
                status = "成功" if execution.result.ok else "失败"
                self._progress(
                    f"[step {step}] 工具 {execution.tool_name}：{status}"
                )

        raise MaxStepsExceeded(
            f"达到最大步骤数 {self.max_steps}，任务仍未返回最终答案。"
        )

    def _execute_tool(self, step: int, call: Any) -> ToolExecution:
        if call.type != "function":
            raise AgentError(f"不支持的工具调用类型：{call.type}")

        raw_arguments = call.function.arguments
        try:
            arguments = parse_function_arguments(raw_arguments)
            result = self.registry.execute(call.function.name, arguments)
        except ChatProtocolError as exc:
            arguments = None
            result = ToolResult.failure(str(exc))

        execution = ToolExecution(
            step=step,
            tool_call_id=call.id,
            tool_name=call.function.name,
            raw_arguments=raw_arguments,
            arguments=arguments,
            result=result,
        )
        self.debug.log(
            f"agent.tool.{step}.{call.id}",
            {
                "name": execution.tool_name,
                "raw_arguments": execution.raw_arguments,
                "parsed_arguments": execution.arguments,
                "result": tool_result_payload(execution.result),
            },
        )
        return execution

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)
