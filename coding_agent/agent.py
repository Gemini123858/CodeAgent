from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .chat_protocol import (
    ChatProtocolError,
    assistant_message,
    tool_message,
    tool_result_payload,
)
from .context import ConversationContext, FullHistoryContext
from .debug import DebugPrinter
from .llm_client import LLMClient
from .request_builder import RequestBuilder
from .tool_middleware import ToolExecution, ToolExecutionMiddleware
from .tools import ToolRegistry
from .turn_context import TurnContext


SYSTEM_PROMPT = """你是一个本地编程智能体。请使用工具完成用户交给你的任务。
读取文件后再修改，不要猜测项目内容。修改后尽可能运行相关命令验证结果。
工具失败时请分析错误并尝试修正。确认任务完成后，不再调用工具，输出修改内容和验证结果。
测试交互程序时通过 run_command 的 stdin 参数传入内容，不要使用 Shell 管道。
临时文件应使用 delete_file 清理；用户拒绝某项操作后不要重复请求或尝试绕过。
"""

ContextFactory = Callable[[], ConversationContext]
ProgressCallback = Callable[[str], None]


class TurnRecorder(Protocol):
    """Persistence boundary used by the runner without coupling it to SQLite."""

    def record_model_response(self, turn_id: int, step: int, response: Any) -> None: ...

    def record_model_error(self, turn_id: int, step: int, error: Exception) -> None: ...

    def record_assistant_message(
        self,
        conversation_id: str,
        turn_id: int,
        message: dict[str, Any],
    ) -> None: ...

    def record_tool_exchange(
        self,
        conversation_id: str,
        turn_id: int,
        step: int,
        assistant: dict[str, Any],
        executions: list[ToolExecution],
        tool_messages: list[dict[str, Any]],
    ) -> None: ...


class AgentError(RuntimeError):
    """Base class for task-level agent failures."""


class MaxStepsExceeded(AgentError):
    """Raised when the agent does not finish within the configured limit."""


class ToolCallLimitExceeded(AgentError):
    """Raised before a tool batch would exceed the per-turn call limit."""


class ModelOutputTruncated(AgentError):
    """Raised when the model stops because its output limit was reached."""


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
        max_tool_calls: int = 32,
        context_factory: ContextFactory | None = None,
        request_builder: RequestBuilder | None = None,
        middleware: ToolExecutionMiddleware | None = None,
        debug: DebugPrinter | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0。")
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls 必须大于 0。")
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.context_factory = context_factory or FullHistoryContext
        self.request_builder = request_builder or RequestBuilder(SYSTEM_PROMPT)
        self.middleware = middleware or ToolExecutionMiddleware(registry)
        self.debug = debug or DebugPrinter()
        self.progress = progress

    def run(
        self,
        task: str,
        *,
        initial_messages: list[dict[str, Any]] | None = None,
        recorder: TurnRecorder | None = None,
        conversation_id: str | None = None,
        turn_id: int | None = None,
    ) -> AgentOutcome:
        self._validate_recording_arguments(recorder, conversation_id, turn_id)
        context = self.context_factory()
        if initial_messages is None:
            context.append({"role": "user", "content": task})
        else:
            context.extend(initial_messages)

        executions: list[ToolExecution] = []
        tool_call_count = 0
        turn_context = TurnContext(turn_id=turn_id)

        for step in range(1, self.max_steps + 1):
            self._progress(f"[step {step}/{self.max_steps}] 请求模型...")
            request = self.request_builder.build(
                context,
                current_user_input=task,
            )
            self.debug.log(
                f"agent.request.{step}",
                {
                    "context_strategy": context.strategy_name,
                    "retained_messages": context.message_count,
                    "messages": request.messages,
                    "tools": request.tools,
                    "tool_choice": request.tool_choice,
                    "thinking": "disabled",
                },
            )

            try:
                response = self.llm.complete(
                    messages=request.messages,
                    tools=request.tools or None,
                    tool_choice=request.tool_choice if request.tools else None,
                )
            except Exception as exc:
                if recorder is not None and turn_id is not None:
                    recorder.record_model_error(turn_id, step, exc)
                raise

            self.debug.log(f"agent.response.{step}", response)
            if recorder is not None and turn_id is not None:
                recorder.record_model_response(turn_id, step, response)

            choice = response.choices[0]
            message = choice.message
            try:
                serialized_assistant = assistant_message(message)
            except ChatProtocolError as exc:
                raise AgentError(str(exc)) from exc

            tool_calls = list(message.tool_calls or [])
            if not tool_calls:
                context.append(serialized_assistant)
                if recorder is not None:
                    recorder.record_assistant_message(
                        conversation_id,
                        turn_id,
                        serialized_assistant,
                    )
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    raise ModelOutputTruncated(
                        "模型输出达到长度上限，任务未确认完成。"
                    )
                if finish_reason not in {None, "stop"}:
                    raise AgentError(
                        f"模型以非正常原因停止：{finish_reason}。"
                    )
                if not message.content:
                    raise AgentError("模型既没有调用工具，也没有返回最终文本。")
                return AgentOutcome(
                    final_answer=message.content,
                    steps=step,
                    tool_executions=tuple(executions),
                    context_strategy=context.strategy_name,
                    retained_messages=context.message_count,
                )

            if tool_call_count + len(tool_calls) > self.max_tool_calls:
                reason = (
                    f"本轮工具调用将超过上限 {self.max_tool_calls}，"
                    "因此整批调用均未执行。"
                )
                batch = [
                    self.middleware.skipped(step, call, reason)
                    for call in tool_calls
                ]
                self._record_tool_batch(
                    context,
                    serialized_assistant,
                    batch,
                    recorder=recorder,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    step=step,
                )
                executions.extend(batch)
                raise ToolCallLimitExceeded(reason)

            batch = [
                self.middleware.execute(
                    step,
                    call,
                    turn_context=turn_context,
                )
                for call in tool_calls
            ]
            self._record_tool_batch(
                context,
                serialized_assistant,
                batch,
                recorder=recorder,
                conversation_id=conversation_id,
                turn_id=turn_id,
                step=step,
            )
            executions.extend(batch)
            tool_call_count += len(batch)
            for execution in batch:
                status = "成功" if execution.result.ok else "失败"
                self._progress(
                    f"[step {step}] 工具 {execution.tool_name}：{status} "
                    f"({execution.duration_ms:.1f} ms)"
                )

        raise MaxStepsExceeded(
            f"达到最大步骤数 {self.max_steps}，任务仍未返回最终答案。"
        )

    def _record_tool_batch(
        self,
        context: ConversationContext,
        assistant: dict[str, Any],
        executions: list[ToolExecution],
        *,
        recorder: TurnRecorder | None,
        conversation_id: str | None,
        turn_id: int | None,
        step: int,
    ) -> None:
        tool_messages = [
            tool_message(execution.tool_call_id, execution.result)
            for execution in executions
        ]
        # One protocol unit: an assistant tool-call message followed by one
        # result for every call id.
        context.extend([assistant, *tool_messages])
        if recorder is not None:
            recorder.record_tool_exchange(
                conversation_id,
                turn_id,
                step,
                assistant,
                executions,
                tool_messages,
            )
        for execution in executions:
            self.debug.log(
                f"agent.tool.{step}.{execution.tool_call_id}",
                {
                    "name": execution.tool_name,
                    "raw_arguments": execution.raw_arguments,
                    "parsed_arguments": execution.arguments,
                    "status": execution.status,
                    "duration_ms": execution.duration_ms,
                    "result": tool_result_payload(execution.result),
                },
            )

    @staticmethod
    def _validate_recording_arguments(
        recorder: TurnRecorder | None,
        conversation_id: str | None,
        turn_id: int | None,
    ) -> None:
        # 判断是否同时提供了 recorder、conversation_id 和 turn_id，如果只提供了部分参数，则抛出 ValueError 异常，确保持久化运行的完整性。
        supplied = (
            recorder is not None,
            conversation_id is not None,
            turn_id is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "持久化运行必须同时提供 recorder、conversation_id 和 turn_id。"
            )

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)
