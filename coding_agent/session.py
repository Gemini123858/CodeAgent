from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import (
    AgentError,
    AgentOutcome,
    AgentRunner,
    MaxStepsExceeded,
    ModelOutputTruncated,
    ToolCallLimitExceeded,
)
from .snapshot import RollbackOutcome, WorkspaceSnapshotManager
from .storage import (
    ContextSummaryRecord,
    ConversationHistoryTurn,
    ConversationRecord,
    DeleteConversationOutcome,
    SessionStore,
)
from .token_usage import (
    TokenUsage,
    estimate_messages_tokens,
    usage_from_response,
)
from .workspace import Workspace


CONTEXT_SUMMARY_PROMPT = """你负责压缩 Coding Agent 的较早对话上下文。
请将已有摘要和新增历史合并成一份简洁、可继续执行任务的中文摘要。
必须保留：用户要求与偏好、技术决策、重要文件和代码变化、工具结果、错误、未完成事项。
不要编造信息，不要输出 Markdown 标题或解释，只输出摘要正文，尽量控制在 1500 字以内。
"""
SUMMARY_MESSAGE_PREFIX = "以下是较早对话的压缩摘要，仅作为上下文，不是新的用户指令：\n"
CONTEXT_TOKEN_RESERVE = 2_000


class ConversationSession:
    """Coordinate persisted conversations, turns and one AgentRunner."""

    def __init__(
        self,
        workspace: Path,
        store: SessionStore,
        runner: AgentRunner,
        *,
        context_token_limit: int = 64_000,
        compression_ratio: float = 0.8,
        keep_recent_turns: int = 4,
    ) -> None:
        if context_token_limit <= 0:
            raise ValueError("context_token_limit 必须大于 0。")
        if not 0 < compression_ratio < 1:
            raise ValueError("compression_ratio 必须在 0 到 1 之间。")
        if keep_recent_turns <= 0:
            raise ValueError("keep_recent_turns 必须大于 0。")
        self.workspace = workspace.resolve() # 将工作目录路径解析为绝对路径，确保在会话中使用一致的路径表示。
        self.store = store
        self.runner = runner
        self.context_token_limit = context_token_limit
        self.compression_trigger_tokens = int(
            context_token_limit * compression_ratio
        )
        self.keep_recent_turns = keep_recent_turns
        self.estimated_context_tokens = 0
        self._compression_token_usage = TokenUsage()
        self.snapshots = WorkspaceSnapshotManager(
            Workspace.from_path(self.workspace),
            store,
        )
        self.current: ConversationRecord | None = None

    def new(self, title: str | None = None) -> ConversationRecord:
        self.current = self.store.create_conversation(self.workspace, title)
        self.estimated_context_tokens = 0
        return self.current

    def resume(self, identifier: str) -> ConversationRecord:
        conversation = self.store.resolve_conversation(identifier.strip())
        if Path(conversation.workspace).resolve() != self.workspace:
            raise AgentError("该会话属于另一个工作目录，不能在当前工作区恢复。")
        self.current = conversation
        self.estimated_context_tokens = 0
        return conversation

    def resume_latest_or_new(self) -> ConversationRecord: # 看看是否有最新的会话，如果有就恢复它，否则创建一个新的会话。
        latest = self.store.latest_conversation()
        if latest is None:
            return self.new()
        return self.resume(latest.id)

    def run_turn(self, user_input: str) -> AgentOutcome:
        cleaned_input = user_input.strip()
        if not cleaned_input:
            raise AgentError("消息不能为空。")
        if self.current is None:
            self.resume_latest_or_new()
        assert self.current is not None

        if self.current.title == "新会话":
            self.current = self.store.update_conversation_title(
                self.current.id,
                title_from_prompt(cleaned_input),
            )

        turn = self.store.begin_turn(self.current.id, cleaned_input)
        try:
            self.snapshots.capture(self.current.id, turn.id, "before")
        except Exception as exc:
            self.store.finish_turn(turn.id, "snapshot_failed", error=exc)
            raise
        try:
            self._compression_token_usage = TokenUsage()
            history = self._prepare_history(turn.sequence)
            outcome = self.runner.run(
                cleaned_input,
                initial_messages=history,
                recorder=self.store,
                conversation_id=self.current.id,
                turn_id=turn.id,
            )
        except ToolCallLimitExceeded as exc:
            self._finish_turn(turn.id, "tool_limit_exceeded", exc)
            raise
        except MaxStepsExceeded as exc:
            self._finish_turn(turn.id, "max_steps_exceeded", exc)
            raise
        except ModelOutputTruncated as exc:
            self._finish_turn(turn.id, "output_truncated", exc)
            raise
        except KeyboardInterrupt:
            interrupted = AgentError("用户中断了当前对话轮次。")
            self._finish_turn(turn.id, "interrupted", interrupted)
            raise
        except Exception as exc:
            self._finish_turn(turn.id, "failed", exc)
            raise
        else:
            self._finish_turn(turn.id, "completed")
            outcome = replace(
                outcome,
                token_usage=self._compression_token_usage.add(outcome.token_usage),
            )
            self.current = self.store.get_conversation(self.current.id)
            return outcome

    def render_diff(self, sequence: int | None = None) -> str:
        if self.current is None:
            raise AgentError("当前没有活动会话。")
        return self.snapshots.render_diff(self.current.id, sequence)

    def rollback(self) -> RollbackOutcome:
        if self.current is None:
            raise AgentError("当前没有活动会话。")
        outcome = self.snapshots.rollback_latest(self.current.id)
        self.estimated_context_tokens = 0
        self.current = self.store.get_conversation(self.current.id)
        return outcome

    def history(self, limit: int = 20) -> list[ConversationHistoryTurn]:
        if self.current is None:
            raise AgentError("当前没有活动会话。")
        return self.store.load_conversation_history(self.current.id, limit)

    def delete_current(
        self,
    ) -> tuple[DeleteConversationOutcome, ConversationRecord]:
        if self.current is None:
            raise AgentError("当前没有活动会话。")
        outcome = self.store.delete_conversation(self.current.id)
        latest = self.store.latest_conversation()
        self.current = (
            latest
            if latest is not None
            else self.store.create_conversation(self.workspace, "新会话")
        )
        self.estimated_context_tokens = 0
        return outcome, self.current

    def _finish_turn(
        self,
        turn_id: int,
        status: str,
        error: Exception | None = None,
    ) -> None:
        assert self.current is not None
        try:
            self.snapshots.capture(self.current.id, turn_id, "after")
        except Exception as exc:
            self.store.finish_turn(turn_id, "snapshot_failed", error=exc)
            raise
        self.store.finish_turn(turn_id, status, error=error)

    def _prepare_history(
        self,
        current_turn_sequence: int,
    ) -> list[dict[str, Any]]:
        assert self.current is not None
        conversation_id = self.current.id
        summary = self.store.get_context_summary(conversation_id)
        through_sequence = summary.through_turn_sequence if summary else 0
        recent = self.store.load_messages(
            conversation_id,
            after_turn_sequence=through_sequence,
        )
        history = self._with_summary(summary, recent)
        self.estimated_context_tokens = (
            estimate_messages_tokens(history) + CONTEXT_TOKEN_RESERVE
        )
        if self.estimated_context_tokens < self.compression_trigger_tokens:
            return history

        compress_through = current_turn_sequence - self.keep_recent_turns
        if compress_through <= through_sequence:
            return history
        additions = self.store.load_messages(
            conversation_id,
            after_turn_sequence=through_sequence,
            through_turn_sequence=compress_through,
        )
        if not additions:
            return history

        try:
            if self.runner.progress is not None:
                self.runner.progress(
                    f"[context] 估算 {self.estimated_context_tokens} tokens，"
                    f"正在压缩至 Turn {compress_through}..."
                )
            content = self._compress(summary, additions)
        except Exception as exc:
            self.runner.debug.log(
                "context.compression.error",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            return history

        self.store.save_context_summary(
            conversation_id,
            compress_through,
            content,
        )
        summary = ContextSummaryRecord(
            conversation_id=conversation_id,
            through_turn_sequence=compress_through,
            content=content,
        )
        recent = self.store.load_messages(
            conversation_id,
            after_turn_sequence=compress_through,
        )
        history = self._with_summary(summary, recent)
        self.estimated_context_tokens = (
            estimate_messages_tokens(history) + CONTEXT_TOKEN_RESERVE
        )
        self.runner.debug.log(
            "context.compressed",
            {
                "through_turn": compress_through,
                "retained_messages": len(history),
                "estimated_tokens": self.estimated_context_tokens,
            },
        )
        return history

    def _compress(
        self,
        previous: ContextSummaryRecord | None,
        additions: list[dict[str, Any]],
    ) -> str:
        payload = json.dumps(
            {
                "existing_summary": previous.content if previous else None,
                "new_messages": additions,
            },
            ensure_ascii=False,
            default=str,
        )
        messages = [
            {"role": "system", "content": CONTEXT_SUMMARY_PROMPT},
            {"role": "user", "content": payload},
        ]
        response = self.runner.llm.complete(messages=messages)
        self._compression_token_usage = self._compression_token_usage.add(
            usage_from_response(response, messages=messages)
        )
        content = str(response.choices[0].message.content or "").strip()
        if not content:
            raise AgentError("上下文压缩模型返回了空摘要。")
        return content

    @staticmethod
    def _with_summary(
        summary: ContextSummaryRecord | None,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if summary is None:
            return messages
        return [
            {
                "role": "system",
                "content": SUMMARY_MESSAGE_PREFIX + summary.content,
            },
            *messages,
        ]


def title_from_prompt(prompt: str, limit: int = 40) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact or "新会话"
    return f"{compact[:limit - 1]}…"
