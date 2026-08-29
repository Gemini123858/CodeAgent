from __future__ import annotations

from pathlib import Path

from .agent import (
    AgentError,
    AgentOutcome,
    AgentRunner,
    MaxStepsExceeded,
    ModelOutputTruncated,
    ToolCallLimitExceeded,
)
from .storage import ConversationRecord, SessionStore


class ConversationSession:
    """Coordinate persisted conversations, turns and one AgentRunner."""

    def __init__(
        self,
        workspace: Path,
        store: SessionStore,
        runner: AgentRunner,
    ) -> None:
        self.workspace = workspace.resolve() # 将工作目录路径解析为绝对路径，确保在会话中使用一致的路径表示。
        self.store = store
        self.runner = runner
        self.current: ConversationRecord | None = None

    def new(self, title: str | None = None) -> ConversationRecord:
        self.current = self.store.create_conversation(self.workspace, title)
        return self.current

    def resume(self, identifier: str) -> ConversationRecord:
        conversation = self.store.resolve_conversation(identifier.strip())
        if Path(conversation.workspace).resolve() != self.workspace:
            raise AgentError("该会话属于另一个工作目录，不能在当前工作区恢复。")
        self.current = conversation
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
        history = self.store.load_messages(self.current.id)
        try:
            outcome = self.runner.run(
                cleaned_input,
                initial_messages=history,
                recorder=self.store,
                conversation_id=self.current.id,
                turn_id=turn.id,
            )
        except ToolCallLimitExceeded as exc:
            self.store.finish_turn(turn.id, "tool_limit_exceeded", error=exc)
            raise
        except MaxStepsExceeded as exc:
            self.store.finish_turn(turn.id, "max_steps_exceeded", error=exc)
            raise
        except ModelOutputTruncated as exc:
            self.store.finish_turn(turn.id, "output_truncated", error=exc)
            raise
        except KeyboardInterrupt:
            interrupted = AgentError("用户中断了当前对话轮次。")
            self.store.finish_turn(turn.id, "interrupted", error=interrupted)
            raise
        except Exception as exc:
            self.store.finish_turn(turn.id, "failed", error=exc)
            raise
        else:
            self.store.finish_turn(turn.id, "completed")
            self.current = self.store.get_conversation(self.current.id)
            return outcome


def title_from_prompt(prompt: str, limit: int = 40) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact or "新会话"
    return f"{compact[:limit - 1]}…"
