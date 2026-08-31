from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .tool_middleware import ToolExecution


STATE_DIRECTORY_NAME = ".coding-agent"
DATABASE_FILE_NAME = "state.db"


class StorageError(RuntimeError):
    """Raised when persisted conversation state cannot be read or written."""


@dataclass(frozen=True)
class ConversationRecord:
    id: str
    title: str
    workspace: str
    status: str
    created_at: str
    updated_at: str
    last_turn_status: str | None = None


@dataclass(frozen=True)
class TurnRecord:
    id: int
    conversation_id: str
    sequence: int
    user_input: str
    status: str


@dataclass(frozen=True)
class SnapshotEntryRecord:
    path: str
    kind: str
    blob_hash: str | None
    size: int
    mode: int


@dataclass(frozen=True)
class RollbackTarget:
    turn: TurnRecord
    before_snapshot_id: int
    after_snapshot_id: int


@dataclass(frozen=True)
class ConversationHistoryTurn:
    sequence: int
    status: str
    started_at: str
    finished_at: str | None
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DeleteConversationOutcome:
    conversation_id: str
    title: str
    turns_deleted: int
    messages_deleted: int
    tool_calls_deleted: int
    snapshots_deleted: int
    blobs_deleted: int
    blob_cleanup_failures: int


@dataclass(frozen=True)
class ContextSummaryRecord:
    conversation_id: str
    through_turn_sequence: int
    content: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class SessionStore:
    """SQLite-backed append-only conversation and tool execution store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True) # 如果数据库目录不存在，则创建它
        try:
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema() # 创建数据库模式（建表等）
            self._recover_interrupted_turns() # 恢复上次未完成的对话轮次，将其状态标记为“interrupted”
        except sqlite3.Error as exc:
            raise StorageError(f"无法初始化会话数据库：{exc}") from exc

    @classmethod
    def for_workspace(cls, workspace: Path) -> SessionStore:
        return cls(workspace / STATE_DIRECTORY_NAME / DATABASE_FILE_NAME)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def create_conversation(
        self,
        workspace: Path,
        title: str | None = None,
    ) -> ConversationRecord:
        conversation_id = uuid.uuid4().hex
        now = utc_now()
        cleaned_title = (title or "新会话").strip() or "新会话"
        try:
            with self.connection:
                self.connection.execute( # 在数据库中插入一个新的会话记录，包含会话 ID、标题、工作目录路径、状态和时间戳。
                    """
                    INSERT INTO conversations(
                        id, title, workspace, status, created_at, updated_at
                    ) VALUES (?, ?, ?, 'active', ?, ?)
                    """,
                    (conversation_id, cleaned_title, str(workspace), now, now),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"无法创建会话：{exc}") from exc
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> ConversationRecord:
        row = self.connection.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise StorageError(f"会话不存在：{conversation_id}")
        return self._conversation_from_row(row)

    def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
    ) -> ConversationRecord:
        cleaned_title = title.strip()
        if not cleaned_title:
            raise StorageError("会话标题不能为空。")
        try:
            with self.connection:
                cursor = self.connection.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (cleaned_title, utc_now(), conversation_id),
                )
                if cursor.rowcount != 1:
                    raise StorageError(f"会话不存在：{conversation_id}")
        except StorageError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"无法更新会话标题：{exc}") from exc
        return self.get_conversation(conversation_id)

    def resolve_conversation(self, identifier: str) -> ConversationRecord:
        exact = self.connection.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (identifier,),
        ).fetchone()
        if exact is not None:
            return self._conversation_from_row(exact)

        rows = self.connection.execute(
            "SELECT * FROM conversations WHERE id LIKE ? ORDER BY updated_at DESC",
            (f"{identifier}%",),
        ).fetchall()
        if not rows:
            raise StorageError(f"会话不存在：{identifier}")
        if len(rows) > 1:
            raise StorageError(f"会话 ID 前缀不唯一：{identifier}")
        return self._conversation_from_row(rows[0])

    def latest_conversation(self) -> ConversationRecord | None:
        row = self.connection.execute(
            """
            SELECT * FROM conversations
            WHERE status = 'active'
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
        return self._conversation_from_row(row) if row is not None else None

    def list_conversations(self, limit: int = 20) -> list[ConversationRecord]:
        rows = self.connection.execute(
            """
            SELECT c.*,
                (
                    SELECT t.status FROM turns t
                    WHERE t.conversation_id = c.id
                    ORDER BY t.sequence DESC LIMIT 1
                ) AS last_turn_status
            FROM conversations c
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._conversation_from_row(row) for row in rows]

    def load_conversation_history(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[ConversationHistoryTurn]:
        if limit <= 0 or limit > 100:
            raise StorageError("历史记录数量必须在 1 到 100 之间。")
        self.get_conversation(conversation_id)
        turn_rows = self.connection.execute(
            """
            SELECT * FROM (
                SELECT id, sequence, status, started_at, finished_at
                FROM turns
                WHERE conversation_id = ?
                ORDER BY sequence DESC
                LIMIT ?
            ) recent
            ORDER BY sequence
            """,
            (conversation_id, limit),
        ).fetchall()
        history: list[ConversationHistoryTurn] = []
        try:
            for turn in turn_rows:
                message_rows = self.connection.execute(
                    """
                    SELECT payload_json FROM messages
                    WHERE turn_id = ?
                    ORDER BY sequence
                    """,
                    (turn["id"],),
                ).fetchall()
                history.append(
                    ConversationHistoryTurn(
                        sequence=turn["sequence"],
                        status=turn["status"],
                        started_at=turn["started_at"],
                        finished_at=turn["finished_at"],
                        messages=tuple(
                            json.loads(row["payload_json"])
                            for row in message_rows
                        ),
                    )
                )
        except json.JSONDecodeError as exc:
            raise StorageError("数据库中的历史消息 JSON 已损坏。") from exc
        return history

    def get_context_summary(
        self,
        conversation_id: str,
    ) -> ContextSummaryRecord | None:
        row = self.connection.execute(
            "SELECT * FROM context_summaries WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return ContextSummaryRecord(
            conversation_id=row["conversation_id"],
            through_turn_sequence=row["through_turn_sequence"],
            content=row["content"],
        )

    def save_context_summary(
        self,
        conversation_id: str,
        through_turn_sequence: int,
        content: str,
    ) -> None:
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO context_summaries(
                        conversation_id, through_turn_sequence, content, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        through_turn_sequence = excluded.through_turn_sequence,
                        content = excluded.content,
                        updated_at = excluded.updated_at
                    """,
                    (
                        conversation_id,
                        through_turn_sequence,
                        content,
                        utc_now(),
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"无法保存上下文摘要：{exc}") from exc

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> DeleteConversationOutcome:
        conversation = self.get_conversation(conversation_id)
        blob_rows = self.connection.execute(
            """
            SELECT DISTINCT se.blob_hash
            FROM snapshot_entries se
            JOIN snapshots s ON s.id = se.snapshot_id
            WHERE s.conversation_id = ? AND se.blob_hash IS NOT NULL
            """,
            (conversation_id,),
        ).fetchall()
        candidate_blobs = {row["blob_hash"] for row in blob_rows}
        counts = {
            "turns": self.connection.execute(
                "SELECT COUNT(*) FROM turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0],
            "messages": self.connection.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0],
            "tool_calls": self.connection.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0],
            "snapshots": self.connection.execute(
                "SELECT COUNT(*) FROM snapshots WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0],
        }
        try:
            with self.connection:
                self.connection.execute(
                    "DELETE FROM context_summaries WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self.connection.execute(
                    """
                    DELETE FROM approval_requests
                    WHERE tool_call_id IN (
                        SELECT id FROM tool_calls WHERE conversation_id = ?
                    )
                    """,
                    (conversation_id,),
                )
                self.connection.execute(
                    """
                    DELETE FROM tool_results
                    WHERE tool_call_id IN (
                        SELECT id FROM tool_calls WHERE conversation_id = ?
                    )
                    """,
                    (conversation_id,),
                )
                self.connection.execute(
                    "DELETE FROM file_changes WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self.connection.execute(
                    "DELETE FROM rollbacks WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self.connection.execute(
                    """
                    DELETE FROM turn_snapshots
                    WHERE turn_id IN (
                        SELECT id FROM turns WHERE conversation_id = ?
                    )
                    """,
                    (conversation_id,),
                )
                self.connection.execute(
                    """
                    DELETE FROM snapshot_entries
                    WHERE snapshot_id IN (
                        SELECT id FROM snapshots WHERE conversation_id = ?
                    )
                    """,
                    (conversation_id,),
                )
                self.connection.execute(
                    "DELETE FROM snapshots WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self.connection.execute(
                    "DELETE FROM tool_calls WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self.connection.execute(
                    """
                    DELETE FROM model_requests
                    WHERE turn_id IN (
                        SELECT id FROM turns WHERE conversation_id = ?
                    )
                    """,
                    (conversation_id,),
                )
                self.connection.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self.connection.execute(
                    "DELETE FROM turns WHERE conversation_id = ?",
                    (conversation_id,),
                )
                self.connection.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"无法删除会话：{exc}") from exc

        blobs_deleted, cleanup_failures = self._cleanup_unreferenced_blobs(
            candidate_blobs
        )
        return DeleteConversationOutcome(
            conversation_id=conversation.id,
            title=conversation.title,
            turns_deleted=counts["turns"],
            messages_deleted=counts["messages"],
            tool_calls_deleted=counts["tool_calls"],
            snapshots_deleted=counts["snapshots"],
            blobs_deleted=blobs_deleted,
            blob_cleanup_failures=cleanup_failures,
        )

    def begin_turn(self, conversation_id: str, user_input: str) -> TurnRecord:
        now = utc_now()
        # 在数据库中创建一个新的对话轮次记录，包含会话 ID、轮次序号、用户输入、状态和时间戳。如果会话不存在，将抛出 StorageError 异常。如果在插入过程中发生 SQLite 错误，也将抛出 StorageError 异常。
        try:
            with self.connection:
                conversation = self.connection.execute(
                    "SELECT id FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if conversation is None:
                    raise StorageError(f"会话不存在：{conversation_id}")
                sequence = self.connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM turns WHERE conversation_id = ?
                    """,
                    (conversation_id,),
                ).fetchone()[0]
                cursor = self.connection.execute(
                    """
                    INSERT INTO turns(
                        conversation_id, sequence, user_input, status, started_at
                    ) VALUES (?, ?, ?, 'running', ?)
                    """,
                    (conversation_id, sequence, user_input, now),
                )
                turn_id = int(cursor.lastrowid)
                self._append_messages(
                    conversation_id,
                    turn_id,
                    [{"role": "user", "content": user_input}],
                )
                self.connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
        except StorageError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"无法创建对话轮次：{exc}") from exc
        return TurnRecord(
            id=turn_id,
            conversation_id=conversation_id,
            sequence=sequence,
            user_input=user_input,
            status="running",
        )

    def load_messages(
        self,
        conversation_id: str,
        *,
        after_turn_sequence: int = 0,
        through_turn_sequence: int | None = None,
    ) -> list[dict[str, Any]]:
        # 从数据库中加载指定会话的所有消息，按顺序返回一个包含消息内容的列表。查询会排除状态为“interrupted”的轮次，以确保只获取完整的消息记录。如果消息的 JSON 数据损坏，将抛出 StorageError 异常。
        parameters: list[Any] = [conversation_id, after_turn_sequence]
        upper_bound = ""
        if through_turn_sequence is not None:
            upper_bound = "AND t.sequence <= ?"
            parameters.append(through_turn_sequence)
        rows = self.connection.execute(
            f"""
            SELECT m.payload_json
            FROM messages m
            JOIN turns t ON t.id = m.turn_id
            WHERE m.conversation_id = ?
              AND t.status NOT IN ('interrupted', 'rolled_back')
              AND t.sequence > ?
              {upper_bound}
            ORDER BY m.sequence
            """,
            parameters,
        ).fetchall()
        try:
            return [json.loads(row["payload_json"]) for row in rows]
        except json.JSONDecodeError as exc:
            raise StorageError("数据库中的消息 JSON 已损坏。") from exc

    def record_model_response(self, turn_id: int, step: int, response: Any) -> None:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        now = utc_now()
        # 向数据库中记录模型响应的相关信息，包括轮次 ID、步骤号、响应 ID、完成原因、使用的 token 数量等。如果在插入过程中发生 SQLite 错误，将抛出 StorageError 异常。
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO model_requests(
                        turn_id, step, status, response_id, finish_reason,
                        prompt_tokens, completion_tokens, total_tokens, created_at
                    ) VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_id,
                        step,
                        getattr(response, "id", None),
                        getattr(choice, "finish_reason", None),
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                        getattr(usage, "total_tokens", None),
                        now,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"无法记录模型响应：{exc}") from exc

    def record_model_error(self, turn_id: int, step: int, error: Exception) -> None:
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO model_requests(
                        turn_id, step, status, error_type, error_message, created_at
                    ) VALUES (?, ?, 'failed', ?, ?, ?)
                    """,
                    (turn_id, step, type(error).__name__, str(error), utc_now()),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"无法记录模型错误：{exc}") from exc

    def record_assistant_message(
        self,
        conversation_id: str,
        turn_id: int,
        message: dict[str, Any],
    ) -> None:
        try:
            with self.connection:
                self._append_messages(conversation_id, turn_id, [message])
        except sqlite3.Error as exc:
            raise StorageError(f"无法记录助手消息：{exc}") from exc

    def record_tool_exchange(
        self,
        conversation_id: str,
        turn_id: int,
        step: int,
        assistant: dict[str, Any],
        executions: list[ToolExecution],
        tool_messages: list[dict[str, Any]],
    ) -> None:
        if len(executions) != len(tool_messages):
            raise StorageError("工具执行记录和工具消息数量不一致。")
        now = utc_now()
        try:
            with self.connection:
                self._append_messages(conversation_id, turn_id, [assistant])
                for execution in executions:
                    cursor = self.connection.execute(
                        """
                        INSERT INTO tool_calls(
                            conversation_id, turn_id, step, call_id, tool_name,
                            raw_arguments, arguments_json, status, duration_ms,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conversation_id,
                            turn_id,
                            step,
                            execution.tool_call_id,
                            execution.tool_name,
                            execution.raw_arguments,
                            self._json(execution.arguments),
                            execution.status,
                            execution.duration_ms,
                            now,
                        ),
                    )
                    database_tool_call_id = int(cursor.lastrowid)
                    self.connection.execute(
                        """
                        INSERT INTO tool_results(
                            tool_call_id, ok, content, details_json, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            database_tool_call_id,
                            int(execution.result.ok),
                            execution.result.content,
                            self._json(execution.result.details),
                            now,
                        ),
                    )
                    self._record_file_change(
                        conversation_id,
                        turn_id,
                        database_tool_call_id,
                        execution,
                        now,
                    )
                    self._record_approval(
                        database_tool_call_id,
                        execution,
                        now,
                    )
                self._append_messages(
                    conversation_id,
                    turn_id,
                    tool_messages,
                )
        except sqlite3.Error as exc:
            raise StorageError(f"无法记录工具调用链：{exc}") from exc

    def finish_turn(
        self,
        turn_id: int,
        status: str,
        *,
        error: Exception | None = None,
    ) -> None:
        now = utc_now()
        try:
            with self.connection:
                row = self.connection.execute(
                    "SELECT conversation_id FROM turns WHERE id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise StorageError(f"对话轮次不存在：{turn_id}")
                self.connection.execute(
                    """
                    UPDATE turns SET
                        status = ?, finished_at = ?,
                        error_type = ?, error_message = ?,
                        model_steps = (
                            SELECT COUNT(*) FROM model_requests WHERE turn_id = ?
                        ),
                        tool_calls = (
                            SELECT COUNT(*) FROM tool_calls WHERE turn_id = ?
                        )
                    WHERE id = ?
                    """,
                    (
                        status,
                        now,
                        type(error).__name__ if error else None,
                        str(error) if error else None,
                        turn_id,
                        turn_id,
                        turn_id,
                    ),
                )
                self.connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, row["conversation_id"]),
                )
        except StorageError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"无法结束对话轮次：{exc}") from exc

    def get_turn(self, turn_id: int) -> TurnRecord:
        row = self.connection.execute(
            "SELECT * FROM turns WHERE id = ?",
            (turn_id,),
        ).fetchone()
        if row is None:
            raise StorageError(f"对话轮次不存在：{turn_id}")
        return TurnRecord(
            id=row["id"],
            conversation_id=row["conversation_id"],
            sequence=row["sequence"],
            user_input=row["user_input"],
            status=row["status"],
        )

    def create_snapshot(
        self,
        conversation_id: str,
        turn_id: int,
        kind: str,
        entries: list[SnapshotEntryRecord],
    ) -> int:
        # 在数据库中创建一个新的工作区快照记录，包含会话 ID、轮次 ID、快照类型（before 或 after）、快照条目数量、总字节数和时间戳。如果快照类型不支持，将抛出 StorageError 异常。如果在插入过程中发生 SQLite 错误，也将抛出 StorageError 异常。返回新创建的快照 ID。
        if kind not in {"before", "after"}:
            raise StorageError(f"不支持的快照类型：{kind}")
        now = utc_now()
        try:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    INSERT INTO snapshots(
                        conversation_id, turn_id, kind, entry_count,
                        total_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        turn_id,
                        kind,
                        len(entries),
                        sum(entry.size for entry in entries if entry.kind == "file"),
                        now,
                    ),
                )
                snapshot_id = int(cursor.lastrowid)
                self.connection.executemany(
                    """
                    INSERT INTO snapshot_entries(
                        snapshot_id, path, kind, blob_hash, size, mode
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            snapshot_id,
                            entry.path,
                            entry.kind,
                            entry.blob_hash,
                            entry.size,
                            entry.mode,
                        )
                        for entry in entries
                    ],
                )
                self.connection.execute(
                    """
                    INSERT INTO turn_snapshots(turn_id, before_snapshot_id, after_snapshot_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(turn_id) DO UPDATE SET
                        before_snapshot_id = COALESCE(
                            excluded.before_snapshot_id,
                            turn_snapshots.before_snapshot_id
                        ),
                        after_snapshot_id = COALESCE(
                            excluded.after_snapshot_id,
                            turn_snapshots.after_snapshot_id
                        )
                    """,
                    (
                        turn_id,
                        snapshot_id if kind == "before" else None,
                        snapshot_id if kind == "after" else None,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"无法保存工作区快照：{exc}") from exc
        return snapshot_id

    def get_snapshot_entries(
        self,
        snapshot_id: int,
    ) -> list[SnapshotEntryRecord]:
        rows = self.connection.execute(
            """
            SELECT path, kind, blob_hash, size, mode
            FROM snapshot_entries
            WHERE snapshot_id = ?
            ORDER BY path
            """,
            (snapshot_id,),
        ).fetchall()
        return [
            SnapshotEntryRecord(
                path=row["path"],
                kind=row["kind"],
                blob_hash=row["blob_hash"],
                size=row["size"],
                mode=row["mode"],
            )
            for row in rows
        ]

    def turn_for_diff(
        self,
        conversation_id: str,
        sequence: int | None = None,
    ) -> RollbackTarget:
        parameters: list[Any] = [conversation_id]
        sequence_filter = ""
        if sequence is not None:
            sequence_filter = "AND t.sequence = ?"
            parameters.append(sequence)
        else:
            sequence_filter = "AND t.status != 'rolled_back'"
        row = self.connection.execute(
            f"""
            SELECT t.*, ts.before_snapshot_id, ts.after_snapshot_id
            FROM turns t
            JOIN turn_snapshots ts ON ts.turn_id = t.id
            WHERE t.conversation_id = ?
              AND ts.before_snapshot_id IS NOT NULL
              AND ts.after_snapshot_id IS NOT NULL
              {sequence_filter}
            ORDER BY t.sequence DESC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
        if row is None:
            label = f"第 {sequence} 轮" if sequence is not None else "当前会话"
            raise StorageError(f"{label}没有可用的前后快照。")
        return RollbackTarget(
            turn=TurnRecord(
                id=row["id"],
                conversation_id=row["conversation_id"],
                sequence=row["sequence"],
                user_input=row["user_input"],
                status=row["status"],
            ),
            before_snapshot_id=row["before_snapshot_id"],
            after_snapshot_id=row["after_snapshot_id"],
        )

    def latest_rollback_target(self, conversation_id: str) -> RollbackTarget:
        return self.turn_for_diff(conversation_id)

    def record_rollback(self, target: RollbackTarget) -> None:
        now = utc_now()
        try:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    UPDATE turns SET status = 'rolled_back'
                    WHERE id = ? AND status != 'rolled_back'
                    """,
                    (target.turn.id,),
                )
                if cursor.rowcount != 1:
                    raise StorageError("该轮次已经回滚，不能重复回滚。")
                self.connection.execute(
                    """
                    INSERT INTO rollbacks(
                        conversation_id, target_turn_id, original_status,
                        restored_snapshot_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        target.turn.conversation_id,
                        target.turn.id,
                        target.turn.status,
                        target.before_snapshot_id,
                        now,
                    ),
                )
                self.connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, target.turn.conversation_id),
                )
                self.connection.execute(
                    """
                    DELETE FROM context_summaries
                    WHERE conversation_id = ? AND through_turn_sequence >= ?
                    """,
                    (target.turn.conversation_id, target.turn.sequence),
                )
        except StorageError:
            raise
        except sqlite3.Error as exc:
            raise StorageError(f"无法记录回滚操作：{exc}") from exc

    def _append_messages(
        self,
        conversation_id: str,
        turn_id: int,
        messages: list[dict[str, Any]],
    ) -> None:
        if not messages:
            return
        sequence = self.connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1
            FROM messages WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()[0]
        now = utc_now()
        for offset, message in enumerate(messages):
            self.connection.execute(
                """
                INSERT INTO messages(
                    conversation_id, turn_id, sequence, role, content,
                    tool_call_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn_id,
                    sequence + offset,
                    message.get("role"),
                    message.get("content"),
                    message.get("tool_call_id"),
                    self._json(message),
                    now,
                ),
            )

    def _cleanup_unreferenced_blobs(
        self,
        candidates: set[str],
    ) -> tuple[int, int]:
        blob_root = self.database_path.parent / "snapshots" / "blobs"
        deleted = 0
        failures = 0
        for blob_hash in candidates:
            referenced = self.connection.execute(
                "SELECT 1 FROM snapshot_entries WHERE blob_hash = ? LIMIT 1",
                (blob_hash,),
            ).fetchone()
            if referenced is not None:
                continue
            blob_path = blob_root / blob_hash[:2] / blob_hash[2:]
            try:
                if not blob_path.exists():
                    continue
                blob_path.unlink()
                deleted += 1
                try:
                    blob_path.parent.rmdir()
                except OSError:
                    pass
            except OSError:
                failures += 1
        return deleted, failures

    def _record_file_change(
        self,
        conversation_id: str,
        turn_id: int,
        database_tool_call_id: int,
        execution: ToolExecution,
        created_at: str,
    ) -> None:
        details = execution.result.details
        change_type = details.get("change_type")
        path = details.get("path")
        if (
            not execution.result.ok
            or change_type not in {"created", "modified", "deleted"}
            or not isinstance(path, str)
        ):
            return
        self.connection.execute(
            """
            INSERT INTO file_changes(
                conversation_id, turn_id, tool_call_id, path, change_type,
                existed_before, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                turn_id,
                database_tool_call_id,
                path,
                change_type,
                int(bool(details.get("existed_before"))),
                created_at,
            ),
        )

    def _record_approval(
        self,
        database_tool_call_id: int,
        execution: ToolExecution,
        created_at: str,
    ) -> None:
        details = execution.result.details
        status = details.get("approval_status")
        if not isinstance(status, str):
            return
        self.connection.execute(
            """
            INSERT INTO approval_requests(
                tool_call_id, status, risk_level, reason, summary,
                audit_advice, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                database_tool_call_id,
                status,
                str(details.get("approval_risk", "unknown")),
                str(details.get("approval_reason", "")),
                str(details.get("approval_summary", "")),
                details.get("audit_advice"),
                created_at,
            ),
        )

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO schema_meta(key, value)
            VALUES ('schema_version', '1');

            CREATE TABLE IF NOT EXISTS conversations(
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                workspace TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                sequence INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                status TEXT NOT NULL,
                error_type TEXT,
                error_message TEXT,
                model_steps INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                UNIQUE(conversation_id, sequence)
            );

            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                turn_id INTEGER NOT NULL REFERENCES turns(id),
                sequence INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(conversation_id, sequence)
            );

            CREATE TABLE IF NOT EXISTS model_requests(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL REFERENCES turns(id),
                step INTEGER NOT NULL,
                status TEXT NOT NULL,
                response_id TEXT,
                finish_reason TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                error_type TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(turn_id, step)
            );

            CREATE TABLE IF NOT EXISTS tool_calls(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                turn_id INTEGER NOT NULL REFERENCES turns(id),
                step INTEGER NOT NULL,
                call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                raw_arguments TEXT NOT NULL,
                arguments_json TEXT,
                status TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(turn_id, call_id)
            );

            CREATE TABLE IF NOT EXISTS tool_results(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_call_id INTEGER NOT NULL UNIQUE REFERENCES tool_calls(id),
                ok INTEGER NOT NULL,
                content TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_changes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                turn_id INTEGER NOT NULL REFERENCES turns(id),
                tool_call_id INTEGER NOT NULL REFERENCES tool_calls(id),
                path TEXT NOT NULL,
                change_type TEXT NOT NULL,
                existed_before INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS approval_requests(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_call_id INTEGER NOT NULL UNIQUE REFERENCES tool_calls(id),
                status TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                reason TEXT NOT NULL,
                summary TEXT NOT NULL,
                audit_advice TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                turn_id INTEGER NOT NULL REFERENCES turns(id),
                kind TEXT NOT NULL,
                entry_count INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(turn_id, kind)
            );

            CREATE TABLE IF NOT EXISTS snapshot_entries(
                snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
                path TEXT NOT NULL,
                kind TEXT NOT NULL,
                blob_hash TEXT,
                size INTEGER NOT NULL,
                mode INTEGER NOT NULL,
                PRIMARY KEY(snapshot_id, path)
            );

            CREATE TABLE IF NOT EXISTS turn_snapshots(
                turn_id INTEGER PRIMARY KEY REFERENCES turns(id),
                before_snapshot_id INTEGER REFERENCES snapshots(id),
                after_snapshot_id INTEGER REFERENCES snapshots(id)
            );

            CREATE TABLE IF NOT EXISTS rollbacks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                target_turn_id INTEGER NOT NULL UNIQUE REFERENCES turns(id),
                original_status TEXT NOT NULL,
                restored_snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS context_summaries(
                conversation_id TEXT PRIMARY KEY REFERENCES conversations(id),
                through_turn_sequence INTEGER NOT NULL,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_turns_conversation
                ON turns(conversation_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_turn
                ON tool_calls(turn_id, step);
            CREATE INDEX IF NOT EXISTS idx_file_changes_turn
                ON file_changes(turn_id, id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_turn
                ON snapshots(turn_id, kind);

            UPDATE schema_meta SET value = '4'
            WHERE key = 'schema_version' AND CAST(value AS INTEGER) < 4;
            """
        )

    def _recover_interrupted_turns(self) -> None: # Recover any turns that were left in a "running" state when the process exited unexpectedly, marking them as "interrupted" to indicate they were not completed. This ensures that the system can handle unexpected shutdowns gracefully and maintain data integrity.
        now = utc_now()
        with self.connection:
            self.connection.execute(
                """
                UPDATE turns SET
                    status = 'interrupted',
                    error_type = 'ProcessInterrupted',
                    error_message = '上次进程在本轮完成前退出。',
                    finished_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> ConversationRecord:
        return ConversationRecord(
            id=row["id"],
            title=row["title"],
            workspace=row["workspace"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_turn_status=(
                row["last_turn_status"]
                if "last_turn_status" in row.keys()
                else None
            ),
        )
