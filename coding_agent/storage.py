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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class SessionStore:
    """SQLite-backed append-only conversation and tool execution store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
            self._recover_interrupted_turns()
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
                self.connection.execute(
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

    def begin_turn(self, conversation_id: str, user_input: str) -> TurnRecord:
        now = utc_now()
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

    def load_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT m.payload_json
            FROM messages m
            JOIN turns t ON t.id = m.turn_id
            WHERE m.conversation_id = ? AND t.status != 'interrupted'
            ORDER BY m.sequence
            """,
            (conversation_id,),
        ).fetchall()
        try:
            return [json.loads(row["payload_json"]) for row in rows]
        except json.JSONDecodeError as exc:
            raise StorageError("数据库中的消息 JSON 已损坏。") from exc

    def record_model_response(self, turn_id: int, step: int, response: Any) -> None:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        now = utc_now()
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
                    self.connection.execute(
                        """
                        INSERT INTO tool_results(
                            tool_call_id, ok, content, details_json, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            int(cursor.lastrowid),
                            int(execution.result.ok),
                            execution.result.content,
                            self._json(execution.result.details),
                            now,
                        ),
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

            CREATE INDEX IF NOT EXISTS idx_turns_conversation
                ON turns(conversation_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_turn
                ON tool_calls(turn_id, step);
            """
        )

    def _recover_interrupted_turns(self) -> None:
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
