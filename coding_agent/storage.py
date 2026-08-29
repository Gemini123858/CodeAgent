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

    def begin_turn(self, conversation_id: str, user_input: str) -> TurnRecord:
        now = utc_now()
        # 先检查会话是否存在，如果不存在则抛出 StorageError 异常。
        # 然后计算该会话的下一个轮次序号，并在 turns 表中插入一条新的记录，表示一个新的对话轮次开始。
        # 最后，将用户输入作为消息记录到 messages 表中，并更新 conversations 表中的更新时间。
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
        # 从数据库中加载指定会话的所有消息，按顺序返回一个包含消息内容的列表。查询会排除状态为“interrupted”的轮次，以确保只获取完整的消息记录。如果消息的 JSON 数据损坏，将抛出 StorageError 异常。
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

            CREATE INDEX IF NOT EXISTS idx_turns_conversation
                ON turns(conversation_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_turn
                ON tool_calls(turn_id, step);
            CREATE INDEX IF NOT EXISTS idx_file_changes_turn
                ON file_changes(turn_id, id);

            UPDATE schema_meta SET value = '2'
            WHERE key = 'schema_version' AND CAST(value AS INTEGER) < 2;
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
