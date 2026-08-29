from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..approval import (
    ApprovalProvider,
    ApprovalRequest,
    DenyApprovalProvider,
)
from ..command_policy import CommandAuditor
from ..turn_context import TurnContext
from ..workspace import Workspace
from .filesystem import FileToolError, FileTools
from .result import ToolResult
from .shell import CommandTool, CommandToolError


ToolHandler = Callable[..., ToolResult]

# A registry for managing and executing various tools.
class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        *,
        command_timeout: int = 30,
        approval_provider: ApprovalProvider | None = None,
        command_auditor: CommandAuditor | None = None,
    ) -> None:
        self.file_tools = FileTools(workspace)
        self.approval_provider = approval_provider or DenyApprovalProvider()
        self.command_tool = CommandTool(
            workspace,
            timeout_seconds=command_timeout,
            approval_provider=self.approval_provider,
            auditor=command_auditor,
        )
        self._handlers: dict[str, ToolHandler] = {
            "list_files": self.file_tools.list_files,
            "read_file": self.file_tools.read_file,
            "write_file": self.file_tools.write_file,
        }

    @property
    def names(self) -> tuple[str, ...]:
        return (*self._handlers, "delete_file", "run_command")

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        turn_context: TurnContext | None = None,
    ) -> ToolResult:
        if not isinstance(arguments, Mapping):
            return ToolResult.failure("工具参数必须是对象。")

        try:
            if name == "run_command":
                result = self.command_tool.run_command(
                    **dict(arguments),
                    turn_context=turn_context,
                )
            elif name == "delete_file":
                result = self._delete_file(arguments, turn_context)
            else:
                handler = self._handlers.get(name)
                if handler is None:
                    return ToolResult.failure(f"未知工具：{name}")
                result = handler(**dict(arguments))
                if (
                    name == "write_file"
                    and result.ok
                    and result.details.get("change_type") == "created"
                    and turn_context is not None
                ):
                    turn_context.record_created_file(str(result.details["path"]))
            return result
        except (FileToolError, CommandToolError) as exc:
            return ToolResult.failure(str(exc))
        except TypeError as exc:
            return ToolResult.failure(f"工具参数错误：{exc}")
        except Exception as exc:
            return ToolResult.failure(
                f"工具执行出现未预期错误：{type(exc).__name__}"
            )

    def _delete_file(
        self,
        arguments: Mapping[str, Any],
        turn_context: TurnContext | None,
    ) -> ToolResult:
        if set(arguments) != {"path"} or not isinstance(
            arguments.get("path"),
            str,
        ):
            return ToolResult.failure("delete_file 只接受字符串参数 path。")
        raw_path = str(arguments["path"])
        normalized_path = self.file_tools.normalize_path(
            raw_path,
            must_exist=True,
        )
        active_context = turn_context or TurnContext()
        auto_allowed = active_context.can_delete_without_approval(normalized_path)
        approval_details: dict[str, object] = {}

        if not auto_allowed:
            fingerprint = f"delete_file:{normalized_path}"
            cached = active_context.approval_decision(fingerprint)
            reason = "该文件不是当前对话轮次通过 write_file 新建的文件。"
            if cached is None:
                approved = self.approval_provider.request(
                    ApprovalRequest(
                        tool_name="delete_file",
                        summary=f"删除既有文件：{normalized_path}",
                        reason=reason,
                        risk_level="high",
                        fingerprint=fingerprint,
                        details={"path": normalized_path},
                    )
                )
                active_context.remember_approval(fingerprint, approved)
                approval_status = "approved" if approved else "denied"
            else:
                approved = cached
                approval_status = (
                    "reused_approved" if approved else "reused_denied"
                )
            approval_details = {
                "approval_status": approval_status,
                "approval_risk": "high",
                "approval_reason": reason,
                "approval_summary": f"删除既有文件：{normalized_path}",
            }
            if not approved:
                return ToolResult.failure(
                    "用户未批准删除该文件。请不要重复请求相同操作。",
                    path=normalized_path,
                    **approval_details,
                )

        result = self.file_tools.delete_file(raw_path)
        active_context.record_deleted_file(normalized_path)
        return ToolResult(
            ok=result.ok,
            content=result.content,
            details={
                **result.details,
                "existed_before": not auto_allowed,
                **approval_details,
            },
        )
