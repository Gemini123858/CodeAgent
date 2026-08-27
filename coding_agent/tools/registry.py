from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..workspace import Workspace
from .filesystem import FileToolError, FileTools
from .result import ToolResult
from .shell import CommandTool, CommandToolError


ToolHandler = Callable[..., ToolResult]


class ToolRegistry:
    def __init__(self, workspace: Workspace, *, command_timeout: int = 30) -> None:
        file_tools = FileTools(workspace)
        command_tool = CommandTool(workspace, timeout_seconds=command_timeout)
        self._handlers: dict[str, ToolHandler] = {
            "list_files": file_tools.list_files,
            "read_file": file_tools.read_file,
            "write_file": file_tools.write_file,
            "run_command": command_tool.run_command,
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult.failure(f"未知工具：{name}")
        if not isinstance(arguments, Mapping):
            return ToolResult.failure("工具参数必须是对象。")

        try:
            return handler(**dict(arguments)) # 将 Mapping 转换为字典，以确保可以使用关键字参数调用处理程序。
        except (FileToolError, CommandToolError) as exc:
            return ToolResult.failure(str(exc))
        except TypeError as exc:
            return ToolResult.failure(f"工具参数错误：{exc}")
        except Exception as exc:
            return ToolResult.failure(f"工具执行出现未预期错误：{type(exc).__name__}")
