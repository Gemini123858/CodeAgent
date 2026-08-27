from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath

from ..workspace import Workspace, WorkspaceError
from .result import ToolResult


DEFAULT_ALLOWED_COMMANDS = frozenset(
    {
        "cp",
        "find",
        "git",
        "head",
        "ls",
        "mkdir",
        "mv",
        "pwd",
        "python",
        "python3",
        "tail",
        "touch",
        "wc",
    }
)

SHELL_OPERATORS = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "<<"})
SAFE_GIT_SUBCOMMANDS = frozenset({"diff", "log", "show", "status", "ls-files"})
SENSITIVE_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class CommandToolError(RuntimeError):
    """Raised when a command is invalid or rejected by policy."""


class CommandTool:
    def __init__(
        self,
        workspace: Workspace,
        *,
        timeout_seconds: int = 30,
        max_output_chars: int = 12_000,
        allowed_commands: frozenset[str] = DEFAULT_ALLOWED_COMMANDS,
    ) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.allowed_commands = allowed_commands

    def run_command(self, command: str | list[str]) -> ToolResult:
        argv = self._parse(command)
        self._validate(argv)

        try:
            completed = subprocess.run(
                argv,
                cwd=self.workspace.root,
                env=self._sanitized_environment(),
                shell=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            partial = self._join_output(exc.stdout or "", exc.stderr or "")
            return ToolResult.failure(
                partial or f"命令执行超过 {self.timeout_seconds} 秒，已终止。",
                command=shlex.join(argv),
                timeout_seconds=self.timeout_seconds,
            )
        except OSError as exc:
            raise CommandToolError(f"无法启动命令：{argv[0]}") from exc

        output = self._join_output(completed.stdout, completed.stderr)
        if len(output) > self.max_output_chars:
            output = "... 输出前部已截断 ...\n" + output[-self.max_output_chars :]
        if not output:
            output = "命令没有输出。"

        details = {
            "command": shlex.join(argv),
            "exit_code": completed.returncode,
        }
        if completed.returncode == 0:
            return ToolResult.success(output, **details)
        return ToolResult.failure(output, **details)

    def _parse(self, command: str | list[str]) -> list[str]:
        if isinstance(command, str):
            try:
                argv = shlex.split(command, posix=True) # Split the command string into a list of arguments using shell-like syntax. This handles quoted strings and escaped characters appropriately.
            except ValueError as exc:
                raise CommandToolError(f"命令格式错误：{exc}") from exc
        elif isinstance(command, list) and all(isinstance(item, str) for item in command):
            argv = command.copy()
        else:
            raise CommandToolError("命令必须是字符串或字符串列表。")

        if not argv:
            raise CommandToolError("命令不能为空。")
        return argv

    def _validate(self, argv: list[str]) -> None:
        executable = argv[0]
        if executable not in self.allowed_commands:
            raise CommandToolError(f"命令不在允许列表中：{executable}")
        if any(argument in SHELL_OPERATORS for argument in argv):
            raise CommandToolError("不允许使用 Shell 管道、重定向或命令连接符。")

        if executable in {"python", "python3"} and "-c" in argv[1:]:
            raise CommandToolError("不允许通过 python -c 执行内联代码。")
        if executable in {"python", "python3"} and any(
            argv[index : index + 2] in (["-m", "pip"], ["-m", "ensurepip"])
            for index in range(1, len(argv) - 1)
        ):
            raise CommandToolError("不允许通过命令工具安装 Python 依赖。")
        if executable == "git":
            if len(argv) < 2 or argv[1] not in SAFE_GIT_SUBCOMMANDS:
                raise CommandToolError("当前阶段只允许只读 Git 子命令。")
        if executable == "find" and any(
            argument in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
            for argument in argv[1:]
        ):
            raise CommandToolError("不允许使用具有执行或删除能力的 find 参数。")

        for argument in argv[1:]:
            value = argument.split("=", 1)[-1] if "=" in argument else argument
            if value.startswith("-") or not value:
                continue
            if ".." in PurePosixPath(value).parts:
                raise CommandToolError("命令参数不允许包含父目录跳转。")
            looks_like_path = (
                Path(value).is_absolute()
                or "/" in value
                or "\\" in value
                or value.startswith(".")
                or (self.workspace.root / value).exists()
            )
            if looks_like_path:
                try:
                    self.workspace.resolve(value)
                except WorkspaceError as exc:
                    raise CommandToolError(str(exc)) from exc

    def _sanitized_environment(self) -> dict[str, str]:
        return {
            key: value
            for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS)
        }

    @staticmethod
    def _join_output(stdout: str, stderr: str) -> str:
        sections = []
        if stdout.strip():
            sections.append(stdout.rstrip())
        if stderr.strip():
            sections.append(f"stderr:\n{stderr.rstrip()}")
        return "\n".join(sections)
