from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath

from ..approval import (
    ApprovalProvider,
    ApprovalRequest,
    DenyApprovalProvider,
)
from ..command_policy import (
    AuditAdvice,
    CommandAuditor,
    CommandPolicy,
    PolicyAction,
)
from ..turn_context import TurnContext
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
    """Raised when a command is invalid or rejected by hard policy."""


class CommandTool:
    def __init__(
        self,
        workspace: Workspace,
        *,
        timeout_seconds: int = 30,
        max_output_chars: int = 12_000,
        max_stdin_chars: int = 20_000,
        allowed_commands: frozenset[str] = DEFAULT_ALLOWED_COMMANDS,
        policy: CommandPolicy | None = None,
        approval_provider: ApprovalProvider | None = None,
        auditor: CommandAuditor | None = None,
    ) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.max_stdin_chars = max_stdin_chars
        self.allowed_commands = allowed_commands
        self.policy = policy or CommandPolicy()
        self.approval_provider = approval_provider or DenyApprovalProvider()
        self.auditor = auditor

    def run_command(
        self,
        command: str | list[str],
        stdin: str | None = None,
        *,
        turn_context: TurnContext | None = None,
    ) -> ToolResult:
        argv = self._parse(command)
        self._validate(argv)
        normalized_stdin = self._validate_stdin(stdin)
        active_context = turn_context or TurnContext()

        assessment = self.policy.assess(argv, self.workspace)
        if assessment.action is PolicyAction.DENY:
            raise CommandToolError(assessment.reason)

        audit_advice = self._audit(shlex.join(argv), argv, active_context)
        needs_approval = (
            assessment.action is PolicyAction.REQUIRE_APPROVAL
            or (
                audit_advice is not None
                and audit_advice.recommendation.lower()
                in {"deny", "review", "require_approval"}
            )
        )
        approval_details: dict[str, object] = {}
        if needs_approval:
            fingerprint = self._fingerprint(argv, normalized_stdin)
            # 检查是否已经有缓存的审批决策
            cached = active_context.approval_decision(fingerprint)
            if cached is None:
                approved = self.approval_provider.request(
                    ApprovalRequest(
                        tool_name="run_command",
                        summary=f"运行命令：{shlex.join(argv)}",
                        reason=assessment.reason,
                        risk_level=assessment.risk_level,
                        fingerprint=fingerprint,
                        audit_advice=audit_advice.reason if audit_advice else None,
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
                "approval_risk": assessment.risk_level,
                "approval_reason": assessment.reason,
                "approval_summary": f"运行命令：{shlex.join(argv)}",
                "audit_advice": audit_advice.reason if audit_advice else None,
            }
            if not approved:
                return ToolResult.failure(
                    "用户未批准该命令。请不要重复请求相同命令，改用更安全的方案。",
                    **approval_details,
                )

        try:
            completed = subprocess.run(
                argv,
                cwd=self.workspace.root,
                env=self._sanitized_environment(),
                shell=False,
                input=normalized_stdin,
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
                stdin_chars=len(normalized_stdin or ""),
                **approval_details,
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
            "stdin_chars": len(normalized_stdin or ""),
            **approval_details,
        }
        if completed.returncode == 0:
            return ToolResult.success(output, **details)
        return ToolResult.failure(output, **details)

    def _parse(self, command: str | list[str]) -> list[str]:
        if isinstance(command, str):
            try:
                argv = shlex.split(command, posix=True)
            except ValueError as exc:
                raise CommandToolError(f"命令格式错误：{exc}") from exc
        elif isinstance(command, list) and all(
            isinstance(item, str) for item in command
        ):
            argv = command.copy()
        else:
            raise CommandToolError("命令必须是字符串或字符串列表。")

        if not argv:
            raise CommandToolError("命令不能为空。")
        return argv

    def _validate(self, argv: list[str]) -> None:
        executable = argv[0]
        if any(argument in SHELL_OPERATORS for argument in argv):
            raise CommandToolError(
                "不允许使用 Shell 管道、重定向或连接符；"
                "需要向程序输入内容时请使用 run_command 的 stdin 参数。"
            )
        if executable not in self.allowed_commands:
            raise CommandToolError(f"命令不在允许列表中：{executable}")

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
            if any(
                argument in {"-c", "--ext-diff", "--textconv", "--config-env"}
                or argument == "--output"
                or argument.startswith("--output=")
                for argument in argv[2:]
            ):
                raise CommandToolError("不允许使用可写文件或执行外部程序的 Git 参数。")
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

    def _validate_stdin(self, stdin: str | None) -> str | None:
        if stdin is not None and not isinstance(stdin, str):
            raise CommandToolError("stdin 必须是字符串或 null。")
        if stdin is not None and len(stdin) > self.max_stdin_chars:
            raise CommandToolError(
                f"stdin 过长：{len(stdin)} 字符，限制为 {self.max_stdin_chars}。"
            )
        return stdin

    def _audit(
        self,
        command: str,
        argv: list[str],
        turn_context: TurnContext,
    ) -> AuditAdvice | None:
        if self.auditor is None:
            return None
        try:
            return self.auditor.audit(command, argv, turn_context)
        except Exception as exc:
            return AuditAdvice(
                recommendation="review",
                reason=f"命令审计器执行失败：{type(exc).__name__}",
            )

    @staticmethod
    def _fingerprint(argv: list[str], stdin: str | None) -> str:
        payload = "\0".join([*argv, stdin or ""])
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"run_command:{digest}"

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
