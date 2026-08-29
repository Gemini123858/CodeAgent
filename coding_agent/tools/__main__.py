from __future__ import annotations

import argparse
import sys

from ..approval import CLIApprovalProvider
from ..turn_context import TurnContext
from ..workspace import Workspace, WorkspaceError
from .registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立测试 Coding Agent 本地工具")
    parser.add_argument("--workspace", default=".", help="工具可操作的工作目录")
    subparsers = parser.add_subparsers(dest="tool", required=True)

    list_parser = subparsers.add_parser("list-files", help="列出目录")
    list_parser.add_argument("path", nargs="?", default=".")

    read_parser = subparsers.add_parser("read-file", help="读取 UTF-8 文本文件")
    read_parser.add_argument("path")

    write_parser = subparsers.add_parser("write-file", help="创建或覆盖 UTF-8 文本文件")
    write_parser.add_argument("path")
    write_parser.add_argument("--content", required=True)

    delete_parser = subparsers.add_parser("delete-file", help="删除普通文件")
    delete_parser.add_argument("path")

    command_parser = subparsers.add_parser("run-command", help="运行允许的本地命令")
    command_parser.add_argument("command", nargs=argparse.REMAINDER)
    command_parser.add_argument("--stdin", dest="stdin_text")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        registry = ToolRegistry(
            Workspace.from_path(args.workspace),
            approval_provider=CLIApprovalProvider(),
        )
    except WorkspaceError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if args.tool == "list-files":
        result = registry.execute("list_files", {"path": args.path})
    elif args.tool == "read-file":
        result = registry.execute("read_file", {"path": args.path})
    elif args.tool == "write-file":
        result = registry.execute(
            "write_file", {"path": args.path, "content": args.content}
        )
    elif args.tool == "delete-file":
        result = registry.execute("delete_file", {"path": args.path})
    else:
        # print(args.command)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        result = registry.execute(
            "run_command",
            {"command": command, "stdin": args.stdin_text},
            turn_context=TurnContext(),
        )

    print(result.render(), file=sys.stdout if result.ok else sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
