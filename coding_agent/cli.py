from __future__ import annotations

import argparse
import sys

from .agent import AgentError, AgentRunner
from .config import ConfigurationError, load_settings
from .debug import DebugPrinter, debug_enabled
from .llm_client import LLMClient, LLMRequestError
from .tool_call import SingleToolCallRunner, ToolCallError
from .tools import ToolRegistry
from .workspace import Workspace, WorkspaceError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="轻量级 Coding Agent 分阶段实验"
    )
    parser.add_argument("prompt", nargs="?", help="发送给模型的一条消息")
    mode = parser.add_mutually_exclusive_group()
    # 默认保留第一阶段的普通对话；工具和 Agent 模式需显式启用。
    mode.add_argument(
        "--tool-call",
        action="store_true",
        help="运行第三阶段的单次工具调用闭环",
    )
    mode.add_argument(
        "--agent",
        action="store_true",
        help="运行第四阶段的单次任务 Agent 循环",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="工具允许操作的工作目录，默认为当前目录",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="向 stderr 输出脱敏后的关键请求、响应和工具信息",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        help="Agent 最大模型请求次数，默认读取 CODING_AGENT_MAX_STEPS",
    )
    return parser


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        prompt = args.prompt or input("请输入消息：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。", file=sys.stderr)
        return 130

    if not prompt.strip():
        print("错误：消息不能为空。", file=sys.stderr)
        return 2

    try:
        settings = load_settings()
        print(f"正在调用模型：{settings.model}", file=sys.stderr)
        debug = DebugPrinter(enabled=debug_enabled(args.debug))
        llm = LLMClient(settings, debug)
        if args.agent:
            workspace = Workspace.from_path(args.workspace)
            runner = AgentRunner(
                llm,
                ToolRegistry(workspace),
                max_steps=args.max_steps or settings.max_steps,
                debug=debug,
                progress=lambda message: print(message, file=sys.stderr),
            )
            outcome = runner.run(prompt)
            print("\n模型最终回答：")
            print(outcome.final_answer)
            print(
                f"\n执行统计：{outcome.steps} 次模型请求，"
                f"{len(outcome.tool_executions)} 次工具调用，"
                f"保留 {outcome.retained_messages} 条消息。"
            )
        elif args.tool_call:
            workspace = Workspace.from_path(args.workspace)
            runner = SingleToolCallRunner(
                llm,
                ToolRegistry(workspace),
                debug,
            )
            outcome = runner.run(prompt)
            print(f"工具调用：{outcome.tool_name}")
            print(f"工具参数：{outcome.arguments}")
            print("工具结果：")
            print(outcome.tool_result.render())
            print("\n模型最终回答：")
            print(outcome.final_answer)
        else:
            print(llm.chat(prompt))
    except (
        ConfigurationError,
        AgentError,
        LLMRequestError,
        ToolCallError,
        WorkspaceError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
