from __future__ import annotations

import argparse
import sys

from .agent import AgentError, AgentOutcome, AgentRunner
from .approval import CLIApprovalProvider
from .config import ConfigurationError, Settings, load_settings
from .debug import DebugPrinter, debug_enabled
from .llm_client import LLMClient, LLMRequestError
from .session import ConversationSession, title_from_prompt
from .storage import SessionStore, StorageError
from .tool_call import SingleToolCallRunner, ToolCallError
from .tools import ToolRegistry
from .workspace import Workspace, WorkspaceError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="轻量级本地 Coding Agent"
    )
    parser.add_argument("prompt", nargs="?", help="发送给模型的一条消息")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tool-call",
        action="store_true",
        help="运行单次工具调用闭环",
    )
    mode.add_argument(
        "--agent",
        action="store_true",
        help="运行一次持久化 Agent 对话轮次",
    )
    mode.add_argument(
        "--session",
        action="store_true",
        help="进入支持 /new 和 /resume 的多轮 Agent 会话",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="工具允许操作的工作目录，默认为当前目录",
    )
    parser.add_argument(
        "--conversation",
        help="恢复完整会话 ID 或唯一 ID 前缀（仅 --agent/--session）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="向 stderr 输出脱敏后的关键请求、响应和工具信息",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        help="每轮最大模型请求次数，默认读取 CODING_AGENT_MAX_STEPS",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=_positive_int,
        help="每轮最大工具调用数，默认读取 CODING_AGENT_MAX_TOOL_CALLS",
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
    if args.conversation and not (args.agent or args.session):
        parser.error("--conversation 只能和 --agent 或 --session 一起使用")

    try:
        settings = load_settings()
        print(f"正在调用模型：{settings.model}", file=sys.stderr)
        debug = DebugPrinter(enabled=debug_enabled(args.debug))
        llm = LLMClient(settings, debug)

        if args.session:
            return _run_interactive_session(args, settings, llm, debug)

        prompt = _get_prompt(args.prompt)
        if args.agent:
            return _run_agent_once(args, settings, llm, debug, prompt)
        if args.tool_call:
            workspace = Workspace.from_path(args.workspace)
            runner = SingleToolCallRunner(
                llm,
                ToolRegistry(
                    workspace,
                    approval_provider=CLIApprovalProvider(),
                ),
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
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。", file=sys.stderr)
        return 130
    except (
        ConfigurationError,
        AgentError,
        LLMRequestError,
        StorageError,
        ToolCallError,
        WorkspaceError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


def _get_prompt(prompt: str | None) -> str:
    value = prompt if prompt is not None else input("请输入消息：")
    cleaned = value.strip()
    if not cleaned:
        raise AgentError("消息不能为空。")
    return cleaned


def _build_agent(
    args: argparse.Namespace,
    settings: Settings,
    llm: LLMClient,
    debug: DebugPrinter,
) -> tuple[Workspace, AgentRunner]:
    workspace = Workspace.from_path(args.workspace)
    runner = AgentRunner(
        llm,
        ToolRegistry(
            workspace,
            approval_provider=CLIApprovalProvider(),
        ),
        max_steps=args.max_steps or settings.max_steps,
        max_tool_calls=args.max_tool_calls or settings.max_tool_calls,
        debug=debug,
        progress=lambda message: print(message, file=sys.stderr),
    )
    return workspace, runner


def _run_agent_once(
    args: argparse.Namespace,
    settings: Settings,
    llm: LLMClient,
    debug: DebugPrinter,
    prompt: str,
) -> int:
    # 构建工作区和 Agent 运行器，然后创建或恢复会话，执行一次对话轮次，并打印结果。
    workspace, runner = _build_agent(args, settings, llm, debug)
    with SessionStore.for_workspace(workspace.root) as store:
        session = ConversationSession(workspace.root, store, runner)
        if args.conversation: # 如果提供了 --conversation 参数，则尝试恢复指定的会话，否则创建一个新的会话。
            conversation = session.resume(args.conversation)
        else:
            conversation = session.new(title_from_prompt(prompt))
        print(f"会话：{conversation.id[:8]}  {conversation.title}", file=sys.stderr)
        outcome = session.run_turn(prompt)
        _print_outcome(outcome)
    return 0


def _run_interactive_session(
    args: argparse.Namespace,
    settings: Settings,
    llm: LLMClient,
    debug: DebugPrinter,
) -> int:
    # 构建工作区和 Agent 运行器，然后创建或恢复会话，进入一个交互式循环，允许用户输入消息或会话命令，并在每轮对话后打印结果。
    workspace, runner = _build_agent(args, settings, llm, debug)
    with SessionStore.for_workspace(workspace.root) as store: # Create a session store for the workspace root directory, which will handle the storage of conversation sessions, turns, messages, and tool executions in a SQLite database.
        session = ConversationSession(workspace.root, store, runner)
        conversation = (
            session.resume(args.conversation)
            if args.conversation
            else session.resume_latest_or_new() # 优先恢复最新的会话，如果没有则创建一个新的会话。
        )
        _print_session_banner(conversation.id, conversation.title)

        if args.prompt:
            _run_session_turn(session, args.prompt)

        while True:
            try:
                line = input(f"agent[{session.current.id[:8]}]> ").strip()
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print("\n已退出会话。")
                return 130
            if not line:
                continue
            if not line.startswith("/"):
                _run_session_turn(session, line)
                continue
            if _handle_session_command(session, store, line):
                return 0


def _handle_session_command(
    session: ConversationSession,
    store: SessionStore,
    line: str,
) -> bool:
    command, _, argument = line.partition(" ")
    argument = argument.strip()
    try:
        if command in {"/exit", "/quit"}:
            return True
        if command == "/new":
            conversation = session.new(argument or None)
            print(f"已创建会话：{conversation.id[:8]}  {conversation.title}")
        elif command == "/resume":
            if not argument:
                print("用法：/resume <会话 ID 或唯一前缀>", file=sys.stderr)
            else:
                conversation = session.resume(argument)
                print(f"已恢复会话：{conversation.id[:8]}  {conversation.title}")
        elif command == "/list":
            conversations = store.list_conversations()
            if not conversations:
                print("暂无会话。")
            for item in conversations:
                marker = "*" if session.current and item.id == session.current.id else " "
                status = item.last_turn_status or "empty"
                print(f"{marker} {item.id[:8]}  {status:<20} {item.title}")
        elif command == "/help":
            print("/new [标题]  /resume <ID>  /list  /help  /exit")
        else:
            print(f"未知命令：{command}；输入 /help 查看帮助。", file=sys.stderr)
    except (AgentError, StorageError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
    return False


def _run_session_turn(session: ConversationSession, prompt: str) -> None:
    try:
        outcome = session.run_turn(prompt)
    except KeyboardInterrupt:
        print("\n当前轮次已中断，仍可继续会话。", file=sys.stderr)
    except (AgentError, LLMRequestError, StorageError, WorkspaceError) as exc:
        print(f"本轮失败：{exc}", file=sys.stderr)
    else:
        _print_outcome(outcome)


def _print_outcome(outcome: AgentOutcome) -> None:
    print("\n模型最终回答：")
    print(outcome.final_answer)
    print(
        f"\n执行统计：{outcome.steps} 次模型请求，"
        f"{len(outcome.tool_executions)} 次工具调用，"
        f"保留 {outcome.retained_messages} 条消息。"
    )


def _print_session_banner(conversation_id: str, title: str) -> None:
    print(f"当前会话：{conversation_id[:8]}  {title}")
    print("输入 /help 查看会话命令，/exit 退出。")


if __name__ == "__main__":
    raise SystemExit(main())
