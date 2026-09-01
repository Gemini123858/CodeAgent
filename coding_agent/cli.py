from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from .agent import AgentError, AgentOutcome, AgentRunner
from .approval import CLIApprovalProvider
from .command_policy import LLMCommandAuditor
from .config import ConfigurationError, Settings, load_settings
from .debug import DebugPrinter, debug_enabled
from .llm_client import LLMClient, LLMRequestError
from .session import (
    ContextCompressionOutcome,
    ConversationSession,
    title_from_prompt,
)
from .snapshot import SnapshotError
from .storage import ConversationHistoryTurn, SessionStore, StorageError
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
                    command_auditor=(
                        LLMCommandAuditor(llm)
                        if settings.llm_command_audit
                        else None
                    ),
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
        SnapshotError,
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
            command_auditor=(
                LLMCommandAuditor(llm) if settings.llm_command_audit else None
            ),
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
        session = ConversationSession(
            workspace.root,
            store,
            runner,
            context_token_limit=settings.context_token_limit,
        )
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
        session = ConversationSession(
            workspace.root,
            store,
            runner,
            context_token_limit=settings.context_token_limit,
        )
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
                print(
                    f"{marker} {item.id[:8]}  "
                    f"{_status_text(status)}  {item.title}"
                )
        elif command == "/diff":
            sequence = None
            if argument:
                try:
                    sequence = int(argument)
                except ValueError:
                    print("用法：/diff [Turn 序号]", file=sys.stderr)
                    return False
                if sequence <= 0:
                    print("Turn 序号必须大于 0。", file=sys.stderr)
                    return False
            _print_diff(session.render_diff(sequence))
        elif command == "/rollback":
            if argument:
                print("用法：/rollback", file=sys.stderr)
            else:
                outcome = session.rollback()
                print(
                    f"已回滚 Turn {outcome.turn_sequence}，"
                    f"恢复 {outcome.restored_paths} 个路径。"
                )
        elif command == "/history":
            limit = 20
            if argument:
                try:
                    limit = int(argument)
                except ValueError:
                    print("用法：/history [1-100]", file=sys.stderr)
                    return False
                if not 1 <= limit <= 100:
                    print("历史记录数量必须在 1 到 100 之间。", file=sys.stderr)
                    return False
            _print_history(session.history(limit))
        elif command == "/context":
            if argument:
                print("用法：/context", file=sys.stderr)
                return False
            _print_context_outcome(session.compact_context())
        elif command in {"/delete", "/delete-session"}:
            if argument:
                print("用法：/delete", file=sys.stderr)
                return False
            assert session.current is not None
            current = session.current
            try:
                answer = input(
                    f"确认删除会话 {current.id[:8]}（{current.title}）"
                    "及其全部历史和快照？[y/N] "
                ).strip().lower()
            except EOFError:
                answer = ""
            if answer not in {"y", "yes"}:
                print("已取消删除。")
                return False
            outcome, next_conversation = session.delete_current()
            print(
                f"已删除会话 {outcome.conversation_id[:8]}："
                f"{outcome.turns_deleted} 个 Turn、"
                f"{outcome.messages_deleted} 条消息、"
                f"{outcome.tool_calls_deleted} 次工具调用、"
                f"{outcome.snapshots_deleted} 个快照、"
                f"{outcome.blobs_deleted} 个独占 blob。"
            )
            if outcome.blob_cleanup_failures:
                print(
                    f"警告：{outcome.blob_cleanup_failures} 个无引用 blob "
                    "未能从磁盘清理。",
                    file=sys.stderr,
                )
            print(
                f"当前会话已切换为：{next_conversation.id[:8]}  "
                f"{next_conversation.title}"
            )
        elif command == "/help":
            print(
                "/new [标题]  /resume <ID>  /list  /diff [Turn]  "
                "/rollback  /history [数量]  /context  /delete  /help  /exit"
            )
        else:
            print(f"未知命令：{command}；输入 /help 查看帮助。", file=sys.stderr)
    except (AgentError, SnapshotError, StorageError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
    return False


def _run_session_turn(session: ConversationSession, prompt: str) -> None:
    try:
        outcome = session.run_turn(prompt)
    except KeyboardInterrupt:
        print("\n当前轮次已中断，仍可继续会话。", file=sys.stderr)
    except (
        AgentError,
        LLMRequestError,
        SnapshotError,
        StorageError,
        WorkspaceError,
    ) as exc:
        print(f"本轮失败：{exc}", file=sys.stderr)
    else:
        _print_outcome(outcome)


def _print_outcome(outcome: AgentOutcome) -> None:
    print("\n模型最终回答：")
    print(outcome.final_answer)
    print(
        f"\n执行统计：{outcome.steps} 次模型请求，"
        f"{len(outcome.tool_executions)} 次工具调用，"
        f"保留 {outcome.retained_messages} 条消息；"
        f"Token {outcome.token_usage.total_tokens} "
        f"（输入 {outcome.token_usage.prompt_tokens} / "
        f"输出 {outcome.token_usage.completion_tokens}，"
        f"{'API usage' if outcome.token_usage.source == 'api' else '本地估算'}）。"
    )


def _print_context_outcome(outcome: ContextCompressionOutcome) -> None:
    before_percent = outcome.before_tokens / outcome.limit_tokens * 100
    trigger_percent = outcome.trigger_tokens / outcome.limit_tokens * 100
    print(
        f"上下文窗口：约 {outcome.before_tokens} / {outcome.limit_tokens} tokens "
        f"（{before_percent:.1f}%）"
    )
    print(
        f"压缩阈值：{outcome.trigger_tokens} tokens "
        f"（{trigger_percent:.0f}%）"
    )
    if outcome.summary_through_sequence is not None:
        print(f"当前摘要已覆盖至 Turn {outcome.summary_through_sequence}。")
    if outcome.compressed:
        after_percent = outcome.after_tokens / outcome.limit_tokens * 100
        print(
            f"{outcome.reason} 压缩后约 {outcome.after_tokens} tokens "
            f"（{after_percent:.1f}%）。"
        )
    else:
        print(f"本次未压缩：{outcome.reason}")


def _print_session_banner(conversation_id: str, title: str) -> None:
    print(f"当前会话：{conversation_id[:8]}  {title}")
    print("输入 /help 查看会话命令，/exit 退出。")


def _print_diff(diff_text: str) -> None:
    print(
        "阅读方式："
        f"{_paint('绿色 + 是新增行', '32')}，"
        f"{_paint('红色 - 是删除行', '31')}，"
        f"{_paint('青色 @@ 标出所在行', '36')}；"
        "--- / +++ 分别表示修改前 / 修改后的文件。"
    )
    print(_paint("黄色内容是二进制文件、权限或文件类型等特殊变化。", "33"))
    print("─" * 72)
    for line in diff_text.splitlines():
        if line.startswith("Turn "):
            print(_paint(line, "1;36"))
        elif line.startswith("diff --coding-agent"):
            print(_paint(line, "1"))
        elif line.startswith("+++ "):
            print(_paint(line, "1;32"))
        elif line.startswith("--- "):
            print(_paint(line, "1;31"))
        elif line.startswith("+"):
            print(_paint(line, "32"))
        elif line.startswith("-"):
            print(_paint(line, "31"))
        elif line.startswith("@@"):
            print(_paint(line, "36"))
        elif line.startswith(
            ("Binary ", "empty file ", "mode change ", "symlink ", "typechange ")
        ):
            print(_paint(line, "33"))
        elif "diff 输出已截断" in line:
            print(_paint(line, "33"))
        else:
            print(line)


def _print_history(turns: list[ConversationHistoryTurn]) -> None:
    if not turns:
        print("当前会话还没有历史交流记录。")
        return
    print(f"最近 {len(turns)} 个 Turn（按时间正序）：")
    for turn in turns:
        timestamp = _local_time(turn.started_at)
        print("\n" + "─" * 72)
        print(
            f"Turn {turn.sequence}  {_status_text(turn.status)}  "
            f"{_paint(timestamp, '2')}"
        )
        tool_names: dict[str, str] = {}
        for message in turn.messages:
            role = message.get("role")
            if role == "user":
                _print_history_block("你", str(message.get("content") or ""), "34")
            elif role == "assistant":
                content = message.get("content")
                if content:
                    _print_history_block("Agent", str(content), "32")
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {})
                    name = str(function.get("name", "<unknown>"))
                    call_id = str(call.get("id", ""))
                    tool_names[call_id] = name
                    arguments = _summarize_tool_arguments(
                        str(function.get("arguments", "{}"))
                    )
                    _print_history_block(
                        "工具请求",
                        f"{name}({arguments})",
                        "33",
                        max_chars=500,
                    )
            elif role == "tool":
                call_id = str(message.get("tool_call_id", ""))
                name = tool_names.get(call_id, call_id or "<unknown>")
                status, content = _tool_history_result(message.get("content"))
                color = "32" if status == "success" else "31"
                _print_history_block(
                    f"工具结果 {name} [{status}]",
                    content,
                    color,
                    max_chars=500,
                )


def _print_history_block(
    label: str,
    content: str,
    color: str,
    *,
    max_chars: int = 2_000,
) -> None:
    cleaned = content.strip() or "<空>"
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + f"\n... 已截断 {len(cleaned) - max_chars} 字符 ..."
    prefix = _paint(f"{label}：", color)
    lines = cleaned.splitlines() or [""]
    print(f"{prefix}{lines[0]}")
    for line in lines[1:]:
        print(f"  {line}")


def _summarize_tool_arguments(raw_arguments: str) -> str:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return _shorten(raw_arguments, 200)
    if not isinstance(arguments, dict):
        return _shorten(str(arguments), 200)
    parts = []
    for key, value in arguments.items():
        if key in {"content", "stdin"} and isinstance(value, str):
            parts.append(f"{key}=<{len(value)} chars>")
        else:
            parts.append(f"{key}={_shorten(repr(value), 100)}")
    return ", ".join(parts)


def _tool_history_result(raw_content: object) -> tuple[str, str]:
    try:
        payload = json.loads(str(raw_content or "{}"))
    except json.JSONDecodeError:
        return "unknown", str(raw_content or "")
    if not isinstance(payload, dict):
        return "unknown", str(payload)
    return str(payload.get("status", "unknown")), str(payload.get("content", ""))


def _shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _local_time(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return timestamp


def _status_text(status: str) -> str:
    labels = {
        "empty": "无记录",
        "completed": "已完成",
        "running": "进行中",
        "rolled_back": "已回滚",
        "failed": "失败",
        "interrupted": "已中断",
        "snapshot_failed": "快照失败",
        "tool_limit_exceeded": "工具调用超限",
        "max_steps_exceeded": "步骤超限",
        "output_truncated": "模型输出截断",
    }
    colors = {
        "completed": "32",
        "running": "36",
        "rolled_back": "33",
        "failed": "31",
        "interrupted": "31",
        "snapshot_failed": "31",
    }
    return _paint(f"[{labels.get(status, status)}]", colors.get(status, "33"))


def _paint(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _color_enabled() -> bool:
    return (
        sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.getenv("TERM", "").lower() != "dumb"
    )


if __name__ == "__main__":
    raise SystemExit(main())
