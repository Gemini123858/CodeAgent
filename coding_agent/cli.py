from __future__ import annotations

import argparse
import sys

from .config import ConfigurationError, load_settings
from .llm_client import LLMClient, LLMRequestError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="第一阶段：验证 DeepSeek 单轮对话 API"
    )
    parser.add_argument("prompt", nargs="?", help="发送给模型的一条消息")
    return parser


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
        reply = LLMClient(settings).chat(prompt)
    except (ConfigurationError, LLMRequestError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
