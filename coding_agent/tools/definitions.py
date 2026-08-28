from __future__ import annotations

from typing import Any


def tool_definitions() -> list[dict[str, Any]]:
    """Return the function schemas sent to the model."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "列出工作区指定目录中的直接子项。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "工作区内的相对目录，默认是当前目录。",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取工作区内的 UTF-8 文本文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "工作区内的相对文件路径。",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "在工作区内创建或覆盖 UTF-8 文本文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "工作区内的相对文件路径。",
                        },
                        "content": {
                            "type": "string",
                            "description": "需要写入的完整文件内容。",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "在工作区内执行一个允许的、非 Shell 开发命令。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "命令字符串，例如 python3 hello.py。",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]
