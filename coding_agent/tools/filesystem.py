from __future__ import annotations

from pathlib import Path

from ..workspace import Workspace, WorkspaceError
from .result import ToolResult


class FileToolError(RuntimeError):
    """Raised when a filesystem tool cannot complete an operation."""


class FileTools:
    def __init__(
        self,
        workspace: Workspace,
        *,
        max_file_bytes: int = 100_000,
        max_list_entries: int = 200,
    ) -> None:
        self.workspace = workspace
        self.max_file_bytes = max_file_bytes
        self.max_list_entries = max_list_entries

    def list_files(self, path: str = ".") -> ToolResult:
        # print(f"Listing files in: {path}")  # Debug statement
        directory = self._resolve(path, must_exist=True)
        if not directory.is_dir():
            raise FileToolError(f"不是目录：{path}")

        try:
            entries = sorted(
                (
                    entry
                    for entry in directory.iterdir()
                    if not self.workspace.is_protected(entry)
                ),
                key=lambda entry: (entry.is_file(), entry.name.lower()),
            )
        except OSError as exc:
            raise FileToolError(f"无法读取目录：{path}") from exc

        visible = entries[: self.max_list_entries]
        lines = []
        for entry in visible:
            kind = "D" if entry.is_dir() else "F" if entry.is_file() else "L"
            lines.append(f"[{kind}] {self.workspace.display(entry)}")

        if len(entries) > len(visible):
            lines.append(f"... 还有 {len(entries) - len(visible)} 项未显示")

        return ToolResult.success(
            "\n".join(lines) if lines else "目录为空。",
            path=self.workspace.display(directory),
            entries=len(entries),
        )

    def read_file(self, path: str) -> ToolResult:
        file_path = self._resolve(path, must_exist=True)
        if not file_path.is_file():
            raise FileToolError(f"不是普通文件：{path}")

        try:
            size = file_path.stat().st_size
            if size > self.max_file_bytes:
                raise FileToolError(
                    f"文件过大：{size} 字节，限制为 {self.max_file_bytes} 字节。"
                )
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise FileToolError(f"文件不是有效的 UTF-8 文本：{path}") from exc
        except OSError as exc:
            raise FileToolError(f"无法读取文件：{path}") from exc

        return ToolResult.success(
            content,
            path=self.workspace.display(file_path),
            bytes=size,
        )

    def write_file(self, path: str, content: str) -> ToolResult:
        if not isinstance(content, str):
            raise FileToolError("写入内容必须是字符串。")

        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise FileToolError(
                f"写入内容过大：{len(encoded)} 字节，限制为 "
                f"{self.max_file_bytes} 字节。"
            )

        file_path = self._resolve(path)
        if file_path.exists() and file_path.is_dir():
            raise FileToolError(f"目标是目录，不能写入：{path}")
        existed = file_path.exists()

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise FileToolError(f"无法写入文件：{path}") from exc

        action = "已覆盖" if existed else "已创建"
        return ToolResult.success(
            f"{action}文件：{self.workspace.display(file_path)}",
            path=self.workspace.display(file_path),
            bytes=len(encoded),
            change_type="modified" if existed else "created",
            existed_before=existed,
        )

    def delete_file(self, path: str) -> ToolResult:
        file_path = self._resolve(path, must_exist=True)
        raw_path = Path(path)
        lexical_path = (
            raw_path if raw_path.is_absolute() else self.workspace.root / raw_path
        )
        if lexical_path.is_symlink():
            raise FileToolError("不允许删除符号链接。")
        if not file_path.is_file():
            raise FileToolError(f"只能删除普通文件：{path}")

        try:
            size = file_path.stat().st_size
            file_path.unlink()
        except OSError as exc:
            raise FileToolError(f"无法删除文件：{path}") from exc

        display_path = self.workspace.display(file_path)
        return ToolResult.success(
            f"已删除文件：{display_path}",
            path=display_path,
            bytes=size,
            change_type="deleted",
        )

    def normalize_path(self, path: str, *, must_exist: bool = False) -> str:
        return self.workspace.display(self._resolve(path, must_exist=must_exist))

    def _resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        try:
            return self.workspace.resolve(path, must_exist=must_exist)
        except WorkspaceError as exc:
            raise FileToolError(str(exc)) from exc
