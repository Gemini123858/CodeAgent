from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(RuntimeError):
    """Base class for workspace validation errors."""


class PathOutsideWorkspaceError(WorkspaceError):
    """Raised when a requested path escapes the configured workspace."""


class ProtectedPathError(WorkspaceError):
    """Raised when a requested path targets protected repository metadata."""


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        if not root.exists():
            raise WorkspaceError(f"工作目录不存在：{root}")
        if not root.is_dir():
            raise WorkspaceError(f"工作路径不是目录：{root}")
        object.__setattr__(self, "root", root)

    @classmethod
    def from_path(cls, path: str | Path) -> Workspace: # Create a Workspace instance from a path.
        return cls(Path(path))

    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else self.root / raw_path
        resolved = candidate.resolve(strict=False)

        try:
            relative = resolved.relative_to(self.root) # Get the relative path of the resolved path with respect to the workspace root. If the resolved path is not within the workspace, this will raise a ValueError.
        except ValueError as exc:
            raise PathOutsideWorkspaceError(
                f"路径超出工作目录：{path}"
            ) from exc

        protected = next(
            (part for part in relative.parts if self._is_protected_name(part)),
            None,
        )
        if protected is not None:
            raise ProtectedPathError(f"不允许直接操作受保护路径：{protected}")
        if must_exist and not resolved.exists():
            raise WorkspaceError(f"路径不存在：{path}")
        return resolved

    def display(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."

    @staticmethod
    def _is_protected_name(name: str) -> bool:
        return name == ".git" or name == "agent_env" or name.startswith(".env")

    def is_protected(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return True
        return any(self._is_protected_name(part) for part in relative.parts)
