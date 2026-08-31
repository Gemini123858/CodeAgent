from __future__ import annotations

import difflib
import hashlib
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .storage import (
    RollbackTarget,
    SessionStore,
    SnapshotEntryRecord,
    StorageError,
)
from .workspace import Workspace


SNAPSHOT_DIRECTORY_NAME = "snapshots"
BLOB_DIRECTORY_NAME = "blobs"
DIFF_HIDDEN_PARTS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})


class SnapshotError(RuntimeError):
    """Raised when a complete workspace snapshot cannot be captured or restored."""


class SnapshotConflictError(SnapshotError):
    """Raised when the workspace no longer matches the latest Turn's after state."""


@dataclass(frozen=True)
class RollbackOutcome:
    turn_sequence: int
    restored_paths: int


class WorkspaceSnapshotManager:
    """Content-addressed Turn snapshots used by diff and rollback."""

    def __init__(
        self,
        workspace: Workspace,
        store: SessionStore,
        *,
        max_file_bytes: int = 20_000_000,
        max_total_bytes: int = 200_000_000,
        max_entries: int = 20_000,
        max_diff_chars: int = 60_000,
    ) -> None:
        self.workspace = workspace
        self.store = store
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_entries = max_entries
        self.max_diff_chars = max_diff_chars
        self.blob_root = (
            store.database_path.parent
            / SNAPSHOT_DIRECTORY_NAME
            / BLOB_DIRECTORY_NAME
        )
        self.blob_root.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        conversation_id: str,
        turn_id: int,
        kind: str,
    ) -> int:
        entries = self._scan(write_blobs=True)
        return self.store.create_snapshot(
            conversation_id,
            turn_id,
            kind,
            list(entries.values()),
        )

    def render_diff(
        self,
        conversation_id: str,
        sequence: int | None = None,
    ) -> str:
        target = self.store.turn_for_diff(conversation_id, sequence)
        before = self._load_manifest(target.before_snapshot_id)
        after = self._load_manifest(target.after_snapshot_id)
        changed = self._changed_paths(before, after)
        visible = [path for path in changed if self._visible_in_diff(path)]
        header = (
            f"Turn {target.turn.sequence} diff "
            f"(status={target.turn.status}, changed={len(changed)})"
        )
        if not changed:
            return f"{header}\n本轮未改变工作区。"
        if not visible:
            return f"{header}\n本轮仅改变了未显示的运行缓存文件。"

        sections = [header]
        for path in visible:
            rendered = self._render_path_diff(path, before.get(path), after.get(path))
            if rendered:
                sections.append(rendered)
            if sum(len(section) for section in sections) > self.max_diff_chars:
                sections.append("... diff 输出已截断 ...")
                break
        return "\n\n".join(sections)

    def rollback_latest(self, conversation_id: str) -> RollbackOutcome:
        target = self.store.latest_rollback_target(conversation_id)
        expected_after = self._load_manifest(target.after_snapshot_id)
        restore_to = self._load_manifest(target.before_snapshot_id)
        current = self._scan(write_blobs=True)
        conflicts = self._changed_paths(expected_after, current)
        if conflicts:
            preview = "、".join(conflicts[:8])
            suffix = "……" if len(conflicts) > 8 else ""
            raise SnapshotConflictError(
                "工作区在该 Turn 完成后又发生了外部变化，已拒绝回滚。"
                f"冲突路径：{preview}{suffix}"
            )

        changed_paths = self._changed_paths(current, restore_to)
        try:
            self._restore_manifest(restore_to, current)
            restored = self._scan(write_blobs=False)
            remaining = self._changed_paths(restore_to, restored)
            if remaining:
                raise SnapshotError(
                    "回滚后工作区校验失败：" + "、".join(remaining[:8])
                )
            self.store.record_rollback(target)
        except Exception as exc:
            try:
                now = self._scan(write_blobs=True)
                self._restore_manifest(current, now)
            except Exception as recovery_exc:
                raise SnapshotError(
                    "回滚失败，且安全恢复也失败："
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                ) from exc
            if isinstance(exc, (SnapshotError, StorageError)):
                raise
            raise SnapshotError(f"回滚工作区失败：{exc}") from exc

        return RollbackOutcome(
            turn_sequence=target.turn.sequence,
            restored_paths=len(changed_paths),
        )

    def _scan(self, *, write_blobs: bool) -> dict[str, SnapshotEntryRecord]:
        # 扫描工作区目录，收集所有文件、目录和符号链接的元数据，并计算其内容的 SHA-256 哈希值。将这些信息存储在一个字典中，其中键是相对于工作区根目录的路径，值是 SnapshotEntryRecord 对象。该方法还会检查文件大小和总大小限制，并在超过限制时抛出 SnapshotError 异常。
        entries: dict[str, SnapshotEntryRecord] = {}
        total_bytes = 0

        def on_error(error: OSError) -> None:
            raise SnapshotError(f"无法扫描工作区：{error}")

        for directory, dirnames, filenames in os.walk(
            self.workspace.root,
            topdown=True,
            followlinks=False,
            onerror=on_error,
        ):
            directory_path = Path(directory)
            if directory_path != self.workspace.root:
                self._add_directory_entry(entries, directory_path)

            kept_directories: list[str] = []
            for name in sorted(dirnames):
                path = directory_path / name
                if self.workspace.is_protected(path):
                    continue
                if path.is_symlink():
                    total_bytes += self._add_symlink_entry(
                        entries,
                        path,
                        write_blobs,
                    )
                else:
                    kept_directories.append(name)
            dirnames[:] = kept_directories

            for name in sorted(filenames):
                path = directory_path / name
                if self.workspace.is_protected(path):
                    continue
                try:
                    metadata = path.lstat()
                    if stat.S_ISLNK(metadata.st_mode):
                        total_bytes += self._add_symlink_entry(
                            entries,
                            path,
                            write_blobs,
                        )
                    elif stat.S_ISREG(metadata.st_mode):
                        total_bytes += self._add_file_entry(
                            entries,
                            path,
                            metadata,
                            write_blobs,
                        )
                    else:
                        raise SnapshotError(
                            "工作区包含无法安全快照的特殊文件："
                            f"{self._relative(path)}"
                        )
                except OSError as exc:
                    raise SnapshotError(f"无法读取快照路径 {path}: {exc}") from exc

                if total_bytes > self.max_total_bytes:
                    raise SnapshotError(
                        f"工作区快照超过总大小限制 {self.max_total_bytes} 字节。"
                    )
                if len(entries) > self.max_entries:
                    raise SnapshotError(
                        f"工作区快照超过条目限制 {self.max_entries}。"
                    )
        return entries

    def _add_directory_entry(
        self,
        entries: dict[str, SnapshotEntryRecord],
        path: Path,
    ) -> None:
        metadata = path.lstat()
        relative = self._relative(path)
        entries[relative] = SnapshotEntryRecord(
            path=relative,
            kind="directory",
            blob_hash=None,
            size=0,
            mode=stat.S_IMODE(metadata.st_mode),
        )

    def _add_file_entry(
        self,
        entries: dict[str, SnapshotEntryRecord],
        path: Path,
        metadata: os.stat_result,
        write_blobs: bool,
    ) -> int:
        if metadata.st_size > self.max_file_bytes:
            raise SnapshotError(
                f"文件过大，无法纳入快照：{self._relative(path)} "
                f"({metadata.st_size} 字节)"
            )
        data = path.read_bytes()
        blob_hash = hashlib.sha256(data).hexdigest()
        if write_blobs:
            self._store_blob(blob_hash, data)
        relative = self._relative(path)
        entries[relative] = SnapshotEntryRecord(
            path=relative,
            kind="file",
            blob_hash=blob_hash,
            size=len(data),
            mode=stat.S_IMODE(metadata.st_mode),
        )
        return len(data)

    def _add_symlink_entry(
        self,
        entries: dict[str, SnapshotEntryRecord],
        path: Path,
        write_blobs: bool,
    ) -> int:
        target = os.readlink(path)
        data = target.encode("utf-8", errors="surrogateescape")
        blob_hash = hashlib.sha256(data).hexdigest()
        if write_blobs:
            self._store_blob(blob_hash, data)
        relative = self._relative(path)
        entries[relative] = SnapshotEntryRecord(
            path=relative,
            kind="symlink",
            blob_hash=blob_hash,
            size=len(data),
            mode=stat.S_IMODE(path.lstat().st_mode),
        )
        return len(data)

    def _store_blob(self, blob_hash: str, data: bytes) -> None:
        blob_path = self._blob_path(blob_hash)
        if blob_path.exists():
            return
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = blob_path.parent / f".{blob_hash}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(data)
            os.replace(temporary, blob_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load_manifest(self, snapshot_id: int) -> dict[str, SnapshotEntryRecord]:
        return {
            entry.path: entry
            for entry in self.store.get_snapshot_entries(snapshot_id)
        }

    @staticmethod
    def _changed_paths(
        left: dict[str, SnapshotEntryRecord],
        right: dict[str, SnapshotEntryRecord],
    ) -> list[str]:
        return sorted(
            path
            for path in set(left) | set(right)
            if left.get(path) != right.get(path)
        )

    def _render_path_diff(
        self,
        path: str,
        before: SnapshotEntryRecord | None,
        after: SnapshotEntryRecord | None,
    ) -> str:
        if (before and before.kind == "directory") or (
            after and after.kind == "directory"
        ):
            return ""
        if before and after and before.kind != after.kind:
            return f"typechange {path}: {before.kind} -> {after.kind}"
        if (before and before.kind == "symlink") or (
            after and after.kind == "symlink"
        ):
            old_target = self._symlink_target(before) if before else "/dev/null"
            new_target = self._symlink_target(after) if after else "/dev/null"
            return f"symlink {path}\n- {old_target}\n+ {new_target}"

        before_data = self._entry_bytes(before)
        after_data = self._entry_bytes(after)
        old_name = f"a/{path}" if before else "/dev/null"
        new_name = f"b/{path}" if after else "/dev/null"
        try:
            before_text = before_data.decode("utf-8")
            after_text = after_data.decode("utf-8")
            is_binary = "\x00" in before_text or "\x00" in after_text
        except UnicodeDecodeError:
            is_binary = True
            before_text = ""
            after_text = ""
        if is_binary:
            action = "changed"
            if before is None:
                action = "added"
            elif after is None:
                action = "deleted"
            return f"Binary file {path} {action}"

        lines = difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile=old_name,
            tofile=new_name,
            n=3,
            lineterm="",
        )
        rendered = "\n".join(lines).rstrip()
        if rendered:
            return f"diff --coding-agent {old_name} {new_name}\n{rendered}"
        if before is None:
            return f"empty file {path} added"
        if after is None:
            return f"empty file {path} deleted"
        if before and after and before.mode != after.mode:
            return f"mode change {path}: {before.mode:o} -> {after.mode:o}"
        return ""

    def _restore_manifest(
        self,
        target: dict[str, SnapshotEntryRecord],
        current: dict[str, SnapshotEntryRecord],
    ) -> None:
        removable = [
            entry
            for path, entry in current.items()
            if path not in target or target[path].kind != entry.kind
        ]
        removable.sort(
            key=lambda entry: (len(PurePosixPath(entry.path).parts), entry.kind),
            reverse=True,
        )
        for entry in removable:
            path = self._workspace_path(entry.path)
            if entry.kind == "directory":
                path.rmdir()
            else:
                path.unlink(missing_ok=True)

        directories = sorted(
            (entry for entry in target.values() if entry.kind == "directory"),
            key=lambda entry: len(PurePosixPath(entry.path).parts),
        )
        for entry in directories:
            path = self._workspace_path(entry.path)
            path.mkdir(parents=True, exist_ok=True)

        for entry in sorted(target.values(), key=lambda item: item.path):
            path = self._workspace_path(entry.path)
            if entry.kind == "directory":
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if entry.kind == "symlink":
                if path.exists() or path.is_symlink():
                    path.unlink()
                os.symlink(self._symlink_target(entry), path)
                continue
            data = self._entry_bytes(entry)
            temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.restore"
            try:
                temporary.write_bytes(data)
                os.chmod(temporary, entry.mode)
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()

        for entry in reversed(directories):
            os.chmod(self._workspace_path(entry.path), entry.mode)

    def _entry_bytes(self, entry: SnapshotEntryRecord | None) -> bytes:
        if entry is None:
            return b""
        if entry.blob_hash is None:
            return b""
        blob_path = self._blob_path(entry.blob_hash)
        try:
            data = blob_path.read_bytes()
        except OSError as exc:
            raise SnapshotError(f"快照内容缺失：{entry.path}") from exc
        if hashlib.sha256(data).hexdigest() != entry.blob_hash:
            raise SnapshotError(f"快照内容校验失败：{entry.path}")
        return data

    def _symlink_target(self, entry: SnapshotEntryRecord | None) -> str:
        if entry is None:
            return ""
        return self._entry_bytes(entry).decode("utf-8", errors="surrogateescape")

    def _blob_path(self, blob_hash: str) -> Path:
        return self.blob_root / blob_hash[:2] / blob_hash[2:]

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace.root).as_posix()

    def _workspace_path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise SnapshotError(f"快照中包含非法路径：{relative}")
        return self.workspace.root.joinpath(*pure.parts)

    @staticmethod
    def _visible_in_diff(path: str) -> bool:
        pure = PurePosixPath(path)
        return not (
            any(part in DIFF_HIDDEN_PARTS for part in pure.parts)
            or pure.suffix in {".pyc", ".pyo"}
        )
