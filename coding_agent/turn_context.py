from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnContext:
    """Mutable execution state whose lifetime is exactly one user turn."""

    turn_id: int | None = None
    _created_files: set[str] = field(default_factory=set)
    _approval_decisions: dict[str, bool] = field(default_factory=dict)

    def record_created_file(self, path: str) -> None:
        self._created_files.add(path)

    def record_deleted_file(self, path: str) -> None:
        self._created_files.discard(path)

    def can_delete_without_approval(self, path: str) -> bool:
        return path in self._created_files

    def approval_decision(self, fingerprint: str) -> bool | None:
        return self._approval_decisions.get(fingerprint)

    def remember_approval(self, fingerprint: str, approved: bool) -> None:
        self._approval_decisions[fingerprint] = approved


