from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


ChatMessage = dict[str, Any]


class ConversationContext(ABC):
    """Pluggable strategy for storing and preparing conversation context."""

    @abstractmethod
    def append(self, message: ChatMessage) -> None:
        """Record one canonical chat message."""

    def extend(self, messages: list[ChatMessage]) -> None:
        for message in messages:
            self.append(message)

    @abstractmethod
    def messages_for_request(self) -> list[ChatMessage]:
        """Build messages for the next request.

        Future truncation or summary strategies must preserve the system/task
        instructions and keep each assistant tool call together with all of
        its corresponding tool result messages.
        """

    @property
    @abstractmethod
    def message_count(self) -> int:
        """Return the number of messages currently retained."""

    @property
    def strategy_name(self) -> str:
        return type(self).__name__


class FullHistoryContext(ConversationContext):
    """Current strategy: retain and resend every message without compression."""

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []

    def append(self, message: ChatMessage) -> None:
        if not isinstance(message, dict):
            raise TypeError("上下文消息必须是字典。")
        if "role" not in message:
            raise ValueError("上下文消息缺少 role。")
        self._messages.append(deepcopy(message))

    def messages_for_request(self) -> list[ChatMessage]:
        return deepcopy(self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)
