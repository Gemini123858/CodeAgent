from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from .context import ConversationContext
from .tools.definitions import tool_definitions


ToolDefinition = dict[str, Any]
ToolDefinitionsFactory = Callable[[], list[ToolDefinition]]


class EmbeddingProvider(Protocol):
    """Future adapter for a remote or local embedding model."""

    @property
    def model_name(self) -> str:
        """Return a stable model identifier used for cache invalidation."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector for each input text."""


class ToolSelector(Protocol):
    def select(
        self,
        *,
        user_input: str,
        messages: list[dict[str, Any]],
        available_tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Select the tools that should be exposed for one model request."""


class AllToolsSelector:
    """Current strategy: expose every registered tool."""

    def select(
        self,
        *,
        user_input: str,
        messages: list[dict[str, Any]],
        available_tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        del user_input, messages
        return deepcopy(available_tools)


class EmbeddingToolSelector:
    """Optional selector ready for a future EmbeddingProvider implementation."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        top_k: int = 6,
        minimum_similarity: float = 0.2,
        mandatory_tools: Sequence[str] = ("list_files", "read_file"),
        fallback_to_all: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")
        self.provider = provider
        self.top_k = top_k
        self.minimum_similarity = minimum_similarity
        self.mandatory_tools = frozenset(mandatory_tools)
        self.fallback_to_all = fallback_to_all
        self._embedding_cache: dict[str, list[float]] = {}

    def select(
        self,
        *,
        user_input: str,
        messages: list[dict[str, Any]],
        available_tools: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        del messages
        if not available_tools:
            return []

        tool_vectors = self._tool_vectors(available_tools)
        query_vectors = self.provider.embed([user_input])
        if len(query_vectors) != 1:
            raise ValueError("EmbeddingProvider 返回的查询向量数量不正确。")

        scored = [
            (self._cosine_similarity(query_vectors[0], vector), index)
            for index, vector in enumerate(tool_vectors)
        ]
        scored.sort(reverse=True)
        selected_indexes = {
            index
            for score, index in scored[: self.top_k]
            if score >= self.minimum_similarity
        }
        for index, definition in enumerate(available_tools):
            if self._tool_name(definition) in self.mandatory_tools:
                selected_indexes.add(index)

        if not selected_indexes and self.fallback_to_all:
            return deepcopy(available_tools)
        return [
            deepcopy(definition)
            for index, definition in enumerate(available_tools)
            if index in selected_indexes
        ]

    def _tool_vectors(
        self,
        available_tools: list[ToolDefinition],
    ) -> list[list[float]]:
        keys = [self._cache_key(definition) for definition in available_tools]
        missing_indexes = [
            index for index, key in enumerate(keys) if key not in self._embedding_cache
        ]
        if missing_indexes:
            texts = [
                self._embedding_text(available_tools[index])
                for index in missing_indexes
            ]
            vectors = self.provider.embed(texts)
            if len(vectors) != len(missing_indexes):
                raise ValueError("EmbeddingProvider 返回的工具向量数量不正确。")
            for index, vector in zip(missing_indexes, vectors, strict=True):
                self._embedding_cache[keys[index]] = vector
        return [self._embedding_cache[key] for key in keys]

    def _cache_key(self, definition: ToolDefinition) -> str:
        schema = json.dumps(definition, ensure_ascii=False, sort_keys=True)
        return f"{self.provider.model_name}:{schema}"

    @staticmethod
    def _embedding_text(definition: ToolDefinition) -> str:
        function = definition.get("function", {})
        return json.dumps(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _tool_name(definition: ToolDefinition) -> str:
        return str(definition.get("function", {}).get("name", ""))

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            raise ValueError("Embedding 向量维度不一致。")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            left_norm * right_norm
        )


@dataclass(frozen=True)
class BuiltRequest:
    messages: list[dict[str, Any]]
    tools: list[ToolDefinition]
    tool_choice: str = "auto"


class RequestBuilder:
    """Construct a fresh messages/tools request for every model step."""

    def __init__(
        self,
        system_prompt: str,
        *,
        tool_selector: ToolSelector | None = None,
        definitions_factory: ToolDefinitionsFactory = tool_definitions,
    ) -> None:
        self.system_prompt = system_prompt
        self.tool_selector = tool_selector or AllToolsSelector()
        self.definitions_factory = definitions_factory

    def build(
        self,
        context: ConversationContext,
        *,
        current_user_input: str,
    ) -> BuiltRequest:
        history = context.messages_for_request()
        available_tools = self.definitions_factory()
        selected_tools = self.tool_selector.select(
            user_input=current_user_input,
            messages=history,
            available_tools=available_tools,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
        ]
        return BuiltRequest(
            messages=messages,
            tools=selected_tools,
            tool_choice="auto" if selected_tools else "none",
        )
