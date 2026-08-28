from __future__ import annotations

from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from .config import Settings
from .debug import DebugPrinter


class LLMRequestError(RuntimeError):
    """A user-facing error raised when a model request cannot be completed."""


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        debug: DebugPrinter | None = None,
    ) -> None:
        self.settings = settings
        self.debug = debug or DebugPrinter()
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=1,
            timeout=60.0,
        )

    def chat(self, prompt: str) -> str:
        """Send one user message and return the assistant's text response."""
        response = self.complete(
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        if not content:
            raise LLMRequestError("DeepSeek API 返回了空文本。")
        return content

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        """Create one non-streaming, non-thinking Chat Completion."""
        request: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        try:
            response = self.client.chat.completions.create(**request)
        except AuthenticationError as exc:
            self._debug_api_error(exc)
            raise LLMRequestError("DeepSeek 鉴权失败，请检查 DEEPSEEK_API_KEY。") from exc
        except RateLimitError as exc:
            self._debug_api_error(exc)
            raise LLMRequestError("DeepSeek 请求受限，请稍后重试或检查账户额度。") from exc
        except APIConnectionError as exc:
            self.debug.log("api.error", {"type": type(exc).__name__})
            raise LLMRequestError("无法连接 DeepSeek API，请检查网络和 API 地址。") from exc
        except APIStatusError as exc:
            self._debug_api_error(exc)
            raise LLMRequestError(
                f"DeepSeek API 返回错误状态码 {exc.status_code}。"
            ) from exc
        except OpenAIError as exc:
            self.debug.log("api.error", {"type": type(exc).__name__})
            raise LLMRequestError("DeepSeek 请求失败，请检查配置后重试。") from exc

        if not response.choices:
            raise LLMRequestError("DeepSeek API 没有返回候选结果。")
        return response

    def _debug_api_error(self, exc: APIStatusError) -> None:
        response = getattr(exc, "response", None)
        details: dict[str, Any] = {
            "type": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
            "request_id": getattr(exc, "request_id", None),
        }
        if response is not None:
            try:
                details["body"] = response.json()
            except Exception:
                details["body"] = getattr(response, "text", "<unavailable>")
        self.debug.log("api.error", details)
