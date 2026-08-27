from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from .config import Settings


class LLMRequestError(RuntimeError):
    """A user-facing error raised when a model request cannot be completed."""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=1,
            timeout=60.0,
        )

    def chat(self, prompt: str) -> str:
        """Send one user message and return the assistant's text response."""
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        except AuthenticationError as exc:
            raise LLMRequestError("DeepSeek 鉴权失败，请检查 DEEPSEEK_API_KEY。") from exc
        except RateLimitError as exc:
            raise LLMRequestError("DeepSeek 请求受限，请稍后重试或检查账户额度。") from exc
        except APIConnectionError as exc:
            raise LLMRequestError("无法连接 DeepSeek API，请检查网络和 API 地址。") from exc
        except APIStatusError as exc:
            raise LLMRequestError(
                f"DeepSeek API 返回错误状态码 {exc.status_code}。"
            ) from exc
        except OpenAIError as exc:
            raise LLMRequestError("DeepSeek 请求失败，请检查配置后重试。") from exc

        if not response.choices:
            raise LLMRequestError("DeepSeek API 没有返回候选结果。")

        content = response.choices[0].message.content
        if not content:
            raise LLMRequestError("DeepSeek API 返回了空文本。")
        return content
