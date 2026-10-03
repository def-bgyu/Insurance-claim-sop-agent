"""Model providers behind one tiny interface: system prompt + messages in, text out.

The harness never relies on provider-specific features (native tool calling,
structured-output modes), so any chat model can be plugged in. Adding a provider
means implementing `complete()`.
"""

import time
from dataclasses import dataclass, field
from typing import Protocol

import anthropic

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None  # set when the call failed; text is then ""


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self, system: str, messages: list[Message], max_tokens: int = 1024
    ) -> LLMResult: ...


@dataclass
class AnthropicProvider:
    api_key: str | None = None  # None -> SDK resolves ANTHROPIC_API_KEY etc.
    model: str = DEFAULT_ANTHROPIC_MODEL
    name: str = "anthropic"
    _client: anthropic.Anthropic = field(init=False, repr=False)

    def __post_init__(self):
        self._client = anthropic.Anthropic(api_key=self.api_key, timeout=30.0, max_retries=2)

    def complete(self, system: str, messages: list[Message], max_tokens: int = 1024) -> LLMResult:
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except anthropic.AuthenticationError:
            return self._failed(started, "authentication_error: invalid API key")
        except anthropic.RateLimitError:
            return self._failed(started, "rate_limited")
        except anthropic.APIStatusError as e:
            return self._failed(started, f"api_error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            return self._failed(started, "connection_error")

        # Only text blocks are used; thinking or other block types are ignored.
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResult(
            text=text,
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _failed(self, started: float, error: str) -> LLMResult:
        return LLMResult(
            text="",
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=error,
        )
