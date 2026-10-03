"""A scripted stand-in for an LLM so SOP behavior can be tested deterministically."""

import json
from collections import deque

from backend.llm.provider import LLMResult, Message


class ScriptedProvider:
    """Extractor calls return the next scripted extraction; responder calls return
    `reply` (or an error by default, which makes the engine use its fallback text)."""

    name = "scripted"
    model = "scripted"

    def __init__(self, extractions: list[dict], reply: str | None = None):
        self.extractions = deque(extractions)
        self.reply = reply
        self.responder_systems: list[str] = []

    def complete(self, system: str, messages: list[Message], max_tokens: int = 1024) -> LLMResult:
        if system.startswith("You extract"):
            data = self.extractions.popleft() if self.extractions else {}
            return LLMResult(text=f"<json>{json.dumps(data)}</json>", model=self.model, latency_ms=0)
        self.responder_systems.append(system)
        if self.reply is None:
            return LLMResult(text="", model=self.model, latency_ms=0, error="scripted: use fallback")
        return LLMResult(text=self.reply, model=self.model, latency_ms=0)
