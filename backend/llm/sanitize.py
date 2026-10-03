"""Clean raw model output before it reaches parsing, the caller, or the trace.

Reasoning models (DeepSeek-R1, Qwen3, ...) emit <think> blocks inline; some small
models leak chat-template tokens. None of that should ever be shown to a caller.
"""

import re

_REASONING_BLOCKS = re.compile(
    r"<(think|thinking|reasoning|reflection)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
# An opening tag with no closing tag: the model was cut off mid-thought.
_UNCLOSED_REASONING = re.compile(r"<(think|thinking|reasoning)>.*\Z", re.DOTALL | re.IGNORECASE)
_TEMPLATE_TOKENS = re.compile(r"<\|[^|>]{1,40}\|>")

MAX_REPLY_CHARS = 1500


def strip_reasoning(text: str) -> str:
    text = _REASONING_BLOCKS.sub("", text)
    text = _UNCLOSED_REASONING.sub("", text)
    text = _TEMPLATE_TOKENS.sub("", text)
    return text.strip()


def clean_reply(text: str) -> str:
    """For text shown to the caller: no reasoning, no wrapper quotes/labels, bounded length."""
    text = strip_reasoning(text)
    text = re.sub(r"^(agent|assistant|alex)\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    if len(text) > MAX_REPLY_CHARS:
        cut = text[:MAX_REPLY_CHARS]
        text = cut[: cut.rfind(".") + 1] or cut  # end on a sentence when possible
    return text
