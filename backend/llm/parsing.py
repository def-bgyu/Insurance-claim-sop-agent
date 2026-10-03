"""Multi-layer parser: get a JSON object out of whatever the model returned.

Small models wrap JSON in prose, code fences, or XML; use single quotes; leave
trailing commas; or skip JSON entirely and emit one tag per field. Each layer
below handles one of those habits. The layer that succeeded is recorded in the
trace, which shows how much each model relies on the fallbacks.
"""

import json
import re
from dataclasses import dataclass

from backend.llm.sanitize import strip_reasoning

@dataclass
class ParseResult:
    data: dict | None
    layer: str  # which layer produced `data`, or "failed"


def _loads_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _outer_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else None


def _repair(text: str) -> str:
    """Common near-JSON mistakes: Python literals, single quotes, trailing commas."""
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    if '"' not in text:
        text = text.replace("'", '"')
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"//[^\n]*", "", text)  # comments
    return text


def _xml_fields(text: str, field_names: list[str]) -> dict | None:
    """Last resort: <full_name>Margaret Chen</full_name> style output, flat fields only."""
    found = {}
    for name in field_names:
        match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if value.lower() in {"", "null", "none"}:
                continue
            found[name] = {"true": True, "false": False}.get(value.lower(), value)
    return found or None


def parse_json_object(raw: str, tag: str = "json", field_names: list[str] = ()) -> ParseResult:
    text = strip_reasoning(raw)

    if (data := _loads_object(text)) is not None:
        return ParseResult(data, "json")

    tagged = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if tagged and (data := _loads_object(tagged.group(1).strip())) is not None:
        return ParseResult(data, "xml_tag")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced and (data := _loads_object(fenced.group(1).strip())) is not None:
        return ParseResult(data, "code_fence")

    braces = _outer_braces(text)
    if braces and (data := _loads_object(braces)) is not None:
        return ParseResult(data, "braces")

    if braces and (data := _loads_object(_repair(braces))) is not None:
        return ParseResult(data, "repaired")

    if field_names and (data := _xml_fields(text, list(field_names))) is not None:
        return ParseResult(data, "xml_fields")

    return ParseResult(None, "failed")
