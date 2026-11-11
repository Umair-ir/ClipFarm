"""Local LLM calls through Ollama with JSON-schema structured output."""
from __future__ import annotations

import json
import logging

import requests

log = logging.getLogger(__name__)


def salvage_truncated(content: str) -> object | None:
    """Recover the complete part of JSON that was cut off mid-way (done_reason=length).

    Keeps everything up to the last value that closed inside a container, then closes the
    still-open brackets: '{"clips": [{...}, {...}, {"start_li' -> '{"clips": [{...}, {...}]}'.
    """
    stack: list[str] = []
    in_str = esc = False
    last: tuple[int, list[str]] | None = None
    for i, ch in enumerate(content):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if stack:
                last = (i, list(stack))
    if last is None:
        return None
    i, open_ = last
    closing = "".join("}" if c == "{" else "]" for c in reversed(open_))
    try:
        return json.loads(content[: i + 1] + closing)
    except json.JSONDecodeError:
        return None


def conform(data: object, schema: dict) -> object | None:
    """Minimal schema check: required keys present; array items that don't conform are dropped.

    Lets callers index required fields safely, including on salvaged (partial) output.
    """
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(data, dict) or any(k not in data for k in schema.get("required", [])):
            return None
        out = dict(data)
        for key, sub in schema.get("properties", {}).items():
            if key in out:
                fixed = conform(out[key], sub)
                if fixed is None:
                    if key in schema.get("required", []):
                        return None
                    out.pop(key)
                else:
                    out[key] = fixed
        return out
    if kind == "array":
        if not isinstance(data, list):
            return None
        items = schema.get("items")
        return [c for c in (conform(x, items) if items else x for x in data) if c is not None]
    return data


def chat_json(system: str, user: str, schema: dict, ocfg: dict) -> dict | None:
    # Structured output can occasionally run away (endless whitespace/repetition); num_predict
    # caps it, a truncated/invalid answer gets one deterministic retry, and if that fails too the
    # complete part of the cut-off answer is kept rather than discarding everything.
    salvaged = None
    for attempt, temperature in enumerate((ocfg["temperature"], 0.0)):
        resp = requests.post(
            f"{ocfg['host']}/api/chat",
            json={
                "model": ocfg["model"],
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "format": schema,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": ocfg["num_ctx"],
                            "num_predict": ocfg["num_predict"]},
            },
            timeout=ocfg["timeout_s"],
        )
        resp.raise_for_status()
        body = resp.json()
        content = body["message"]["content"]
        try:
            result = conform(json.loads(content), schema)
            if result is not None:
                return result
            log.warning("LLM JSON missing required fields", extra={"attempt": attempt + 1})
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON", extra={"attempt": attempt + 1,
                                                             "done_reason": body.get("done_reason"),
                                                             "content": content[:200]})
            partial = salvage_truncated(content)
            partial = conform(partial, schema) if partial is not None else None
            if partial is not None and salvaged is None:
                salvaged = partial
    if salvaged is not None:
        log.info("using the complete part of a truncated LLM answer")
    return salvaged
