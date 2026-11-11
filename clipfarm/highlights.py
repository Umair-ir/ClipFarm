"""Find clip-worthy moments in a transcript using a local Ollama model.

The transcript is split into overlapping windows. Each window is sent to the
LLM as numbered lines; the model answers with *line ranges* (not raw seconds),
which small models get right far more often. Ranges are then mapped back to
Whisper segment boundaries, so clips always start and end on full sentences,
and stretched/trimmed into the configured 20–60 s range.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .llm import chat_json

log = logging.getLogger(__name__)

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """You are an expert YouTube Shorts editor for livestream content.
You read numbered transcript lines from a stream and pick the moments most likely to go viral as a 20-60 second Short:
funny moments, arguments, unexpected reactions, controversial or hot takes, and genuinely interesting conversations.

Rules:
- Each clip must make sense on its own to a viewer with no context. Include the setup, not just the punchline.
- Start on the line where the moment begins (include the setup) and end on the line where it resolves (the punchline or the reaction to it). Never cut mid-thought.
- Stop when the topic changes. One clip = one story, joke or argument; never merge two moments.
- Duration = end time of end_line minus start time of start_line, using the [start-end] seconds shown. It must be between {min_s} and {max_s} seconds.
- start_line and end_line must be line numbers that appear in the excerpt.
- Prefer fewer, stronger clips. Skip filler, greetings, reading donations, and dead air.
- score is 1-10: how likely a random viewer keeps watching and shares it. Be strict; 8+ is rare.
- If nothing in the excerpt is worth clipping, return an empty list."""

CATEGORIES = ["funny", "argument", "reaction", "controversial", "conversation", "other"]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["start_line", "end_line", "category", "score", "title",
                             "description", "hashtags", "reason"],
            },
        }
    },
    "required": ["clips"],
}

_TERMINAL = (".", "!", "?", "…", '"')


def _windows(segments: list[dict], window_s: float, overlap_s: float):
    """Yield (first_index, last_index_exclusive) over segments in overlapping time windows."""
    if not segments:
        return
    start_t = segments[0]["start"]
    end_t = segments[-1]["end"]
    while start_t < end_t:
        idx = [i for i, s in enumerate(segments) if start_t <= s["start"] < start_t + window_s]
        if idx:
            yield idx[0], idx[-1] + 1
        start_t += window_s - overlap_s


REFINE_PROMPT = """You are trimming a clip for a YouTube Shorts editor.
Below are numbered transcript lines around a moment titled "{title}" ({category}).
The first guess was lines {a} to {b}.

Choose the exact start_line and end_line for the complete moment:
- start_line: where the setup begins, so a new viewer understands what is happening.
- end_line: the LAST line that still belongs to this moment — include the punchline AND the reactions to it
  (laughing, disbelief, follow-up jokes). End right before the speaker moves on to an unrelated topic.
- Duration (end time of end_line minus start time of start_line) must be between {min_s} and {max_s} seconds."""

BOUNDS_SCHEMA = {
    "type": "object",
    "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
    "required": ["start_line", "end_line"],
}


def _format_lines(segments: list[dict], lo: int, hi: int) -> str:
    return "\n".join(f"{i} [{segments[i]['start']:.0f}-{segments[i]['end']:.0f}] {segments[i]['text']}"
                     for i in range(lo, hi))


def _ask_ollama(lines: str, ocfg: dict, max_clips: int, min_s: int, max_s: int) -> list[dict]:
    system = SYSTEM_PROMPT.format(min_s=min_s, max_s=max_s)
    user = (f"Transcript excerpt (line number, [start-end seconds], text):\n\n{lines}\n\n"
            f"Return up to {max_clips} clips as JSON. Titles: punchy, under 70 characters, no clickbait lies. "
            "description: 1-2 sentences. hashtags: 3-6, each starting with #, always include #shorts.")
    result = chat_json(system, user, RESPONSE_SCHEMA, ocfg)
    return result.get("clips", []) if result else []


def _refine_bounds(segments: list[dict], a: int, b: int, clip: dict, ocfg: dict, hcfg: dict) -> tuple[int, int]:
    """Second, narrower LLM pass: re-pick start/end with surrounding context visible."""
    lo, hi = max(0, a - 4), min(len(segments), b + 10)
    system = REFINE_PROMPT.format(title=clip["title"], category=clip["category"], a=a, b=b,
                                  min_s=hcfg["min_clip_s"], max_s=hcfg["max_clip_s"])
    result = chat_json(system, _format_lines(segments, lo, hi), BOUNDS_SCHEMA, ocfg)
    if result:
        ra, rb = sorted((int(result["start_line"]), int(result["end_line"])))
        if lo <= ra <= rb < hi:
            return ra, rb
    return a, b


def _fit_to_bounds(segments: list[dict], a: int, b: int, hcfg: dict, duration: float) -> tuple[int, int] | None:
    """Adjust the inclusive segment range [a, b] to satisfy min/max length and end on a full sentence."""
    lo, hi = hcfg["min_clip_s"], hcfg["max_clip_s"]
    length = lambda: segments[b]["end"] - segments[a]["start"]  # noqa: E731

    # Too long: drop trailing segments, then leading ones.
    while length() > hi and b > a:
        b -= 1
    while length() > hi and a < b:
        a += 1
    if length() > hi:
        return None  # a single segment longer than max — skip rather than cut mid-sentence

    # Too short: grow forward (aftermath/reaction), then backward (setup).
    while length() < lo:
        can_fwd = b + 1 < len(segments) and segments[b + 1]["end"] - segments[a]["start"] <= hi
        can_back = a > 0 and segments[b]["end"] - segments[a - 1]["start"] <= hi
        if can_fwd:
            b += 1
        elif can_back:
            a -= 1
        else:
            break
    if length() < lo and length() < min(lo, duration):
        return None

    # Try to end on terminal punctuation if the next segment(s) finish the sentence.
    j = b
    while (not segments[j]["text"].rstrip().endswith(_TERMINAL) and j + 1 < len(segments)
           and segments[j + 1]["end"] - segments[a]["start"] <= hi):
        j += 1
    if segments[j]["text"].rstrip().endswith(_TERMINAL):
        b = j
    return a, b


def _overlap(x: dict, y: dict) -> float:
    inter = max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))
    return inter / max(1e-6, min(x["end"] - x["start"], y["end"] - y["start"]))


def find_highlights(transcript: dict, file_hash: str, cfg: dict) -> list[dict]:
    hcfg, ocfg = cfg["highlights"], cfg["ollama"]
    key = hashlib.sha1(json.dumps([PROMPT_VERSION, ocfg["model"], hcfg], sort_keys=True).encode()).hexdigest()[:8]
    cache_file = cfg["paths"]["cache"] / "highlights" / f"{file_hash}.{transcript['model']}.{key}.json"
    if cache_file.exists():
        log.info("highlight cache hit", extra={"file": cache_file.name})
        return json.loads(cache_file.read_text())

    segments = [s for s in transcript["segments"] if s["text"]]
    candidates: list[dict] = []
    for w_from, w_to in _windows(segments, hcfg["window_s"], hcfg["overlap_s"]):
        lines = _format_lines(segments, w_from, w_to)
        log.info("analyzing window", extra={"from_s": round(segments[w_from]["start"]),
                                            "to_s": round(segments[w_to - 1]["end"])})
        for c in _ask_ollama(lines, ocfg, hcfg.get("clips_per_window", 3), hcfg["min_clip_s"],
                               hcfg["max_clip_s"]):
            a, b = sorted((int(c["start_line"]), int(c["end_line"])))
            if not (w_from <= a <= b < w_to):
                log.info("dropped clip (line numbers outside excerpt)", extra={"title": c["title"], "lines": f"{a}-{b}"})
                continue
            if c["score"] < hcfg["min_score"]:
                continue
            a, b = _refine_bounds(segments, a, b, c, ocfg, hcfg)
            fitted = _fit_to_bounds(segments, a, b, hcfg, transcript["duration"])
            if not fitted:
                log.info("dropped clip (could not fit length)", extra={"title": c["title"]})
                continue
            a, b = fitted
            tags = [t if t.startswith("#") else f"#{t}" for t in c["hashtags"]]
            if "#shorts" not in (t.lower() for t in tags):
                tags.append("#shorts")
            candidates.append({
                "start": segments[a]["start"],
                "end": segments[b]["end"],
                "score": c["score"],
                "category": c["category"],
                "title": c["title"].strip()[:100],
                "description": c["description"].strip(),
                "hashtags": tags,
                "reason": c["reason"],
                "text": " ".join(s["text"] for s in segments[a:b + 1]),
            })

    # Rank and drop near-duplicates (overlapping windows often find the same moment twice).
    chosen: list[dict] = []
    for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if all(_overlap(c, k) < 0.3 for k in chosen):
            chosen.append(c)
        if len(chosen) >= hcfg["max_clips_per_vod"]:
            break

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(chosen, indent=1))
    log.info("highlights found", extra={"candidates": len(candidates), "chosen": len(chosen)})
    return chosen
