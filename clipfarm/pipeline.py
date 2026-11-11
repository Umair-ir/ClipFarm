"""End-to-end processing of a queued VOD: transcribe -> highlights -> render + metadata."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .highlights import find_highlights
from .render import render_clip
from .store import Queue
from .transcribe import transcribe

log = logging.getLogger(__name__)


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "clip"


def process_vod(q: Queue, vod_id: int, cfg: dict, max_clips: int | None = None) -> list[Path]:
    vod = q.get(vod_id)
    video = Path(vod["path"])
    ctx = {"vod_id": vod_id, "streamer": vod["streamer"]}
    try:
        transcript = transcribe(video, vod["file_hash"], cfg)
        q.set_status(vod_id, "transcribed")

        clips = find_highlights(transcript, vod["file_hash"], cfg)
        q.set_status(vod_id, "analyzed")
        if not clips:
            log.warning("no highlights found", extra=ctx)

        out_dir = cfg["paths"]["clips"] / vod["streamer"] / vod["stream_date"]
        outputs = []
        for rank, clip in enumerate(clips[:max_clips] if max_clips else clips, 1):
            name = f"{video.stem}_{rank:02d}_{_slug(clip['title'])}"
            out = render_clip(video, clip, transcript, out_dir / f"{name}.mp4", cfg)
            meta = {
                "title": clip["title"],
                "description": f"{clip['description']}\n\n{' '.join(clip['hashtags'])}",
                "hashtags": clip["hashtags"],
                "category": clip["category"],
                "score": clip["score"],
                "reason": clip["reason"],
                "source": {"vod_id": vod_id, "path": str(video), "streamer": vod["streamer"],
                           "start": clip["start"], "end": clip["end"]},
                "transcript": clip["text"],
            }
            (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
            q.add_clip(vod_id, out, clip["start"], clip["end"], clip["score"], clip["title"])
            outputs.append(out)

        q.set_status(vod_id, "rendered")
        log.info("vod done", extra={**ctx, "clips": len(outputs)})
        return outputs
    except Exception as exc:
        q.set_status(vod_id, "failed", f"{type(exc).__name__}: {exc}")
        log.exception("vod failed", extra=ctx)
        raise
