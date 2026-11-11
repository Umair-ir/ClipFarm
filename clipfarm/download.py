"""Download VODs from URLs (YouTube, Twitch, …) with yt-dlp.

Only use this for content you own or have the creator's permission to clip.
Downloads land in data/inbox/ and are then ingested like any local file.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def is_url(text: str) -> bool:
    return text.startswith(("http://", "https://"))


def download(url: str, dest_dir: Path, max_height: int = 1080) -> tuple[Path, dict]:
    """Download `url` as MP4 (≤ max_height). Returns (path, info) where info has
    title, channel, upload date, duration and license."""
    from yt_dlp import YoutubeDL

    last_pct = [-10]

    def progress(d: dict) -> None:
        # Log every 10% (yt-dlp's own bar is noisy in log files).
        if d["status"] != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if not total:
            return
        pct = int(d["downloaded_bytes"] * 100 / total)
        if pct >= last_pct[0] + 10:
            last_pct[0] = pct - pct % 10
            eta = d.get("eta")
            log.info("downloading", extra={"pct": pct, "mb": int(d["downloaded_bytes"] / 2**20),
                                           "of_mb": int(total / 2**20),
                                           "eta_min": round(eta / 60) if eta else None})

    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "%(extractor)s_%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "progress_hooks": [progress],
        # Twitch/HLS: retry failed pieces and never silently skip them (a skipped piece = a shorter VOD).
        "retries": 10,
        "fragment_retries": 20,
        "skip_unavailable_fragments": False,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        raise FileNotFoundError(f"yt-dlp finished but {path} is missing")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        log.warning("this stream was still live or processing — the VOD may be incomplete; "
                    "re-run after the stream ends for the full video", extra={"url": url})
    expected = info.get("duration")
    if expected:
        from .render import probe
        got = probe(path)["duration"]
        if got < 0.95 * expected:
            log.warning("downloaded video is shorter than reported", extra={
                "got_min": round(got / 60, 1), "expected_min": round(expected / 60, 1)})

    upload = info.get("upload_date")
    meta = {
        "url": info.get("webpage_url", url),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": datetime.strptime(upload, "%Y%m%d").date() if upload else None,
        "duration": info.get("duration"),
        "license": info.get("license"),
    }
    log.info("downloaded", extra={"path": path.name, "title": meta["title"], "channel": meta["channel"],
                                  "license": meta["license"]})
    return path, meta
