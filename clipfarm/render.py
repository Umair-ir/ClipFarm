"""Render a highlight as a vertical 9:16 Short with burned-in captions.

Pipeline per clip (one FFmpeg pass):
  1. OpenCV samples frames in the clip range and finds faces -> horizontal crop center.
  2. Crop 9:16 from the source, scale to 1080x1920, optional slow zoom-in (zoompan).
  3. Captions are drawn with Pillow as transparent PNGs (current word highlighted)
     and fed to FFmpeg through the concat demuxer as a single overlay stream.
     This avoids needing an FFmpeg build with libass/freetype (Homebrew's has neither).
"""
from __future__ import annotations

import json
import logging
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

CAPTION_BAND_H = 360


def _rgba(hex_color: str) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


def _caption_style(rcfg: dict) -> dict:
    return {
        "text": _rgba(rcfg.get("caption_color", "#FFFFFF")),
        "highlight": _rgba(rcfg.get("caption_highlight", "#FFDD00")),
        "stroke": _rgba(rcfg.get("caption_stroke", "#000000")),
        "y_frac": rcfg.get("caption_y", 0.66),  # top of caption band, fraction of output height
        "uppercase": rcfg.get("caption_uppercase", True),
    }


class RenderError(RuntimeError):
    pass


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(video)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise RenderError(f"No video stream in {video}")
    num, den = (v.get("avg_frame_rate") or "30/1").split("/")
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "fps": float(num) / float(den) if float(den) else 30.0,
        "duration": float(data["format"].get("duration", 0)),
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


# ---------------------------------------------------------------- face crop

FACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


def face_center_x(video: Path, start: float, end: float, samples: int = 16) -> float | None:
    """Median horizontal center (0..1) of the largest detected face across sampled frames.

    Uses OpenCV's YuNet detector (OpenCV 5 removed the old Haar cascades).
    """
    if not FACE_MODEL.exists():
        log.warning("face model missing — run scripts/install.sh; using center crop",
                    extra={"path": str(FACE_MODEL)})
        return None
    cap = cv2.VideoCapture(str(video))
    detector = None
    centers = []
    try:
        for i in range(samples):
            t = start + (end - start) * (i + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            scale = 640 / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            if detector is None:
                detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (small.shape[1], small.shape[0]),
                                                     score_threshold=0.7)
            _, faces = detector.detect(small)
            if faces is not None and len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
                centers.append((x + w / 2) / small.shape[1])
    finally:
        cap.release()
    if len(centers) < max(2, samples // 4):  # too few hits — probably false positives
        return None
    return statistics.median(centers)


# ---------------------------------------------------------------- captions

def _caption_chunks(words: list[dict], per_chunk: int) -> list[list[dict]]:
    """Group words into short captions, breaking on punctuation and pauses."""
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        pause = nxt is not None and nxt["start"] - w["end"] > 0.6
        if len(cur) >= per_chunk or w["word"].endswith((".", "!", "?", ",")) or pause or nxt is None:
            chunks.append(cur)
            cur = []
    return chunks


def _draw_caption(words: list[str], active: int, font: ImageFont.FreeTypeFont, width: int,
                  style: dict) -> Image.Image:
    img = Image.new("RGBA", (width, CAPTION_BAND_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke = max(4, font.size // 11)
    space = draw.textlength(" ", font=font)
    sizes = [draw.textlength(w, font=font) for w in words]

    # Greedy wrap into lines that fit 90% of the width.
    lines, line = [], []
    for idx, sz in enumerate(sizes):
        if line and sum(sizes[j] for j in line) + space * len(line) + sz > width * 0.9:
            lines.append(line)
            line = []
        line.append(idx)
    lines.append(line)

    line_h = font.size * 1.15
    y = (CAPTION_BAND_H - line_h * len(lines)) / 2
    for line in lines:
        total = sum(sizes[j] for j in line) + space * (len(line) - 1)
        x = (width - total) / 2
        for j in line:
            draw.text((x, y), words[j], font=font, fill=style["highlight"] if j == active else style["text"],
                      stroke_width=stroke, stroke_fill=style["stroke"])
            x += sizes[j] + space
        y += line_h
    return img


def build_caption_track(words: list[dict], clip_start: float, clip_len: float,
                        rcfg: dict, workdir: Path) -> Path | None:
    """Write caption PNGs + an ffconcat file timed to the clip. Returns the ffconcat path."""
    words = [w for w in words if w["word"] and clip_start <= w["start"] < clip_start + clip_len]
    if not words:
        return None
    font = ImageFont.truetype(rcfg["subtitle_font"], rcfg["subtitle_size"])
    style = _caption_style(rcfg)
    width = rcfg["width"]
    blank = workdir / "blank.png"
    Image.new("RGBA", (width, CAPTION_BAND_H), (0, 0, 0, 0)).save(blank)

    entries: list[tuple[Path, float]] = []
    t = 0.0
    n = 0

    def emit(path: Path, until: float):
        nonlocal t
        until = min(until, clip_len)
        if until - t > 0.005:
            entries.append((path, until - t))
            t = until

    chunks = _caption_chunks(words, rcfg["words_per_caption"])
    for ci, chunk in enumerate(chunks):
        texts = [w["word"].upper() if style["uppercase"] else w["word"] for w in chunk]
        chunk_end = chunk[-1]["end"] - clip_start
        next_start = chunks[ci + 1][0]["start"] - clip_start if ci + 1 < len(chunks) else clip_len
        emit(blank, chunk[0]["start"] - clip_start)
        for wi, w in enumerate(chunk):
            png = workdir / f"cap_{n:05d}.png"
            n += 1
            _draw_caption(texts, wi, font, width, style).save(png)
            # Each word stays highlighted until the next word starts; the last one
            # lingers briefly (or until the next caption) so text doesn't flicker.
            until = (chunk[wi + 1]["start"] - clip_start if wi + 1 < len(chunk)
                     else min(next_start, chunk_end + 0.4))
            emit(png, until)
    emit(blank, clip_len)

    concat = workdir / "captions.ffconcat"
    body = ["ffconcat version 1.0"]
    for path, dur in entries:
        body += [f"file '{path}'", f"duration {dur:.3f}"]
    body.append(f"file '{entries[-1][0]}'")  # concat demuxer quirk: repeat last file
    concat.write_text("\n".join(body) + "\n")
    return concat


# ---------------------------------------------------------------- render

def crop_window(info: dict, cx: float | None, W: int, H: int) -> tuple[int, int, int, int]:
    """W:H crop (w, h, x, y) in source pixels, horizontally centered on cx (0..1) or the frame center."""
    src_w, src_h = info["width"], info["height"]
    crop_h = src_h
    crop_w = min(src_w, int(round(src_h * W / H / 2)) * 2)
    if crop_w == src_w:  # source already narrower than W:H (e.g. vertical) — crop height instead
        crop_h = min(src_h, int(round(src_w * H / W / 2)) * 2)
    cx_px = (cx if cx is not None else 0.5) * src_w
    crop_x = int(max(0, min(src_w - crop_w, cx_px - crop_w / 2)))
    return crop_w, crop_h, crop_x, (src_h - crop_h) // 2


def render_clip(video: Path, clip: dict, transcript: dict, out_path: Path, cfg: dict) -> Path:
    rcfg = cfg["render"]
    info = probe(video)
    start, end = clip["start"], min(clip["end"], info["duration"] or clip["end"])
    dur = end - start
    W, H = rcfg["width"], rcfg["height"]

    cx = face_center_x(video, start, end) if rcfg["face_tracking"] else None
    crop_w, crop_h, crop_x, crop_y = crop_window(info, cx, W, H)
    log.info("crop", extra={"face_found": cx is not None, "crop": f"{crop_w}x{crop_h}+{crop_x}+{crop_y}"})

    vf = f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={W}:{H}:flags=lanczos,setsar=1"
    zoom = float(rcfg.get("zoom", 0) or 0)
    if zoom > 0:
        fps = round(info["fps"]) or 30
        frames = max(1, int(dur * fps))
        vf += (f",zoompan=z='1+{zoom}*on/{frames}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2'"
               f":d=1:s={W}x{H}:fps={fps}")

    workdir = Path(tempfile.mkdtemp(prefix="clipfarm_"))
    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video)]
        words = [w for s in transcript["segments"] for w in s.get("words", [])]
        concat = build_caption_track(words, start, dur, rcfg, workdir) if rcfg["subtitles"] else None
        if concat:
            cmd += ["-f", "concat", "-safe", "0", "-i", str(concat)]
            cap_y = int(H * _caption_style(rcfg)["y_frac"])
            graph = f"{vf}[v];[1:v]format=rgba[c];[v][c]overlay=0:{cap_y}:eof_action=pass[out]"
        else:
            graph = f"{vf}[out]"

        cmd += ["-filter_complex", graph, "-map", "[out]"]
        if info["has_audio"]:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", rcfg["audio_bitrate"], "-ar", "48000"]
        cmd += ["-c:v", "libx264", "-preset", rcfg["preset"], "-crf", str(rcfg["crf"]),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-t", f"{dur:.3f}", str(out_path)]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RenderError(f"ffmpeg failed: {proc.stderr.strip()[-1500:]}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log.info("rendered", extra={"out": out_path.name, "dur_s": round(dur, 1)})
    return out_path
