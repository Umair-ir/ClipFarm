"""Pick extra moments from the same VOD to feed montage shots.

Sources, in priority order:
  1. the VOD's other AI-selected highlights (their loudest point),
  2. loudness peaks across the whole VOD (shouting, laughter, hype).
The loudness curve is cached per file hash.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SR = 8000
WIN_S = 0.5


def loudness_curve(video: Path, file_hash: str, cache_dir: Path) -> np.ndarray:
    """RMS loudness (dB) per 0.5 s window."""
    cache = cache_dir / "energy" / f"{file_hash}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    win = int(SR * WIN_S)
    n = len(y) // win
    rms = np.sqrt((y[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([round(float(v), 2) for v in db]))
    return db


def peak_in(db: np.ndarray, start: float, end: float) -> float:
    a, b = int(start / WIN_S), max(int(start / WIN_S) + 1, int(end / WIN_S))
    seg = db[a:b]
    return start if not len(seg) else (a + int(np.argmax(seg))) * WIN_S


def loud_peaks(db: np.ndarray, exclude: list[tuple[float, float]], min_gap_s: float = 8.0,
               limit: int = 30) -> list[float]:
    """Times of the loudest moments (smoothed), at least min_gap_s apart, outside `exclude`."""
    if len(db) < 4:
        return []
    smooth = np.convolve(db, np.ones(3) / 3, mode="same")
    order = np.argsort(smooth)[::-1]
    picked: list[float] = []
    for i in order:
        t = i * WIN_S
        if any(s - 1 <= t <= e + 1 for s, e in exclude):
            continue
        if smooth[i] < np.percentile(smooth, 60):  # only genuinely loud windows
            break
        if all(abs(t - p) >= min_gap_s for p in picked):
            picked.append(t)
        if len(picked) >= limit:
            break
    return picked


def montage_moments(video: Path, file_hash: str, cache_dir: Path, main: tuple[float, float],
                    other_highlights: list[tuple[float, float]]) -> list[float]:
    """Anchor times for outro shots, best first. Falls back to the main clip if the VOD has nothing else."""
    db = loudness_curve(video, file_hash, cache_dir)
    anchors = [peak_in(db, s, e) for s, e in other_highlights if not (s < main[1] and e > main[0])]
    for t in loud_peaks(db, exclude=[main] + other_highlights):
        if all(abs(t - a) >= 4 for a in anchors):
            anchors.append(t)
    if not anchors:
        log.info("no other moments in VOD — montage reuses the main clip")
        anchors = intro_moments(db, main, 6)
    return anchors


def intro_moments(db: np.ndarray, main: tuple[float, float], n: int) -> list[float]:
    """Loudest points inside the main clip, for teaser shots."""
    s, e = main
    a, b = int(s / WIN_S), max(int(s / WIN_S) + 1, int(e / WIN_S))
    seg = db[a:b]
    if not len(seg):
        return [s]
    idx = np.argsort(seg)[::-1]
    picked: list[float] = []
    for i in idx:
        t = (a + int(i)) * WIN_S
        if all(abs(t - p) >= 2.0 for p in picked):
            picked.append(t)
        if len(picked) >= n:
            break
    return picked
