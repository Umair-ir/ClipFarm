"""Local licensed-music library: sidecar metadata, cached beat analysis, mood-based selection.

Every track in music/ needs a sidecar <track>.yaml next to it:

    title: Neon Run
    artist: Some Artist
    license: CC-BY-4.0            # or: royalty-free, owned, CC0, ...
    source: https://...           # where you got it / proof of license
    attribution: "Music: Neon Run by Some Artist (CC BY 4.0)"   # optional, added to the description
    moods: [hype, energetic]

Tracks without a sidecar or a license field are never used. Add one with
`python -m clipfarm music add FILE --license ... --source ... --moods ...`.
Nothing here downloads music.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import yaml

from ..store import file_hash
from . import beats

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


class Track(dict):
    @property
    def path(self) -> Path:
        return Path(self["path"])


def _sidecar(audio: Path) -> Path:
    return audio.with_suffix(".yaml")


def library(music_dir: Path, require_license: bool = True) -> list[Track]:
    tracks = []
    for audio in sorted(p for p in music_dir.glob("*") if p.suffix.lower() in AUDIO_EXTS):
        side = _sidecar(audio)
        meta = yaml.safe_load(side.read_text()) if side.exists() else None
        if require_license and not (meta and meta.get("license")):
            log.warning("skipping track without license sidecar", extra={"track": audio.name})
            continue
        meta = meta or {}
        meta.setdefault("title", audio.stem)
        meta["moods"] = [m.lower() for m in meta.get("moods", [])]
        tracks.append(Track(path=str(audio), **meta))
    return tracks


def analysis(track: Track, cache_dir: Path, engine: str = "numpy") -> dict:
    """Beat analysis, cached per file and engine ("numpy" built in, or "librosa")."""
    suffix = "" if engine == "numpy" else f".{engine}"
    cache = cache_dir / "music" / f"{file_hash(track.path)}{suffix}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    log.info("analyzing track", extra={"track": track.path.name, "engine": engine})
    a = beats.analyze_librosa(track.path) if engine == "librosa" else beats.analyze(track.path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(a))
    log.info("track analyzed", extra={"track": track.path.name, "bpm": a["bpm"], "beats": len(a["beats"]),
                                      "duration_s": round(a["duration"])})
    return a


def select(tracks: list[Track], preset: dict, cache_dir: Path, need_s: float, seed: str = "",
           engine: str = "numpy") -> Track | None:
    """Best track for a preset: mood overlap first, then BPM in range, then length. Deterministic per seed."""
    lo, hi = preset["bpm_range"]
    wanted = {m.lower() for m in preset["music_moods"]}
    best, best_score = None, -1e9
    for t in tracks:
        a = analysis(t, cache_dir, engine)
        score = 3.0 * len(wanted & set(t["moods"]))
        score += 1.0 if lo <= a["bpm"] <= hi else -1.0
        score += 0.5 if a["duration"] >= need_s else -2.0
        # Stable tie-break that varies per clip, so equal tracks rotate across edits.
        score += int(hashlib.md5(f"{seed}|{t.path.name}".encode()).hexdigest(), 16) % 100 / 1000
        if score > best_score:
            best, best_score = t, score
    return best


def add(audio: Path, music_dir: Path, license: str, source: str, moods: list[str],
        title: str | None = None, artist: str | None = None, attribution: str | None = None) -> Path:
    """Copy a track into the library and write its license sidecar."""
    if audio.suffix.lower() not in AUDIO_EXTS:
        raise ValueError(f"Unsupported audio type: {audio.suffix}")
    music_dir.mkdir(parents=True, exist_ok=True)
    dest = music_dir / audio.name
    if audio.resolve() != dest.resolve():
        dest.write_bytes(audio.read_bytes())
    meta = {"title": title or audio.stem, "artist": artist, "license": license, "source": source,
            "attribution": attribution, "moods": moods}
    _sidecar(dest).write_text(yaml.safe_dump({k: v for k, v in meta.items() if v}, sort_keys=False))
    return dest
