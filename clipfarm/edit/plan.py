"""Build an edit decision list (EDL): pure data, saved next to the render for inspection.

Timeline:  [intro: teaser shots from the main clip] [main clip] [outro: montage from the VOD]

Music is laid as two segments:
  A  plays under intro + main clip, starting on a beat (ideally so the drop arrives on time by itself)
  B  starts exactly on the track's drop when the outro starts
If the song's drop naturally lands at the outro, A and B are contiguous and there's no music cut;
otherwise the cut is hidden on the drop (a beat, under a white flash), as editors do by hand.
Intro cuts follow A's beats, outro cuts follow B's, every `cut_every_beats` beats.
"""
from __future__ import annotations

import hashlib
import logging
import random

from . import beats as beatlib

log = logging.getLogger(__name__)

MIN_SHOT_S = 0.25


def _cut_points(beat_times: list[float], start: float, end: float, every: int) -> list[float]:
    """Beat-aligned cut points inside [start, end], always including both ends."""
    inside = [b for b in beat_times if start + MIN_SHOT_S <= b <= end - MIN_SHOT_S][::max(1, every)]
    return [start] + inside + [end]


def _shot_effects(rng: random.Random, dur: float, p: dict, first_of_outro: bool, on_bar: bool) -> dict:
    ti, si = p["transition_intensity"], p["slowmo_intensity"]
    slow = 1.0 - 0.6 * si  # playback speed for slow-mo (intensity 1 → 0.4x)
    speed: dict = {"type": "normal"}
    if si > 0 and (first_of_outro or rng.random() < 0.6 * si):
        if p["speed_ramps"] and dur >= 0.8 and rng.random() < 0.5:
            speed = {"type": "ramp", "fast": 1.6, "slow": round(slow, 3), "at": 0.4}
        else:
            speed = {"type": "slow", "slow": round(slow, 3)}
    zoom = 1.0 + (p["punch_in"] * rng.uniform(0.5, 1.0) if rng.random() < 0.6 else 0.0)
    # White flash on the drop and on bar starts only (hard cuts in between), never > 30% of the shot.
    flash = 0.0
    if ti > 0 and (first_of_outro or on_bar):
        flash = round(min((0.04 + 0.16 * ti) * (1.6 if first_of_outro else 1.0), 0.3 * dur), 3)
    return {"speed": speed, "zoom": round(zoom, 3), "flash": flash, "blur": bool(p["motion_blur"])}


def _source_duration(dur: float, speed: dict) -> float:
    if speed["type"] == "slow":
        return dur * speed["slow"]
    if speed["type"] == "ramp":
        a = speed["at"]
        return dur * (a * speed["fast"] + (1 - a) * speed["slow"])
    return dur


def _place(anchor: float, src_dur: float, lo: float, hi: float) -> float:
    """Source start for a shot around `anchor`, kept inside [lo, hi - src_dur]."""
    start = anchor - 0.4 * src_dur
    return round(max(lo, min(start, hi - src_dur)), 3)


def build(clip: dict, preset: dict, out_cfg: dict, music: dict | None, music_analysis: dict | None,
          vod_duration: float, intro_anchors: list[float], outro_anchors: list[float]) -> dict:
    p = preset
    main_s = clip["end"] - clip["start"]
    intro_s = float(p["intro_s"])
    outro_s = float(p["outro_s"])
    over = intro_s + main_s + outro_s - out_cfg["max_total_s"]
    if over > 0:
        outro_s = max(0.0, outro_s - over)
        if outro_s < 2:
            outro_s = 0.0
        log.info("outro shortened to fit max_total_s", extra={"outro_s": round(outro_s, 2)})

    a = music_analysis or beatlib.fixed_grid(p["fallback_bpm"], intro_s + main_s + outro_s + 30)
    outro_start = intro_s + main_s

    first_beat = a["beats"][0] if a["beats"] else 0.0
    if outro_s > 0:
        drop = a["beats"][beatlib.find_drop(a, need=outro_s)]
        # Start A so the drop arrives at the outro on its own, if the song is long enough; else from the top.
        a_start = drop - outro_start if drop - outro_start >= first_beat else first_beat
    else:
        drop, a_start = None, first_beat
    if intro_s > 0 and drop is not None and drop - intro_s >= first_beat and a_start != drop - outro_start:
        # Intro plays the build-up and the drop hits as the streamer clip starts (B replays it at the outro).
        a_start = drop - intro_s
    beats_a = [round(b - a_start, 3) for b in a["beats"]]                            # timeline times, segment A
    beats_b = [round(b - drop + outro_start, 3) for b in a["beats"]] if drop is not None else beats_a

    # Snap the outro end to a cut point so the edit ends on the beat.
    if outro_s > 0:
        grid = [b for b in beats_b if b > outro_start][::max(1, p["cut_every_beats"])]
        if grid:
            end = min(grid, key=lambda b: abs(b - (outro_start + outro_s)))
            if end - outro_start >= 1.0:
                outro_s = round(end - outro_start, 3)
    total = round(intro_s + main_s + outro_s, 3)

    rng = random.Random(int(hashlib.md5(f"{clip['start']}|{p['name']}".encode()).hexdigest(), 16))
    every = p["cut_every_beats"]

    def shots(start: float, end: float, anchors: list[float], lo: float, hi: float, is_outro: bool) -> list[dict]:
        out = []
        if end - start < MIN_SHOT_S:
            return out
        points = _cut_points(beats_b if is_outro else beats_a, start, end, every)
        for i, (t0, t1) in enumerate(zip(points, points[1:])):
            dur = round(t1 - t0, 3)
            fx = _shot_effects(rng, dur, p, first_of_outro=is_outro and i == 0, on_bar=(i * every) % 4 == 0)
            src_dur = round(_source_duration(dur, fx["speed"]), 3)
            # Few moments available: on each repeat, move further into the moment instead of replaying it.
            anchor = anchors[i % len(anchors)] + 1.5 * (i // len(anchors))
            out.append({"at": round(t0, 3), "dur": dur, "src_start": _place(anchor, src_dur, lo, hi),
                        "src_dur": src_dur, **fx})
        return out

    intro = shots(0.0, intro_s, intro_anchors, clip["start"], clip["end"], False) if intro_s > 0 else []
    outro = shots(outro_start, outro_start + outro_s, outro_anchors, 0.0, vod_duration, True) if outro_s > 0 else []

    return {
        "preset": p["name"],
        "output": {k: out_cfg[k] for k in ("width", "height", "fps", "crf", "preset")},
        "timeline": {"intro_s": intro_s, "main_s": round(main_s, 3), "outro_s": outro_s, "total_s": total},
        "main": {"src_start": clip["start"], "src_end": clip["end"]},
        "intro": intro,
        "outro": outro,
        "music": None if not music else {
            "path": str(music["path"]), "title": music.get("title"), "license": music.get("license"),
            "attribution": music.get("attribution"), "bpm": a["bpm"],
            # (timeline start, track position, duration)
            "segments": [{"at": 0.0, "src": round(a_start, 3), "dur": round(outro_start if outro_s > 0 else total, 3)}]
            + ([{"at": round(outro_start, 3), "src": round(drop, 3), "dur": round(outro_s, 3)}] if outro_s > 0 else []),
            "seamless": outro_s == 0 or abs((a_start + outro_start) - drop) < 0.01,
        },
        "mix": {"music_volume": p["music_volume"], "music_under_speech": p["music_under_speech"],
                "speech_volume": p["speech_volume"]},
        "look": {"grade": p["grade_filter"], "grade_strength": p["grade_strength"], "vignette": p["vignette"],
                 "main_zoom": p["main_zoom"], "captions": p["captions"]},
        "beat_grid": "music" if music_analysis else f"fixed {p['fallback_bpm']} bpm (no music)",
    }
