"""Beat tracking, tempo, per-beat energy and drop detection — NumPy only.

Adapted from easyedit (https://github.com/blixvip/easyedit, easyedit/beats.py @ e46bf77),
MIT License, Copyright (c) 2026 wasely. See THIRD_PARTY_NOTICES.md.
Changes: cache handling moved to music.py, added an energy curve, drop refinement
(find_drop no longer stops at the start of a riser), and a fixed-grid fallback for edits without music.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR, HOP, NFFT = 22050, 512, 2048
FPS = SR / HOP  # onset-envelope frames per second


def load_audio(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def onset_envelope(y: np.ndarray) -> np.ndarray:
    n = 1 + (len(y) - NFFT) // HOP
    if n <= 8:
        return np.zeros(8)
    win = np.hanning(NFFT).astype(np.float32)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * win, axis=1))
    # mel-ish compression: log magnitude, emphasize lows where kicks live
    logspec = np.log1p(100 * spec[:, : NFFT // 4])
    flux = np.maximum(0, np.diff(logspec, axis=0)).sum(axis=1)
    flux = np.concatenate([[0], flux])
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")
    flux = np.maximum(flux, 0)
    return flux / (flux.max() + 1e-9)


def tempo(env: np.ndarray) -> float:
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    bpms = np.arange(70, 181)
    lags = FPS * 60 / bpms
    strength = np.interp(lags, np.arange(len(ac)), ac)
    prior = np.exp(-0.5 * (np.log2(bpms / 120) / 0.9) ** 2)
    return float(bpms[np.argmax(strength * prior)])


def track(env: np.ndarray, bpm: float, tightness: float = 100) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker. Returns beat times in seconds."""
    period = FPS * 60 / bpm
    score = env.copy()
    back = np.full(len(env), -1)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for i in range(lo, len(env)):
        prev = np.arange(max(0, i - hi), i - lo + 1)
        if not len(prev):
            continue
        penalty = -tightness * np.log((i - prev) / period) ** 2
        cand = score[prev] + penalty
        j = int(np.argmax(cand))
        score[i] = env[i] + cand[j]
        back[i] = prev[j]
    tail = max(0, len(score) - int(period))
    i = tail + int(np.argmax(score[tail:]))
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1], dtype=np.float32) / FPS


def analyze(path: Path) -> dict:
    """BPM, beat times, per-beat loudness/onset strength, overall energy for one track."""
    y = load_audio(path)
    env = onset_envelope(y)
    bpm = tempo(env)
    beats = track(env, bpm)
    rms_frames = np.sqrt(np.convolve(y ** 2, np.ones(HOP) / HOP, mode="same")[::HOP] + 1e-12)
    loud = [float(rms_frames[int(b * FPS): int(b * FPS) + int(FPS * 0.5)].mean()) if int(b * FPS) < len(rms_frames) else 0
            for b in beats]
    strength = [float(env[max(0, int(b * FPS) - 2): int(b * FPS) + 3].max()) if int(b * FPS) < len(env) else 0
                for b in beats]
    return {
        "duration": len(y) / SR,
        "bpm": bpm,
        "beats": [round(float(b), 3) for b in beats],
        "loud": loud,
        "strength": strength,
        "energy": float(np.mean(rms_frames)),  # used for mood matching
    }


def analyze_librosa(path: Path) -> dict:
    """Same output as analyze(), using librosa's beat tracker (more robust on complex mixes)."""
    import librosa

    y, sr = librosa.load(str(path), sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo_raw, beat_t = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP, units="time")
    bpm = float(np.atleast_1d(tempo_raw)[0])
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    env = env / (env.max() + 1e-9)
    loud = [float(rms[int(b * FPS): int(b * FPS) + int(FPS * 0.5)].mean()) if int(b * FPS) < len(rms) else 0
            for b in beat_t]
    strength = [float(env[max(0, int(b * FPS) - 2): int(b * FPS) + 3].max()) if int(b * FPS) < len(env) else 0
                for b in beat_t]
    return {"duration": len(y) / SR, "bpm": round(bpm, 2), "beats": [round(float(b), 3) for b in beat_t],
            "loud": loud, "strength": strength, "energy": float(np.mean(rms)), "engine": "librosa"}


def find_drop(a: dict, need: float, min_t: float = 0.0) -> int:
    """Index of the beat where the music hits hardest, leaving `need` seconds after it."""
    beats, loud, strength = a["beats"], np.array(a["loud"]), np.array(a["strength"])
    if len(beats) < 8:
        return 0
    per = np.median(np.diff(beats))
    look = max(4, int(round(6 / per)))
    best, best_i = -1e9, 0
    for i, t in enumerate(beats):
        if t < min_t or t + need > a["duration"]:
            continue
        before = loud[max(0, i - look): i].mean() if i > 0 else loud[:look].mean() * 0.5
        after = loud[i: i + look * 2].mean()
        score = (after - before) / (after + before + 1e-6) + 0.35 * after / (loud.max() + 1e-6) \
            + 0.15 * strength[i]
        if score > best:
            best, best_i = score, i
    return _refine_drop(a, best_i, need)


def _refine_drop(a: dict, i: int, need: float) -> int:
    """The coarse pick often lands on the start of a riser/snare roll. Within the next two bars,
    move to the beat with the sharpest loudness jump (the actual hit)."""
    beats, loud = a["beats"], np.array(a["loud"])
    best_j, best_jump = i, -1e9
    for j in range(i, min(len(beats) - 2, i + 9)):
        if beats[j] + need > a["duration"]:
            break
        before = loud[max(0, j - 2): j].mean() if j >= 2 else loud[j]
        jump = loud[j: j + 2].mean() - before
        if jump > best_jump:
            best_j, best_jump = j, jump
    return best_j


def fixed_grid(bpm: float, duration: float) -> dict:
    """Synthetic beat grid for edits rendered without music, so cuts stay rhythmic."""
    period = 60.0 / bpm
    beats = [round(i * period, 3) for i in range(int(duration / period) + 1)]
    return {"duration": duration, "bpm": bpm, "beats": beats, "loud": [1.0] * len(beats),
            "strength": [1.0] * len(beats), "energy": 0.0}
