"""Synthesize an original 128 BPM test track (build-up, then a drop) — no third-party audio.

    python tests/make_sample_music.py [OUT.wav]
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "samples" / "sample_beat_128bpm.wav"
SR, BPM = 44100, 128
BUILD_BARS, DROP_BARS = 8, 16


def _env(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (SR * decay))


def kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    freq = 50 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(n, 0.18)


def snare(n: int, rng) -> np.ndarray:
    return rng.uniform(-1, 1, n) * _env(n, 0.07) * 0.6


def hat(n: int, rng) -> np.ndarray:
    noise = rng.uniform(-1, 1, n)
    return np.diff(noise, prepend=0) * _env(n, 0.02) * 0.25


def make(out: Path = DEFAULT_OUT) -> Path:
    rng = np.random.default_rng(7)
    beat = int(SR * 60 / BPM)
    total_beats = 4 * (BUILD_BARS + DROP_BARS)
    y = np.zeros(beat * total_beats + SR)
    roots = [55.0, 55.0, 43.65, 49.0]  # A, A, F, G bass line per bar
    for b in range(total_beats):
        i = b * beat
        bar = b // 4
        in_drop = bar >= BUILD_BARS
        progress = min(1.0, bar / BUILD_BARS)
        for half in (0, beat // 2):  # off-beat hats
            y[i + half: i + half + beat // 2] += hat(beat // 2, rng) * (0.4 + 0.6 * progress)
        if in_drop:
            y[i: i + beat] += kick(beat) * 0.9
            if b % 4 in (1, 3):
                y[i: i + beat] += snare(beat, rng)
            t = np.arange(beat) / SR
            f = roots[bar % 4]
            y[i: i + beat] += 0.35 * np.sign(np.sin(2 * np.pi * f * t)) * _env(beat, 0.25)
        else:
            # rising pad + snare roll in the last build bar
            t = np.arange(beat) / SR
            y[i: i + beat] += 0.08 * progress * np.sin(2 * np.pi * (220 + 220 * progress) * t)
            if bar == BUILD_BARS - 1:
                for k in range(4):
                    j = i + k * beat // 4
                    y[j: j + beat // 4] += snare(beat // 4, rng) * (0.3 + 0.15 * k)
    y = y / np.max(np.abs(y)) * 0.9
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    print(out)
    return out


if __name__ == "__main__":
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
