"""Local Whisper transcription (faster-whisper / CTranslate2) with word timestamps.

Transcripts are cached at cache/transcripts/<file_hash>.<model>.json so a VOD
is only ever transcribed once per model.

Whisper runs in a child process (`python -m clipfarm.transcribe ...`) because
faster-whisper's PyAV and OpenCV each bundle their own libavdevice, which
can crash when both load into one process on macOS. It also returns
Whisper's RAM to the system before the LLM loads.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

_model_cache: dict[tuple, object] = {}


def _load_model(wcfg: dict):
    key = (wcfg["model"], wcfg["device"], wcfg["compute_type"])
    if key not in _model_cache:
        from faster_whisper import WhisperModel  # heavy import, keep lazy
        log.info("loading whisper model", extra={"model": wcfg["model"]})
        _model_cache[key] = WhisperModel(wcfg["model"], device=wcfg["device"],
                                         compute_type=wcfg["compute_type"])
    return _model_cache[key]


def transcribe(video: Path, file_hash: str, cfg: dict) -> dict:
    wcfg = cfg["whisper"]
    cache_file = cfg["paths"]["cache"] / "transcripts" / f"{file_hash}.{wcfg['model']}.json"
    if cache_file.exists():
        log.info("transcript cache hit", extra={"file": cache_file.name})
    else:
        log.info("transcribing", extra={"video": video.name, "model": wcfg["model"]})
        subprocess.run([sys.executable, "-m", "clipfarm.transcribe", str(video), str(cache_file),
                        json.dumps(wcfg)], check=True, cwd=Path(__file__).resolve().parent.parent)
    return json.loads(cache_file.read_text())


def _decode_audio(video: Path, sr: int = 16000):
    """Decode to 16 kHz mono float32 with the ffmpeg CLI.

    Bypasses faster-whisper's PyAV decoder (version-fragile: PyAV 19 broke it).
    Memory: ~230 MB per hour of audio.
    """
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _transcribe_to_file(video: Path, cache_file: Path, wcfg: dict) -> None:
    model = _load_model(wcfg)
    t0 = time.time()
    segments, info = model.transcribe(
        _decode_audio(video),
        language=wcfg.get("language"),
        word_timestamps=True,
        vad_filter=wcfg.get("vad_filter", True),
        beam_size=5,
    )
    out_segments = []
    for seg in segments:  # generator — decoding happens here
        out_segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                      for w in (seg.words or [])],
        })
        if len(out_segments) % 50 == 0:
            log.info("transcribing", extra={"at_s": round(seg.end), "of_s": round(info.duration)})

    result = {
        "language": info.language,
        "duration": info.duration,
        "model": wcfg["model"],
        "segments": out_segments,
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=1))
    log.info("transcribed", extra={"segments": len(out_segments), "audio_s": round(info.duration),
                                   "took_s": round(time.time() - t0, 1), "lang": info.language})


if __name__ == "__main__":
    from .config import load_config
    from .log import setup_logging

    setup_logging(load_config()["paths"]["logs"])
    _transcribe_to_file(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))
