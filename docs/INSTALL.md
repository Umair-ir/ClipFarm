# Installation log

Every command used to set up ClipFarm on the development machine, in order,
with what each step does and the problems hit along the way.
`scripts/install.sh` automates all of it and is safe to re-run.

**Reference machine:** MacBook (Apple M2, 8 GB RAM, macOS 26.5), Homebrew already installed,
Python 3.13.2 from Homebrew, ~26 GB free disk.

## 1. Check what's already installed

```bash
sw_vers; uname -m                     # macOS version, arm64 vs x86_64
for c in brew python3 ffmpeg ffprobe ollama git; do command -v $c || echo "$c MISSING"; done
sysctl -n hw.memsize                  # RAM — decides which models fit
df -h ~                               # free disk
```

Result: brew, python3 (3.13), and git present; **ffmpeg and ollama missing**.

## 2. System packages

```bash
brew install ffmpeg ollama
brew services start ollama            # runs the Ollama server at http://127.0.0.1:11434, starts at login
```

> **Gotcha:** Homebrew's `ffmpeg` (9.x) is built **without libass and freetype**, so the
> `subtitles` and `drawtext` filters don't exist. Check with
> `ffmpeg -hide_banner -filters | grep -E "subtitles|drawtext"`.
> ClipFarm works around this by drawing captions with Pillow and overlaying PNGs,
> so a custom FFmpeg build isn't needed.

## 3. Local LLM

```bash
ollama pull qwen2.5:3b                # 1.9 GB on disk, ~2.2 GB RAM when loaded
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:3b","prompt":"Reply with just: OK","stream":false}'
```

Why `qwen2.5:3b`: on 8 GB RAM a 7B/8B model runs alongside the OS only barely; 3B models
leave headroom and Qwen 2.5 follows JSON schemas reliably. On a 16 GB+ machine try
`qwen2.5:7b` or `llama3.1:8b` (set `CLIPFARM_OLLAMA_MODEL` in `.env`).

## 4. Python environment

```bash
cd ~/Desktop/ClipFarm
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**Whisper choice:** `faster-whisper` instead of the `openai-whisper` package. It runs the
same OpenAI Whisper weights through CTranslate2: about 4× faster on CPU, lower RAM, and
no PyTorch dependency (saves ~2 GB of disk). Apple Silicon has no CTranslate2 GPU
backend, so it runs on CPU with int8 quantization.

## 5. Models downloaded on first use

```bash
# Face detector for speaker-focused cropping (OpenCV YuNet, 230 KB)
mkdir -p models
curl -sSL -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Whisper "small" (~460 MB) — cached under ~/.cache/huggingface
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

## 6. URL downloads (optional)

```bash
.venv/bin/pip install yt-dlp          # included in requirements.txt
brew install deno                     # JS runtime yt-dlp now needs for YouTube
```

YouTube also requires solving a JavaScript challenge to unlock every format. Without the
solver, yt-dlp warns `n challenge solving failed: Some formats may be missing`, but
downloads still work with the formats it can see. To enable the solver, which downloads
yt-dlp's official solver script from GitHub:
`yt-dlp --remote-components ejs:github …` (CLI), or pass `remote_components={"ejs:github"}`
in `clipfarm/download.py`. Upgrade yt-dlp regularly (`pip install -U yt-dlp`), because
YouTube changes break older versions.

## 7. Verify

```bash
cp .env.example .env
.venv/bin/python -m clipfarm doctor        # all [ok]
.venv/bin/python tests/test_e2e.py         # generates a sample video, clips it, asserts 1080x1920 + audio + metadata
.venv/bin/python -m pytest tests -q
```

## Problems found and how they were fixed

| Problem | Cause | Fix in code |
|---|---|---|
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` during transcription | PyAV 19 removed an argument that faster-whisper 1.2.1 still passes | Decode audio with the `ffmpeg` CLI into a NumPy array (`transcribe._decode_audio`), skipping PyAV |
| `objc: Class AVFFrameReceiver is implemented in both …` | `opencv-python` and `av` each bundle their own `libavdevice`; loading both in one process can crash | Whisper runs in a child process (`python -m clipfarm.transcribe`). This also frees Whisper's RAM before Ollama loads |
| Ollama call hung for 300 s and timed out | Structured JSON output occasionally runs away | `num_predict` cap + one retry at temperature 0 |
| `cv2 has no attribute CascadeClassifier` | OpenCV 5 removed the Haar cascades | Switched to `cv2.FaceDetectorYN` (YuNet), which is more accurate |
| No burned-in subtitles possible | Homebrew FFmpeg lacks libass/freetype | Pillow-rendered caption PNGs, combined into one overlay stream via the concat demuxer |
| Clips ended too early or ran into the next topic | 3B model is imprecise about line ranges | Second "refine bounds" LLM pass per clip that shows ±context and asks only for start and end lines |

## Measured performance (M2, 8 GB)

- Transcription, Whisper `small` int8: **132 s audio in 28 s** (~4.7× real time) → a 3-hour VOD ≈ 40 min.
- Highlight detection: ~20 s per 5-minute window + ~3 s per candidate refinement → a 3-hour VOD ≈ 15–20 min.
- Render: ~6 s per 30 s clip (libx264 `medium`, CRF 20).
