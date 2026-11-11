#!/usr/bin/env bash
# ClipFarm installer for macOS (Apple Silicon or Intel). Safe to re-run.
#   bash scripts/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

step() { printf "\n==> %s\n" "$*"; }

step "Homebrew"
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }

step "System packages: ffmpeg, ollama, deno (for yt-dlp), python"
for pkg in ffmpeg ollama deno; do
  brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"
done
command -v python3 >/dev/null || brew install python@3.13

step "Ollama service + model ($OLLAMA_MODEL)"
brew services start ollama >/dev/null
for _ in {1..20}; do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
ollama pull "$OLLAMA_MODEL"

step "Python virtualenv (.venv)"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

step "Face detection model (YuNet, 230 KB)"
mkdir -p models
[ -f models/face_detection_yunet_2023mar.onnx ] || curl -sSL -o models/face_detection_yunet_2023mar.onnx "$YUNET_URL"

step "Whisper model ($WHISPER_MODEL) — downloads once to ~/.cache/huggingface"
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('$WHISPER_MODEL', device='cpu', compute_type='int8')"

step "Config"
[ -f .env ] || cp .env.example .env

step "Dependency check"
.venv/bin/python -m clipfarm doctor

printf "\nDone. Try:  .venv/bin/python tests/test_e2e.py\n"
