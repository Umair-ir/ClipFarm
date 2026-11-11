"""`python -m clipfarm doctor` — verify every external dependency is usable."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import requests


def _ok(label: str, detail: str = "") -> bool:
    print(f"  [ok]   {label} {detail}")
    return True


def _fail(label: str, fix: str) -> bool:
    print(f"  [FAIL] {label}\n         fix: {fix}")
    return False


def run(cfg: dict) -> bool:
    print("ClipFarm dependency check")
    results = []

    v = sys.version_info
    results.append(_ok("python", f"{v.major}.{v.minor}.{v.micro}") if v >= (3, 10)
                   else _fail("python >= 3.10", "brew install python@3.12"))

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            ver = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout.split("\n")[0]
            results.append(_ok(tool, ver.split(" Copyright")[0]))
        else:
            results.append(_fail(tool, "brew install ffmpeg"))

    for mod in ("faster_whisper", "cv2", "PIL", "requests", "yaml"):
        # find_spec, not import: cv2 and av must not load into one process (see transcribe.py)
        if importlib.util.find_spec(mod):
            results.append(_ok(f"python module {mod}"))
        else:
            results.append(_fail(f"python module {mod}", "pip install -r requirements.txt"))

    font = Path(cfg["render"]["subtitle_font"])
    results.append(_ok("subtitle font", font.name) if font.exists()
                   else _fail(f"subtitle font {font}", "set render.subtitle_font to an existing .ttf"))

    face_model = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
    results.append(_ok("face model", face_model.name) if face_model.exists()
                   else _fail("face model (YuNet)", "bash scripts/install.sh"))

    host, model = cfg["ollama"]["host"], cfg["ollama"]["model"]
    try:
        tags = requests.get(f"{host}/api/tags", timeout=5).json()
        results.append(_ok("ollama server", host))
        names = {m["name"] for m in tags.get("models", [])}
        results.append(_ok("ollama model", model) if model in names or f"{model}:latest" in names
                       else _fail(f"ollama model {model}", f"ollama pull {model}"))
    except requests.RequestException:
        results.append(_fail(f"ollama server at {host}", "brew services start ollama"))

    ok = all(results)
    print("All good." if ok else "Some checks failed.")
    return ok
