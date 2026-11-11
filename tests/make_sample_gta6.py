"""Synthetic stand-in inputs for the GTA 6 generator (no game footage needed). macOS: uses `say`.

    python tests/make_sample_gta6.py [INPUT_DIR]     # default data/test_gta6/input

Creates: commentary.mp4 (scene cuts + voice), footage.mp4 (silent scenes), images/*.png,
research/sample.yaml, sources.yaml.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "test_gta6" / "input"

SCENES = [  # lavfi sources: varied color/motion so scene detection + scoring have something to find
    "gradients=s=1920x1080:c0=0xff2d95:c1=0x00e5ff:c2=0xffb000:speed=0.02:rate=30",
    "mandelbrot=s=1920x1080:rate=30",
    "life=s=480x270:mold=10:rate=30:ratio=0.3:death_color=#1a0033:life_color=#ff2d95,scale=1920:1080",
    "testsrc2=s=1920x1080:rate=30",
    "gradients=s=1920x1080:c0=0x0d0221:c1=0xff6b00:speed=0.05:rate=30",
    "cellauto=s=480x270:rule=110:rate=30,scale=1920:1080,colorchannelmixer=rr=1:gg=0.4:bb=0.8",
    "smptehdbars=s=1920x1080:rate=30",
]

COMMENTARY = [
    "Okay so let's break down this clip, because there is way more going on than people noticed.",
    "First, look at the colors in this shot. The whole city is lit up in pink and orange, that's the sunset look.",
    "Now watch this car chase. The camera follows the car really low to the ground and it feels insanely fast.",
    "Here's the part nobody is talking about. In the background you can see a crowd at the beach reacting.",
    "That tells me the world reacts to what you do, which honestly changes how missions could play out.",
    "And this last shot, the skyline at night, is just unreal. The reflections on the water look real.",
    "So which detail did you catch first? Honestly I watched this ten times and kept finding new things.",
]


def _scene(src: str, dur: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src, "-t", f"{dur}",
                    "-vf", "format=yuv420p,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", str(out)],
                   check=True)


def _concat(parts: list[Path], out: Path, audio: Path | None = None) -> None:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio:
        cmd += ["-i", str(audio), "-shortest", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "copy", str(out)], check=True)
    listing.unlink()


def _images(dir_: Path) -> list[Path]:
    dir_.mkdir(parents=True, exist_ok=True)
    out = []
    rng = np.random.default_rng(4)
    palettes = [((255, 94, 91), (61, 18, 92)), ((255, 176, 0), (255, 45, 149)), ((0, 229, 255), (13, 2, 33)),
                ((255, 107, 0), (30, 10, 60))]
    for i, (top, bottom) in enumerate(palettes):
        W, H = 1920, 1080
        grad = np.linspace(0, 1, H)[:, None, None]
        arr = (np.array(top) * (1 - grad) + np.array(bottom) * grad).repeat(W, axis=1).astype(np.uint8)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        d.ellipse([W * 0.6, H * 0.25, W * 0.6 + 300, H * 0.25 + 300], fill=(255, 230, 120))
        for _ in range(28):  # skyline
            x = int(rng.uniform(0, W)); w = int(rng.uniform(40, 140)); h = int(rng.uniform(150, 600))
            d.rectangle([x, H - h, x + w, H], fill=(10, 5, 25))
            for _ in range(10):
                wx, wy = x + int(rng.uniform(5, w - 10)), H - int(rng.uniform(10, h - 10))
                d.rectangle([wx, wy, wx + 6, wy + 9], fill=(255, 220, 120))
        p = dir_ / f"shot_{i + 1}.png"
        img.filter(ImageFilter.SMOOTH).save(p)
        out.append(p)
    return out


def make(input_dir: Path = DEFAULT_DIR) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        # Commentary video: voice over scenes that cut every ~6 s.
        voice = t / "voice.aiff"
        subprocess.run(["say", "-v", "Daniel", "-r", "190", "-o", str(voice), "[[slnc 400]] " + " [[slnc 700]] ".join(COMMENTARY)],
                       check=True)
        vdur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     str(voice)], capture_output=True, text=True).stdout)
        parts, total, i = [], 0.0, 0
        while total < vdur + 1:
            p = t / f"c{i}.mp4"
            _scene(SCENES[i % len(SCENES)], 6.0, p)
            parts.append(p); total += 6.0; i += 1
        _concat(parts, input_dir / "commentary.mp4", voice)
        # Silent footage for the cinematic edit.
        parts = []
        for k, src in enumerate(SCENES[::-1] + SCENES[:3]):
            p = t / f"f{k}.mp4"
            _scene(src, 4.0, p)
            parts.append(p)
        _concat(parts, input_dir / "footage.mp4")
    images = _images(input_dir / "images")

    (input_dir / "research").mkdir(exist_ok=True)
    (input_dir / "research" / "sample.yaml").write_text(yaml.safe_dump({
        "topic": "What we actually know about GTA 6",
        "facts": [
            {"id": "release", "claim": "GTA 6 is scheduled to release on November 19, 2026 for PS5 and Xbox Series X|S.",
             "status": "confirmed", "source": "https://www.gamesradar.com/gta-6-release-date-news-trailers-map-gameplay-rumours/"},
            {"id": "setting", "claim": "The game is set in the state of Leonida, which includes Vice City.",
             "status": "confirmed", "source": "https://www.rockstargames.com/VI"},
            {"id": "leads", "claim": "The two protagonists are Lucia and Jason.",
             "status": "confirmed", "source": "https://www.rockstargames.com/VI"},
            {"id": "pc", "claim": "A PC version could arrive later than the console versions, as with past GTA games.",
             "status": "speculation", "source": "https://www.gamesradar.com/gta-6-release-date-news-trailers-map-gameplay-rumours/"},
            {"id": "mapsize", "claim": "Some fans believe the map will be bigger than GTA 5's.", "status": "rumor"},
        ],
    }, sort_keys=False))

    files = {
        "commentary.mp4": {"origin": "synthetic test asset", "credit": "Test commentary (generated)",
                           "description": "Commentary over colorful city, car chase and skyline scenes"},
        "footage.mp4": {"origin": "synthetic test asset", "credit": "Test footage (generated)",
                        "description": "Fast colorful action scenes"},
    }
    descs = ["Sunset skyline in pink", "Neon city at dusk", "Night skyline in blue", "Orange sunset over towers"]
    for img, desc in zip(images, descs):
        files[f"images/{img.name}"] = {"origin": "synthetic test asset", "credit": "Test image (generated)",
                                       "description": desc}
    (input_dir / "sources.yaml").write_text(yaml.safe_dump({"files": files}, sort_keys=False))
    print(input_dir)
    return input_dir


if __name__ == "__main__":
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR)
