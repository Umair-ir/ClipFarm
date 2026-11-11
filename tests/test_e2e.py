"""End-to-end smoke test: sample video -> transcript -> highlight -> vertical clip.

    python tests/test_e2e.py            # or: pytest tests/test_e2e.py -s
    python tests/test_e2e.py my.mp4     # run against your own (authorized) footage

Uses an isolated copy of the config under data/test_run/ so it never touches
your real queue or library.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from clipfarm.config import load_config  # noqa: E402
from clipfarm.log import setup_logging  # noqa: E402
from clipfarm.pipeline import process_vod  # noqa: E402
from clipfarm.render import probe  # noqa: E402
from clipfarm.store import open_queue  # noqa: E402

TEST_DIR = ROOT / "data" / "test_run"


def run(video: Path | None = None) -> list[Path]:
    if video is None:
        from make_sample_video import DEFAULT_OUT, main as make_sample
        video = DEFAULT_OUT if DEFAULT_OUT.exists() else make_sample()

    cfg = load_config()
    # Fresh queue/library/clips each run, but share the transcript/LLM cache.
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    for key in ("library", "clips", "logs"):
        cfg["paths"][key] = TEST_DIR / key
        cfg["paths"][key].mkdir(parents=True)
    cfg["paths"]["db"] = TEST_DIR / "test.db"
    setup_logging(cfg["paths"]["logs"])

    q = open_queue(cfg)
    vod_id = q.ingest(video, streamer="test_streamer")
    outputs = process_vod(q, vod_id, cfg, max_clips=1)

    assert outputs, "pipeline produced no clips"
    clip = outputs[0]
    info = probe(clip)
    assert (info["width"], info["height"]) == (1080, 1920), info
    assert info["has_audio"], "clip lost its audio"
    lo, hi = cfg["highlights"]["min_clip_s"], cfg["highlights"]["max_clip_s"]
    assert lo - 1 <= info["duration"] <= hi + 1, f"duration {info['duration']:.1f}s outside {lo}-{hi}s"
    meta = json.loads(clip.with_suffix(".json").read_text())
    assert meta["title"] and "#shorts" in meta["hashtags"]

    print(f"\nPASS  {clip}\n      {info['duration']:.1f}s  {info['width']}x{info['height']}  "
          f"score={meta['score']}  [{meta['category']}] {meta['title']}")
    return outputs


def test_end_to_end():
    run()


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
