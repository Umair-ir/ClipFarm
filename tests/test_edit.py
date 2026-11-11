"""Fan-edit tests: license gate, beat-aligned planning, no-music fallback, full render.

    pytest tests/test_edit.py -s
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from clipfarm.edit import load_edit_config, resolve_preset  # noqa: E402
from clipfarm.edit import beats as beatlib  # noqa: E402
from clipfarm.edit import music as musiclib  # noqa: E402
from clipfarm.edit.plan import build  # noqa: E402

CLIP = {"start": 30.0, "end": 55.0}


def _track(tmp_path: Path) -> Path:
    from make_sample_music import make
    return make(tmp_path / "beat.wav")


def test_library_refuses_unlicensed_tracks(tmp_path):
    lib = tmp_path / "music"
    lib.mkdir()
    shutil.copy(_track(tmp_path), lib / "nolicense.wav")
    assert musiclib.library(lib) == []
    musiclib.add(lib / "nolicense.wav", lib, license="CC0", source="test", moods=["hype"])
    assert [t.path.name for t in musiclib.library(lib)] == ["nolicense.wav"]


def test_drop_detection_on_synthetic_track(tmp_path):
    a = beatlib.analyze(_track(tmp_path))
    assert 125 <= a["bpm"] <= 131
    drop = a["beats"][beatlib.find_drop(a, need=6)]
    assert abs(drop - 15.0) < 0.25, drop  # build is 8 bars at 128 BPM


def test_plan_cuts_on_beats_and_drop_at_outro(tmp_path):
    ecfg = load_edit_config()
    p = resolve_preset(ecfg, "hype")
    a = beatlib.analyze(_track(tmp_path))
    track = {"path": "beat.wav", "title": "t", "license": "CC0"}
    plan = build(CLIP, p, ecfg["output"], track, a, vod_duration=300,
                 intro_anchors=[35.0, 40.0], outro_anchors=[100.0, 150.0, 200.0])
    tl = plan["timeline"]
    assert tl["total_s"] <= ecfg["output"]["max_total_s"]
    seg_a, seg_b = plan["music"]["segments"]
    assert seg_b["at"] == round(tl["intro_s"] + tl["main_s"], 3)
    # Every outro cut lands on a track beat (timeline -> track time via segment B).
    track_beats = a["beats"]
    for shot in plan["outro"]:
        t_track = shot["at"] - seg_b["at"] + seg_b["src"]
        assert min(abs(t_track - b) for b in track_beats) < 0.01, shot
    assert plan["outro"][0]["flash"] > 0  # the drop always flashes


def test_plan_without_music_uses_fixed_grid():
    ecfg = load_edit_config()
    p = resolve_preset(ecfg, "gaming-cinematic")
    plan = build(CLIP, p, ecfg["output"], None, None, vod_duration=300,
                 intro_anchors=[35.0], outro_anchors=[100.0])
    assert plan["music"] is None and plan["beat_grid"].startswith("fixed")
    period = 60 / p["fallback_bpm"]
    for shot in plan["outro"][1:]:
        rel = shot["at"] - plan["outro"][0]["at"]
        assert abs(rel / period - round(rel / period)) < 0.01


def test_full_edit_render(tmp_path):
    """Renders a real edit from the e2e test's clip (runs the e2e pipeline first if needed)."""
    from test_e2e import TEST_DIR, run as run_e2e
    from clipfarm.config import load_config
    from clipfarm.edit.run import make_edit
    from clipfarm.render import probe

    clip_mp4 = run_e2e()[0]
    cfg = load_config()
    for key in ("library", "clips", "logs"):
        cfg["paths"][key] = TEST_DIR / key
    cfg["paths"]["db"] = TEST_DIR / "test.db"
    cfg["paths"]["edits"] = TEST_DIR / "edits"

    ecfg = load_edit_config()
    lib = tmp_path / "music"
    musiclib.add(_track(tmp_path), lib, license="owned", source="tests/make_sample_music.py",
                 moods=["hype"], attribution="Music: test beat")
    ecfg["music"]["library"] = lib

    out = make_edit(clip_mp4.with_suffix(".json"), "hype", cfg, ecfg)
    info = probe(out)
    plan = json.loads(out.with_suffix(".plan.json").read_text())
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    assert abs(info["duration"] - plan["timeline"]["total_s"]) < 0.2
    meta = json.loads(out.with_suffix(".json").read_text())
    assert "Music: test beat" in meta["description"]
    print(f"\nPASS  {out}  {info['duration']:.1f}s")
