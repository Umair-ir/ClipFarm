"""GTA 6 generator tests.

    pytest tests/test_gta6.py -s                 # unit tests + cinematic render (~2 min)
    CLIPFARM_SLOW=1 pytest tests/test_gta6.py -s # also renders breakdown + facts (~6 min more)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from clipfarm.gta6.script import _new_specifics, _short  # noqa: E402
from clipfarm.gta6.sources import discover, init_manifest  # noqa: E402
from clipfarm.gta6.run import _load_research  # noqa: E402
from clipfarm.gta6.styles import _auto_emphasis  # noqa: E402

TEST_DIR = ROOT / "data" / "test_gta6"


def test_manifest_gate(tmp_path):
    inp = tmp_path / "gta6"
    (inp / "images").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "samples" / "sample_stream.mp4", inp / "clip.mp4")
    assert discover(inp, require_sources=True)["videos"] == []          # unlisted -> skipped
    path, added = init_manifest(inp)
    assert added == 1 and discover(inp, True)["videos"] == []           # listed but empty origin -> skipped
    data = yaml.safe_load(path.read_text())
    data["files"]["clip.mp4"]["origin"] = "own gameplay"
    path.write_text(yaml.safe_dump(data))
    assert [v["id"] for v in discover(inp, True)["videos"]] == ["clip.mp4"]


def test_research_validation(tmp_path):
    bad = tmp_path / "r.yaml"
    bad.write_text(yaml.safe_dump({"topic": "t", "facts": [{"claim": "x", "status": "confirmed"}]}))
    with pytest.raises(ValueError, match="needs a source"):
        _load_research(bad)
    bad.write_text(yaml.safe_dump({"topic": "t", "facts": [{"claim": "x", "status": "maybe"}]}))
    with pytest.raises(ValueError, match="status"):
        _load_research(bad)


def test_expansion_guard_catches_new_specifics():
    claim = "A PC version could arrive later than the console versions."
    assert _new_specifics("Rockstar confirmed it drops in 2027 for PC.", claim) == {"Rockstar", "2027"}
    assert _new_specifics("PC version might come later. Players should be patient.", claim) == set()


def test_short_trims_dangling_words():
    assert _short("GTA 6 Release Date & more", 5) == "GTA 6 Release Date"


def test_auto_emphasis_spacing():
    words = [{"word": w, "start": i * 1.0, "end": i * 1.0 + 0.5}
             for i, w in enumerate("look at these insane reflections and the gigantic skyline over vice city".split())]
    em = _auto_emphasis(words, "Insane reflections", [(0.0, 20.0)], hook_s=1.0)
    assert em and em[0]["word"] == "reflections"
    times = [next(w["start"] for w in words if w["word"] == e["word"]) for e in em]
    assert all(abs(a - b) >= 5 for i, a in enumerate(times) for b in times[i + 1:])


def _run_styles(styles: str) -> list[Path]:
    from make_sample_gta6 import make as make_inputs
    from clipfarm.config import load_config
    from clipfarm.gta6 import load_gta6_config
    from clipfarm.gta6 import run

    if not (TEST_DIR / "input" / "sources.yaml").exists():
        make_inputs(TEST_DIR / "input")
    cfg = load_config()
    g = load_gta6_config()
    g["paths"]["input"], g["paths"]["output"] = TEST_DIR / "input", TEST_DIR / "output"
    shutil.rmtree(TEST_DIR / "output", ignore_errors=True)
    return run.make(cfg, g, styles)


def _check_output(d: Path) -> dict:
    from clipfarm.render import probe

    info = probe(d / "video.mp4")
    meta = json.loads((d / "metadata.json").read_text())
    assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
    assert abs(info["duration"] - meta["duration_s"]) < 0.25
    assert (d / "thumbnail.jpg").exists() and (d / "project.json").exists()
    assert meta["review"]["status"] == "pending" and meta["sources"]
    assert "#gta6" in (t.lower() for t in meta["hashtags"])
    return meta


def test_cinematic_render():
    outs = _run_styles("B")
    assert len(outs) == 1
    meta = _check_output(outs[0])
    assert 30 <= meta["duration_s"] <= 60


@pytest.mark.skipif(not os.environ.get("CLIPFARM_SLOW"), reason="set CLIPFARM_SLOW=1")
def test_breakdown_and_facts_render():
    outs = _run_styles("AC")
    styles = {json.loads((d / "metadata.json").read_text())["style"]: d for d in outs}
    assert set(styles) == {"breakdown", "facts"}
    facts = _check_output(styles["facts"])
    for f in facts["facts"]:
        if f["status"] != "confirmed":
            assert f["status"].upper() in facts["description"]
    _check_output(styles["breakdown"])
