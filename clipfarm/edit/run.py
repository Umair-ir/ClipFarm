"""Make a fan edit from a rendered clip's metadata JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..render import probe
from ..store import file_hash, open_queue
from ..transcribe import transcribe
from . import music as musiclib
from . import resolve_preset
from .moments import intro_moments, loudness_curve, montage_moments
from .plan import build

log = logging.getLogger(__name__)


class EditError(RuntimeError):
    pass


def _pick_music(ecfg: dict, preset: dict, cfg: dict, music_path: Path | None, need_s: float,
                seed: str) -> musiclib.Track | None:
    tracks = musiclib.library(ecfg["music"]["library"], ecfg["music"]["require_license"])
    if music_path:
        match = [t for t in tracks if t.path.resolve() == music_path.resolve()]
        if not match:
            raise EditError(f"{music_path} is not a licensed track in {ecfg['music']['library']}. "
                            "Add it first: python -m clipfarm music add FILE --license ... --source ...")
        return match[0]
    track = musiclib.select(tracks, preset, cfg["paths"]["cache"], need_s, seed)
    if track is None:
        log.warning("no licensed music in library — rendering without music (fixed beat grid)")
    return track


def make_edit(clip_json: Path, preset_name: str, cfg: dict, ecfg: dict,
              music_path: Path | None = None, plan_only: bool = False) -> Path:
    meta = json.loads(clip_json.read_text())
    src = meta["source"]
    video = Path(src["path"])
    if not video.exists():
        raise EditError(f"Source VOD missing: {video}")
    clip = {"start": src["start"], "end": src["end"]}
    preset = resolve_preset(ecfg, preset_name, meta.get("category"))
    log.info("editing", extra={"clip": clip_json.stem, "preset": preset["name"]})

    digest = file_hash(video)
    transcript = transcribe(video, digest, cfg)
    vod_dur = probe(video)["duration"]

    q = open_queue(cfg)
    others = [(r["start_s"], r["end_s"]) for r in
              q.db.execute("SELECT start_s, end_s FROM clips WHERE vod_id = ?", (src["vod_id"],))
              if abs(r["start_s"] - clip["start"]) > 0.01]
    db = loudness_curve(video, digest, cfg["paths"]["cache"])
    outro_anchors = montage_moments(video, digest, cfg["paths"]["cache"], (clip["start"], clip["end"]), others)
    intro_anchors = intro_moments(db, (clip["start"], clip["end"]), 8)

    need_s = preset["intro_s"] + (clip["end"] - clip["start"]) + preset["outro_s"] + 10
    track = _pick_music(ecfg, preset, cfg, music_path, need_s, seed=clip_json.stem)
    analysis = musiclib.analysis(track, cfg["paths"]["cache"]) if track else None

    plan = build(clip, preset, ecfg["output"], track, analysis, vod_dur, intro_anchors, outro_anchors)

    out_dir = cfg["paths"]["edits"] / src["streamer"] / clip_json.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{clip_json.stem}_{preset['name']}"
    (out_dir / f"{stem}.plan.json").write_text(json.dumps(plan, indent=2))
    if plan_only:
        log.info("plan written", extra={"path": str(out_dir / f"{stem}.plan.json")})
        return out_dir / f"{stem}.plan.json"

    from .render import render_plan  # cv2 import kept out of plan-only runs
    out = render_plan(plan, video, clip, transcript, cfg, out_dir / f"{stem}.mp4")

    edit_meta = dict(meta)
    if plan["music"] and plan["music"].get("attribution"):
        edit_meta["description"] = f"{meta['description']}\n\n{plan['music']['attribution']}"
    edit_meta["edit"] = {"preset": preset["name"], "music": plan["music"], "timeline": plan["timeline"],
                         "from_clip": str(clip_json.with_suffix(".mp4"))}
    (out_dir / f"{stem}.json").write_text(json.dumps(edit_meta, indent=2))
    return out
