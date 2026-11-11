"""Render an edit plan with FFmpeg.

  1. main clip  -> existing render_clip() (captions, face tracking, push-in), at the edit's size
  2. each shot  -> one short FFmpeg encode: 9:16 crop, punch-in, speed/ramp, motion blur, white flash
  3. final pass -> concat all segments, color grade + vignette, mix speech with the music bed
                   (ducked under speech, full during intro/outro, faded out), encode once.
Intermediates are near-lossless (CRF 14) so the final encode is the only real quality loss.
"""
from __future__ import annotations

import copy
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..render import RenderError, crop_window, face_center_x, probe, render_clip

log = logging.getLogger(__name__)

INTERMEDIATE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p"]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg failed: {proc.stderr.strip()[-1500:]}")


def _speed_filter(speed: dict, dur: float) -> str:
    if speed["type"] == "slow":
        return f"setpts=PTS/{speed['slow']}"
    if speed["type"] == "ramp":
        f, s = speed["fast"], speed["slow"]
        r = dur * speed["at"] * f  # ramp point in source seconds (fast part = `at` of the timeline duration)
        return f"setpts='if(lt(T,{r:.3f}),PTS/{f},({r:.3f}/{f}+(T-{r:.3f})/{s})/TB)'"
    return ""


def _render_shot(video: Path, info: dict, shot: dict, out: Path, W: int, H: int, fps: int, face: bool,
                 pre_crop: list[int] | None = None) -> None:
    """pre_crop: [w, h, x, y] content box inside the frame (e.g. letterbox bars removed) to crop from."""
    cx = face_center_x(video, shot["src_start"], shot["src_start"] + shot["src_dur"], samples=4) if face else None
    chain = []
    if pre_crop:
        pw, ph, px, py = pre_crop
        chain.append(f"crop={pw}:{ph}:{px}:{py}")
        info = {**info, "width": pw, "height": ph}
    cw, ch, cx_px, cy = crop_window(info, cx, W, H)
    zw, zh = int(W * shot["zoom"] / 2) * 2, int(H * shot["zoom"] / 2) * 2
    chain += [f"crop={cw}:{ch}:{cx_px}:{cy}", f"scale={zw}:{zh}:flags=lanczos", f"crop={W}:{H}", "setsar=1",
              "setpts=PTS-STARTPTS"]
    speed = _speed_filter(shot["speed"], shot["dur"])
    if speed:
        chain.append(speed)
    chain.append(f"fps={fps}")
    if shot["blur"]:
        chain.append("tmix=frames=3:weights='1 2 1'")
    chain += [f"trim=duration={shot['dur']:.3f}", "setpts=PTS-STARTPTS"]
    if shot["flash"] > 0:
        chain.append(f"fade=t=in:st=0:d={shot['flash']:.3f}:color=white")
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
          "-ss", f"{shot['src_start']:.3f}", "-t", f"{shot['src_dur'] + 0.5:.3f}", "-i", str(video),
          "-an", "-vf", ",".join(chain), *INTERMEDIATE, "-t", f"{shot['dur']:.3f}", str(out)])


def _music_volume_expr(intro_s: float, main_s: float, full: float, duck: float, ramp: float = 0.3) -> str:
    """Volume as a function of timeline t: `duck` during the main clip, `full` elsewhere, smooth ramps."""
    end = intro_s + main_s
    x = f"max({intro_s}-t,t-{end})"  # < 0 inside the main clip
    return f"{duck}+({full}-{duck})*clip(({x}+{ramp})/{ramp},0,1)"


def render_plan(plan: dict, video: Path, clip: dict, transcript: dict, cfg: dict, out_path: Path) -> Path:
    o = plan["output"]
    W, H, fps = o["width"], o["height"], o["fps"]
    tl = plan["timeline"]
    info = probe(video)
    face = cfg["render"]["face_tracking"]

    workdir = Path(tempfile.mkdtemp(prefix="clipfarm_edit_"))
    try:
        # 1. Main clip with the preset's caption style, at the edit's resolution.
        mcfg = copy.deepcopy(cfg)
        mcfg["render"].update({"width": W, "height": H, "zoom": plan["look"]["main_zoom"],
                               "crf": 14, "preset": "veryfast", **plan["look"]["captions"]})
        main = render_clip(video, {"start": clip["start"], "end": clip["end"]}, transcript,
                           workdir / "main.mp4", mcfg)
        main_has_audio = probe(main)["has_audio"]

        # 2. Montage shots.
        segments: list[Path] = []
        for name, shots in (("intro", plan["intro"]), ("outro", plan["outro"])):
            paths = []
            for i, shot in enumerate(shots):
                p = workdir / f"{name}_{i:03d}.mp4"
                _render_shot(video, info, shot, p, W, H, fps, face)
                paths.append(p)
            if name == "intro":
                segments += paths + [main]
            else:
                segments += paths
        log.info("segments rendered", extra={"intro": len(plan["intro"]), "outro": len(plan["outro"])})

        # 3. Final assembly.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for seg in segments:
            cmd += ["-i", str(seg)]
        music = plan["music"]
        if music:
            cmd += ["-i", music["path"]]
        n = len(segments)
        graph = [f"[{i}:v]fps={fps},scale={W}:{H},setsar=1,format=yuv420p[v{i}]" for i in range(n)]
        graph.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vc]")

        look = plan["look"]
        last = "vc"
        if look["grade"]:
            s = float(look["grade_strength"])
            if s >= 0.999:
                graph.append(f"[{last}]{look['grade']}[vg]")
            else:
                graph.append(f"[{last}]split[va][vb];[vb]{look['grade']}[vgr];"
                             f"[va][vgr]blend=all_mode=normal:all_opacity={s:.3f}[vg]")
            last = "vg"
        if look["vignette"]:
            graph.append(f"[{last}]vignette=PI/5[vv]")
            last = "vv"

        mix = plan["mix"]
        main_idx = len(plan["intro"])
        delay_ms = int(round(tl["intro_s"] * 1000))
        if main_has_audio:
            graph.append(f"[{main_idx}:a]volume={mix['speech_volume']},adelay={delay_ms}:all=1,"
                         f"apad=whole_dur={tl['total_s']}[sp]")
        else:
            graph.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={tl['total_s']}[sp]")
        if music:
            segs = music["segments"]
            # Each segment: exact slice of the track, padded with silence if the track runs out.
            graph.append(f"[{n}:a]aresample=48000,asplit={len(segs)}" + "".join(f"[ms{i}]" for i in range(len(segs))))
            for i, sg in enumerate(segs):
                graph.append(f"[ms{i}]atrim=start={sg['src']:.3f},asetpts=PTS-STARTPTS,"
                             f"apad=whole_dur={sg['dur']:.3f},atrim=duration={sg['dur']:.3f}[mt{i}]")
            graph.append("".join(f"[mt{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=0:a=1[mcat]")
            vol = _music_volume_expr(tl["intro_s"], tl["main_s"], mix["music_volume"], mix["music_under_speech"])
            fade_st = max(0.0, tl["total_s"] - 1.2)
            graph.append(f"[mcat]volume='{vol}':eval=frame,"
                         f"afade=t=in:d=0.2,afade=t=out:st={fade_st:.3f}:d=1.2[mu]")
            graph.append("[sp][mu]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[aout]")
        else:
            graph.append("[sp]anull[aout]")

        cmd += ["-filter_complex", ";".join(graph), "-map", f"[{last}]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", o["preset"], "-crf", str(o["crf"]), "-pix_fmt", "yuv420p",
                "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart", "-t", f"{tl['total_s']:.3f}", str(out_path)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run(cmd)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    log.info("edit rendered", extra={"out": out_path.name, "total_s": tl["total_s"], "preset": plan["preset"]})
    return out_path
