"""ClipFarm CLI.

  python -m clipfarm doctor
  python -m clipfarm ingest PATH --streamer NAME [--date YYYY-MM-DD]
  python -m clipfarm ingest-inbox --streamer NAME      # everything in data/inbox/
  python -m clipfarm process [VOD_ID ...] [--max-clips N]   # default: all pending
  python -m clipfarm run PATH --streamer NAME           # ingest + process in one go
  (PATH can also be a URL — YouTube, Twitch, … — downloaded with yt-dlp. Authorized content only.)
  python -m clipfarm status

  python -m clipfarm edit CLIP.json [...] [--preset auto|hype|...] [--music FILE] [--plan-only]
  python -m clipfarm music add FILE --license L --source URL --moods hype,epic [--attribution TEXT]
  python -m clipfarm music list

  python -m clipfarm gta6 init                     # create input/gta6/ folders + sources.yaml template
  python -m clipfarm gta6 import DIR --origin O --credit C [--url U]   # bulk-add a media folder
  python -m clipfarm gta6 make [--styles ABC] [--cinematic-count N]
  python -m clipfarm gta6 review                   # list outputs and their review status
  python -m clipfarm gta6 approve NAME | reject NAME [--note TEXT]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from . import doctor
from .download import download, is_url
from .config import load_config
from .log import setup_logging
from .pipeline import process_vod
from .store import VIDEO_EXTS, open_queue

log = logging.getLogger("clipfarm")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clipfarm")
    p.add_argument("--config", type=Path)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")
    sub.add_parser("status")

    for name in ("ingest", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("paths", nargs="+", help="video files or URLs")
        sp.add_argument("--streamer", required=True)
        sp.add_argument("--date", type=date.fromisoformat)
        if name == "run":
            sp.add_argument("--max-clips", type=int)

    sp = sub.add_parser("ingest-inbox")
    sp.add_argument("--streamer", required=True)

    sp = sub.add_parser("process")
    sp.add_argument("vod_ids", nargs="*", type=int)
    sp.add_argument("--max-clips", type=int)

    sp = sub.add_parser("edit", help="beat-synced fan edit from rendered clips")
    sp.add_argument("clips", nargs="+", type=Path, help="clip metadata .json (or the .mp4 next to it)")
    sp.add_argument("--preset", default="auto")
    sp.add_argument("--music", type=Path, help="a licensed track already in the music library")
    sp.add_argument("--plan-only", action="store_true", help="write the edit plan JSON without rendering")

    sp = sub.add_parser("music", help="manage the licensed music library")
    msub = sp.add_subparsers(dest="music_cmd", required=True)
    ma = msub.add_parser("add")
    ma.add_argument("file", type=Path)
    ma.add_argument("--license", required=True, help="e.g. CC0, CC-BY-4.0, royalty-free, owned")
    ma.add_argument("--source", required=True, help="where the track and its license came from")
    ma.add_argument("--moods", default="", help="comma-separated, e.g. hype,epic")
    ma.add_argument("--title")
    ma.add_argument("--artist")
    ma.add_argument("--attribution", help="credit line appended to video descriptions")
    msub.add_parser("list")

    sp = sub.add_parser("gta6", help="GTA 6 Shorts generator")
    gsub = sp.add_subparsers(dest="gta6_cmd", required=True)
    gsub.add_parser("init")
    gi = gsub.add_parser("import", help="copy a media folder into input/gta6 and list it in sources.yaml")
    gi.add_argument("folder", type=Path)
    gi.add_argument("--origin", required=True, help='e.g. "official Rockstar media", "own gameplay"')
    gi.add_argument("--credit", required=True, help='e.g. "Image: Rockstar Games"')
    gi.add_argument("--url")
    gi.add_argument("--keep-logos", action="store_true", help="also import *_With_Logos variants")
    gm = gsub.add_parser("make")
    gm.add_argument("--styles", default="ABC", help="A=breakdown, B=cinematic, C=facts (e.g. AC)")
    gm.add_argument("--cinematic-count", type=int, default=1)
    gsub.add_parser("review")
    for verb in ("approve", "reject"):
        g = gsub.add_parser(verb)
        g.add_argument("name", help="output folder name (or a unique prefix)")
        g.add_argument("--note")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg["paths"]["logs"])

    if args.cmd == "doctor":
        return 0 if doctor.run(cfg) else 1

    if args.cmd in ("edit", "music"):
        return _edit_commands(args, cfg)
    if args.cmd == "gta6":
        return _gta6_commands(args, cfg)

    q = open_queue(cfg)

    if args.cmd == "status":
        for v in q.all_vods():
            print(f"#{v['id']:<4} {v['status']:<12} {v['streamer']:<16} {v['stream_date']}  {Path(v['path']).name}"
                  + (f"  ERROR: {v['error']}" if v["error"] else ""))
        return 0

    if args.cmd in ("ingest", "run", "ingest-inbox"):
        ids = []
        if args.cmd == "ingest-inbox":
            for path in sorted(f for f in cfg["paths"]["inbox"].iterdir() if f.suffix.lower() in VIDEO_EXTS):
                ids.append(q.ingest(path, args.streamer, move=True))
        for item in getattr(args, "paths", []):
            if is_url(item):
                try:
                    path, meta = download(item, cfg["paths"]["inbox"])
                except Exception as exc:
                    log.error("download failed: %s", str(exc).removeprefix("ERROR: "), extra={"url": item})
                    continue
                ids.append(q.ingest(path, args.streamer, args.date or meta["upload_date"],
                                    move=True, source=meta["url"]))
            else:
                ids.append(q.ingest(Path(item), args.streamer, args.date))
        if not ids:
            print("Nothing was queued.", file=sys.stderr)
            return 1
        if args.cmd != "run":
            print("Queued VOD ids:", ids)
            return 0
    else:
        ids = args.vod_ids or [v["id"] for v in q.pending()]

    failed = 0
    for vod_id in ids:
        try:
            outs = process_vod(q, vod_id, cfg, args.max_clips)
            for o in outs:
                print(o)
        except Exception:
            failed += 1  # already logged; continue with the rest of the batch
    return 1 if failed else 0


def _edit_commands(args: argparse.Namespace, cfg: dict) -> int:
    from .edit import load_edit_config
    from .edit import music as musiclib

    ecfg = load_edit_config()
    lib = ecfg["music"]["library"]

    if args.cmd == "music":
        if args.music_cmd == "add":
            moods = [m.strip() for m in args.moods.split(",") if m.strip()]
            dest = musiclib.add(args.file, lib, args.license, args.source, moods,
                                args.title, args.artist, args.attribution)
            a = musiclib.analysis(musiclib.Track(path=str(dest)), cfg["paths"]["cache"])
            print(f"Added {dest.name}: {a['bpm']:.0f} BPM, {a['duration']:.0f}s, license={args.license}")
        else:
            for t in musiclib.library(lib, ecfg["music"]["require_license"]):
                a = musiclib.analysis(t, cfg["paths"]["cache"])
                print(f"{t.path.name:<40} {a['bpm']:>4.0f} BPM {a['duration']:>5.0f}s  "
                      f"{t['license']:<14} moods={','.join(t['moods'])}")
        return 0

    from .edit.run import make_edit

    failed = 0
    for clip in args.clips:
        try:
            print(make_edit(clip.with_suffix(".json"), args.preset, cfg, ecfg, args.music, args.plan_only))
        except Exception:
            log.exception("edit failed", extra={"clip": str(clip)})
            failed += 1
    return 1 if failed else 0


def _gta6_commands(args: argparse.Namespace, cfg: dict) -> int:
    from .gta6 import load_gta6_config
    from .gta6 import run as gta6
    from .gta6.sources import init_manifest

    gcfg = load_gta6_config()
    if args.gta6_cmd == "init":
        path, added = init_manifest(gcfg["paths"]["input"])
        print(f"{path}: {added} new file(s) listed. Fill in `origin` (and credit/description) for each.")
        return 0
    if args.gta6_cmd == "import":
        from .gta6.importer import import_folder
        counts = import_folder(args.folder, gcfg["paths"]["input"], args.origin, args.credit, args.url,
                               skip_logos=not args.keep_logos)
        print(f"Imported {counts['images']} image(s), {counts['videos']} video(s) "
              f"({counts['skipped_existing']} already present). Listed in input/gta6/sources.yaml.")
        return 0
    if args.gta6_cmd == "make":
        outs = gta6.make(cfg, gcfg, args.styles, args.cinematic_count)
        for o in outs:
            print(o)
        print(f"{len(outs)} Short(s) waiting for review: python -m clipfarm gta6 review")
        return 0 if outs else 1
    if args.gta6_cmd == "review":
        rows = gta6.reviews(gcfg)
        for d, m in rows:
            print(f"{m['review']['status']:<9} {m['style']:<10} {m['duration_s']:>5.1f}s  {d.name}\n"
                  f"          {m['title']}")
        if not rows:
            print("No outputs yet.")
        return 0
    status = "approved" if args.gta6_cmd == "approve" else "rejected"
    print(f"{status}: {gta6.set_review(gcfg, args.name, status, args.note).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
