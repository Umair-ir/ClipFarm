# ClipFarm

Local, subscription-free pipeline that turns authorized livestream VODs into
vertical YouTube Shorts. Transcription, highlight detection and rendering all run
on your machine.

```
VOD ─▶ ingest/queue ─▶ Whisper transcript ─▶ Ollama highlight picks ─▶ 9:16 render + captions ─▶ clip + metadata.json
       (SQLite)        (cached by file hash)   (cached, ranked 1–10)    (face-centered crop)
```

**Status: MVP + fan-edit module + GTA 6 Shorts mode.** Ingest → transcribe → detect → render works end to end.
Not built yet: the review dashboard, YouTube upload, and n8n.

## Quick start

```bash
bash scripts/install.sh                 # brew ffmpeg+ollama, venv, models, doctor
.venv/bin/python tests/test_e2e.py      # self-contained smoke test on a generated sample video
```

Clip your own footage:

```bash
source .venv/bin/activate
python -m clipfarm run /path/to/vod.mp4 --streamer somestreamer   # ingest + process
python -m clipfarm status
open data/clips/somestreamer/
```

In VS Code, open the folder, accept the recommended extensions, and use the
**Run and Debug** panel: *E2E test*, *Clip a video file…*, *Process pending queue* and *Doctor*.

## Commands

| Command | What it does |
|---|---|
| `python -m clipfarm doctor` | Check ffmpeg, Ollama and its model, Python modules, fonts, and the face model |
| `python -m clipfarm ingest FILE… --streamer NAME [--date YYYY-MM-DD]` | Hard-link/copy into `data/library/<streamer>/<date>/` and queue |
| `python -m clipfarm ingest-inbox --streamer NAME` | Move everything in `data/inbox/` into the library and queue it |
| `python -m clipfarm process [ID…] [--max-clips N]` | Process the given VODs, or every pending one (batch) |
| `python -m clipfarm run FILE… --streamer NAME` | `ingest` and `process` in one step |
| `python -m clipfarm run URL --streamer NAME` | Download with yt-dlp (≤1080p MP4, into `data/inbox/`), then ingest + process. **Only for content you own or have permission to clip.** The source URL is stored in the queue |
| `python -m clipfarm status` | Queue state per VOD (and the error if one failed) |

## Fan edits (beat-synced)

Turns a rendered clip into a music edit: an optional intro teaser, the captioned clip with music
ducked underneath, and an outro montage of other moments from the same VOD. Cuts land on the
beat, and the outro starts on the track's drop. Effects include white flashes, punch-ins,
slow motion, speed ramps, motion blur, color grading and vignette.

```bash
python -m clipfarm music add track.mp3 --license CC-BY-4.0 --source https://... --moods hype   # once per track
python -m clipfarm edit data/clips/<streamer>/<date>/<clip>.json --preset auto      # or hype, funny, ...
python -m clipfarm edit <clip>.json --preset hype --music music/track.mp3           # pick the track yourself
python -m clipfarm edit <clip>.json --plan-only                                     # inspect the plan without rendering
```

Output goes to `data/edits/<streamer>/<date>/`: the `.mp4`, a `.json` with metadata (including the
music credit line) and a `.plan.json` listing every cut, its effects and the music segments.

- **Presets:** `gaming-cinematic`, `funny`, `dramatic`, `hype`, `podcast`. `auto` picks one from the clip's category.
- **Settings:** [config/edit.yaml](config/edit.yaml) controls intro/outro length, music volume and
  ducking, caption style, transition intensity, cut frequency, slow-motion intensity, speed ramps,
  punch-in, color grades, resolution and fps. Every preset can override any of them.
- **Music:** only tracks in `music/` that have a license sidecar are used ([music/README.md](music/README.md)).
  Nothing is downloaded. With no licensed music the edit still renders, cutting to a fixed tempo.
- **Reused code:** beat tracking is adapted from easyedit (MIT); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
  and the repo review below.

### Repositories reviewed

| Repo | License | Used? |
|---|---|---|
| [blixvip/easyedit](https://github.com/blixvip/easyedit) | MIT | `beats.py` adapted (NumPy beat/drop detection). The rest isn't used: it auto-downloads movie scenes and songs from YouTube, renders through Node/Chrome, and needs a Claude Code or Codex login |
| [ZiadAbdelkarim/beat-synced-edit](https://github.com/ZiadAbdelkarim/beat-synced-edit) | MIT | Ideas only (flash cuts, FFmpeg color grades). Depends on librosa and scenedetect, which are heavy for this machine |
| [Merserk/BeatSync-Engine](https://github.com/Merserk/BeatSync-Engine) | **AGPL-3.0** | Studied, **no code copied**: AGPL would require releasing ClipFarm under AGPL. It's also Windows/CUDA-oriented |

## GTA 6 Shorts mode

Builds Shorts from your own footage, screenshots, commentary and research. There are three styles:

| Style | Input | What you get |
|---|---|---|
| **A: breakdown** | a video with your commentary | The AI picks the best 30–60 s, then: pauses cut out, punch-ins alternating, hook card, word-highlight captions, zoom and sound hit on key words, screenshot/footage B-roll, closing question |
| **B: cinematic** | any footage + images | Title card over slow motion, cuts on the beat that speed up after the drop, ramps, flashes, teal-orange grade, title slam, fade out |
| **C: facts** | `research/*.yaml` | Local TTS narration, a CONFIRMED / RUMOR / SPECULATION badge on every line, matched visuals, sources listed in the description |

```bash
python -m clipfarm gta6 init                 # create input/gta6/ + sources.yaml (see input/gta6/README.md)
python -m clipfarm gta6 make                 # all styles; or --styles A / B / C / AC ...; --cinematic-count 3
python -m clipfarm gta6 review               # list outputs + status
python -m clipfarm gta6 approve <folder>     # or: reject <folder> --note "too long"
```

Each output folder in `output/gta6/` has `video.mp4` (1080x1920), `thumbnail.jpg`, and
`metadata.json` (title, description with credits and sources, hashtags, review status). It also
has `project.json`: every shot, text card and sound, so you can see exactly what was built.
Everything starts as `pending` review.

Settings live in [config/gta6.yaml](config/gta6.yaml): fps (30/60), durations, TTS voice and speed,
caption style, badge colors, and per-style pacing, zoom, music and grade. Scene detection uses
PySceneDetect, beats use librosa (`beat_engine: numpy` is the lighter fallback), and sound
effects are generated locally into `assets/sfx/`.

## Layout

```
clipfarm/
  config.py       settings.yaml + CLIPFARM_* env overrides
  log.py          console logs + JSON lines (data/logs/clipfarm.jsonl)
  store.py        ingestion, file hashing, SQLite queue (vods, clips tables)
  transcribe.py   faster-whisper in a subprocess, word timestamps, cached
  highlights.py   windowed Ollama analysis → refine bounds → fit 20–60 s → dedupe/rank, cached
  render.py       face-centered 9:16 crop, slow zoom, word-highlight captions, one FFmpeg pass
  pipeline.py     process_vod(): runs the stages, writes metadata JSON, updates the queue
  doctor.py       dependency check
  download.py     yt-dlp URL downloads
  edit/           fan-edit generator: beats, music library, moments, plan (EDL), render
  gta6/           GTA 6 mode: sources, scene analysis, scripts, TTS, SFX, timeline compositor, styles
  llm.py          shared Ollama JSON-schema helper
config/edit.yaml       fan-edit settings + presets
config/gta6.yaml       GTA 6 templates
input/gta6/            GTA 6 footage, images, research, sources.yaml
music/                 your licensed tracks + .yaml license sidecars
config/settings.yaml   all tunables (models, clip lengths, caption style…)
scripts/install.sh     one-shot installer
tests/                 sample-video generator + end-to-end test
data/                  (gitignored) inbox, library, cache, clips, logs, clipfarm.db
docs/INSTALL.md        every install command, issues hit, and measured timings
```

## Configuration

Edit `config/settings.yaml`, or override any key in `.env` as `CLIPFARM_<SECTION>_<KEY>`,
e.g. `CLIPFARM_WHISPER_MODEL=medium` or `CLIPFARM_RENDER_ZOOM=0`.

Caches live in `data/cache/`. Transcripts are keyed by file hash and Whisper model.
Highlights are keyed by file hash, model, prompt version and the highlight settings,
so changing a highlight setting re-runs only the LLM step. Delete a cache file to force a redo.

## Output

For each clip, `data/clips/<streamer>/<date>/<vod>_<rank>_<slug>.mp4` plus a `.json` with
`title`, `description` (with hashtags), `hashtags`, `category`, `score`, `reason`, the source
timestamps and the transcript. Clips are also recorded in the `clips` table with
`review=pending` and `publish=unpublished`. Nothing gets published without approval.

## Known limitations / next steps

1. **Review dashboard** (local web UI): preview, approve/reject, regenerate, edit titles.
2. **YouTube Data API uploader**: OAuth per channel, scheduled publishing, approved clips only.
3. **Layouts for gaming streams**: a facecam-over-gameplay split layout. The current crop is a single
   static 9:16 window per clip, centered on the dominant face.
4. **Dynamic speaker tracking** within a clip; the crop is currently one position per clip.
5. **Per-streamer authorization list**: record which channels you have permission to clip and refuse URLs from anything else.
6. **Audio-energy signals**: laughter/volume spikes to boost highlight scores.
7. Face-centered cropping has only been tested on the synthetic sample, which has no faces.
   Validate it on real footage.
