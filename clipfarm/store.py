"""Ingestion + SQLite processing queue.

VODs are copied (or hard-linked) into library/<streamer>/<YYYY-MM-DD>/ and a
row is added to `vods`. Each stage advances `status`:
  queued -> transcribed -> analyzed -> rendered   (or failed, with `error`)
Rendered clips live in `clips` with their own review/publish status.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".flv", ".webm", ".ts", ".m4v"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS vods (
    id          INTEGER PRIMARY KEY,
    streamer    TEXT NOT NULL,
    stream_date TEXT NOT NULL,
    path        TEXT NOT NULL UNIQUE,
    file_hash   TEXT NOT NULL,
    source      TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    id           INTEGER PRIMARY KEY,
    vod_id       INTEGER NOT NULL REFERENCES vods(id),
    path         TEXT NOT NULL UNIQUE,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    score        REAL,
    title        TEXT,
    review       TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
    publish      TEXT NOT NULL DEFAULT 'unpublished', -- unpublished | scheduled | published
    channel      TEXT,
    youtube_id   TEXT,
    created_at   TEXT NOT NULL
);
"""


def file_hash(path: Path, chunk: int = 4 << 20) -> str:
    """Fast content fingerprint: size + first/last 4 MB. Good enough to key caches on."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Queue:
    def __init__(self, db_path: Path, library: Path):
        self.library = library
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def ingest(self, src: Path, streamer: str, stream_date: date | None = None,
               move: bool = False, source: str | None = None) -> int:
        """Place `src` in the library and enqueue it. Returns the vod id (existing if already ingested)."""
        if src.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Not a supported video file: {src}")
        digest = file_hash(src)
        row = self.db.execute("SELECT id FROM vods WHERE file_hash = ?", (digest,)).fetchone()
        if row:
            log.info("already ingested", extra={"vod_id": row["id"], "src": str(src)})
            return row["id"]

        stream_date = stream_date or date.fromtimestamp(src.stat().st_mtime)
        dest_dir = self.library / _slug(streamer) / stream_date.isoformat()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if move:
            shutil.move(src, dest)
        elif not dest.exists():
            try:
                os.link(src, dest)          # no extra disk space if same volume
            except OSError:
                shutil.copy2(src, dest)

        cur = self.db.execute(
            "INSERT INTO vods (streamer, stream_date, path, file_hash, source, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (streamer, stream_date.isoformat(), str(dest), digest, source or str(src), _now(), _now()),
        )
        self.db.commit()
        log.info("ingested", extra={"vod_id": cur.lastrowid, "streamer": streamer, "path": str(dest)})
        return cur.lastrowid

    def get(self, vod_id: int) -> sqlite3.Row:
        return self.db.execute("SELECT * FROM vods WHERE id = ?", (vod_id,)).fetchone()

    def pending(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM vods WHERE status NOT IN ('rendered', 'failed') ORDER BY id"
        ).fetchall()

    def set_status(self, vod_id: int, status: str, error: str | None = None) -> None:
        self.db.execute("UPDATE vods SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                        (status, error, _now(), vod_id))
        self.db.commit()

    def add_clip(self, vod_id: int, path: Path, start: float, end: float,
                 score: float, title: str) -> int:
        cur = self.db.execute(
            "INSERT OR REPLACE INTO clips (vod_id, path, start_s, end_s, score, title, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vod_id, str(path), start, end, score, title, _now()),
        )
        self.db.commit()
        return cur.lastrowid

    def all_vods(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM vods ORDER BY id").fetchall()


def open_queue(cfg: dict) -> Queue:
    return Queue(cfg["paths"]["db"], cfg["paths"]["library"])


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip().lower()) or "unknown"
