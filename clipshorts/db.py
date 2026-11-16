"""SQLite storage (WAL, stdlib sqlite3, single connection guarded by a lock).

Workload is tiny, so synchronous calls are fine. Safe to call from the worker
thread and the event loop.
"""

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from .models import Clip, Job, Metadata, Publication

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT NOT NULL,
    slug TEXT NOT NULL,
    channel TEXT,
    title TEXT,
    game TEXT,
    duration_s REAL,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    language TEXT NOT NULL DEFAULT 'auto',
    layout_type TEXT,
    layout_confidence REAL,
    layout_override TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
CREATE INDEX IF NOT EXISTS idx_clips_slug ON clips(slug);
CREATE TABLE IF NOT EXISTS renders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    layout_type TEXT,
    params_json TEXT NOT NULL DEFAULT '{}',
    path TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clip_metadata (
    clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    caption TEXT NOT NULL DEFAULT '',
    hashtags_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'manual',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS publications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    url TEXT,
    published_at TEXT NOT NULL,
    views_24h INTEGER,
    rating TEXT,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_publications_clip ON publications(clip_id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    ref_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, id);
"""

CLIP_FIELDS = {
    "channel", "title", "game", "duration_s", "status", "stage", "progress", "error",
    "language", "layout_type", "layout_confidence", "layout_override",
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class DB:
    def __init__(self, path: Path | str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    # -- settings -------------------------------------------------------
    def get_setting(self, key: str, default=None):
        r = self._one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(r["value"]) if r else default

    def set_setting(self, key: str, value) -> None:
        self._exec(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    # -- clips ----------------------------------------------------------
    def create_clip(
        self,
        source_url: str,
        slug: str,
        channel: str | None = None,
        language: str = "auto",
        layout_override: str | None = None,
    ) -> Clip:
        ts = now_iso()
        cur = self._exec(
            "INSERT INTO clips(source_url, slug, channel, language, layout_override, "
            "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
            (source_url, slug, channel, language, layout_override, ts, ts),
        )
        clip = self.get_clip(cur.lastrowid)
        assert clip
        return clip

    def get_clip(self, clip_id: int) -> Clip | None:
        r = self._one("SELECT * FROM clips WHERE id=?", (clip_id,))
        return Clip(**dict(r)) if r else None

    def find_clip_by_slug(self, slug: str) -> Clip | None:
        r = self._one("SELECT * FROM clips WHERE slug=? ORDER BY id DESC LIMIT 1", (slug,))
        return Clip(**dict(r)) if r else None

    def list_clips(self, status: str | None = None) -> list[Clip]:
        if status:
            rows = self._all("SELECT * FROM clips WHERE status=? ORDER BY id DESC", (status,))
        else:
            rows = self._all("SELECT * FROM clips ORDER BY id DESC")
        return [Clip(**dict(r)) for r in rows]

    def update_clip(self, clip_id: int, **fields) -> Clip | None:
        sets, args = ["updated_at=?"], [now_iso()]
        for k, v in fields.items():
            if k not in CLIP_FIELDS:
                raise ValueError(k)
            sets.append(f"{k}=?")
            args.append(v)
        self._exec(f"UPDATE clips SET {', '.join(sets)} WHERE id=?", (*args, clip_id))
        return self.get_clip(clip_id)

    def delete_clip(self, clip_id: int) -> bool:
        """Remove the clip row, its renders (cascade) and its finished/failed jobs."""
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE ref_id=?", (clip_id,))
            n = self._conn.execute("DELETE FROM clips WHERE id=?", (clip_id,)).rowcount
            self._conn.commit()
        return n > 0

    # -- renders --------------------------------------------------------
    def add_render(self, clip_id: int, layout_type: str, params: dict, path: str) -> int:
        cur = self._exec(
            "INSERT INTO renders(clip_id, layout_type, params_json, path, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (clip_id, layout_type, json.dumps(params, sort_keys=True), path, now_iso()),
        )
        return cur.lastrowid

    def latest_render(self, clip_id: int) -> dict | None:
        r = self._one("SELECT * FROM renders WHERE clip_id=? ORDER BY id DESC LIMIT 1",
                      (clip_id,))
        if not r:
            return None
        d = dict(r)
        d["params"] = json.loads(d.pop("params_json") or "{}")
        return d

    # -- metadata (phase 4) ---------------------------------------------
    def get_metadata(self, clip_id: int) -> Metadata | None:
        r = self._one("SELECT * FROM clip_metadata WHERE clip_id=?", (clip_id,))
        if not r:
            return None
        return Metadata(title=r["title"], caption=r["caption"],
                        hashtags=json.loads(r["hashtags_json"] or "[]"),
                        source=r["source"], updated_at=r["updated_at"])

    def set_metadata(self, clip_id: int, title: str, caption: str, hashtags: list[str],
                     source: str) -> Metadata:
        self._exec(
            "INSERT INTO clip_metadata(clip_id, title, caption, hashtags_json, source, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(clip_id) DO UPDATE SET "
            "title=excluded.title, caption=excluded.caption, "
            "hashtags_json=excluded.hashtags_json, source=excluded.source, "
            "updated_at=excluded.updated_at",
            (clip_id, title, caption, json.dumps(hashtags, ensure_ascii=False), source, now_iso()),
        )
        m = self.get_metadata(clip_id)
        assert m
        return m

    # -- publications ---------------------------------------------------
    def add_publication(self, clip_id: int, platform: str, url: str | None = None,
                        published_at: str | None = None, views_24h: int | None = None,
                        rating: str | None = None, note: str | None = None) -> Publication:
        cur = self._exec(
            "INSERT INTO publications(clip_id, platform, url, published_at, views_24h, rating, "
            "note) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (clip_id, platform, url, published_at or now_iso(), views_24h, rating, note),
        )
        p = self.get_publication(cur.lastrowid)
        assert p
        return p

    def get_publication(self, pub_id: int) -> Publication | None:
        r = self._one("SELECT * FROM publications WHERE id=?", (pub_id,))
        return Publication(**dict(r)) if r else None

    def list_publications(self, clip_id: int) -> list[Publication]:
        rows = self._all("SELECT * FROM publications WHERE clip_id=? ORDER BY id", (clip_id,))
        return [Publication(**dict(r)) for r in rows]

    def update_publication(self, pub_id: int, **fields) -> Publication | None:
        allowed = {"url", "views_24h", "rating", "note"}
        sets, args = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(k)
            sets.append(f"{k}=?")
            args.append(v)
        if sets:
            self._exec(f"UPDATE publications SET {', '.join(sets)} WHERE id=?", (*args, pub_id))
        return self.get_publication(pub_id)

    def delete_publication(self, pub_id: int) -> bool:
        return self._exec("DELETE FROM publications WHERE id=?", (pub_id,)).rowcount > 0

    def rating_report(self) -> list[dict]:
        """channel x layout x rating counts, the only reporting phase 4 has."""
        rows = self._all(
            "SELECT COALESCE(c.channel, '') AS channel, COALESCE(c.layout_type, '') AS layout, "
            "p.rating AS rating, COUNT(*) AS n FROM publications p "
            "JOIN clips c ON c.id = p.clip_id WHERE p.rating IS NOT NULL "
            "GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
        )
        table: dict[tuple[str, str], dict] = {}
        for r in rows:
            row = table.setdefault(
                (r["channel"], r["layout"]),
                {"channel": r["channel"], "layout": r["layout"], "good": 0, "ok": 0, "flop": 0},
            )
            row[r["rating"]] = r["n"]
        return list(table.values())

    # -- jobs -----------------------------------------------------------
    @staticmethod
    def _job(r: sqlite3.Row) -> Job:
        return Job(**dict(r))

    def enqueue_job(self, kind: str, ref_id: int) -> Job:
        ts = now_iso()
        cur = self._exec(
            "INSERT INTO jobs(kind, ref_id, status, created_at, updated_at) "
            "VALUES (?, ?, 'queued', ?, ?)",
            (kind, ref_id, ts, ts),
        )
        r = self._one("SELECT * FROM jobs WHERE id=?", (cur.lastrowid,))
        assert r
        return self._job(r)

    def get_job(self, job_id: int) -> Job | None:
        r = self._one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._job(r) if r else None

    def list_jobs(self, status: str | None = None) -> list[Job]:
        if status:
            rows = self._all("SELECT * FROM jobs WHERE status=? ORDER BY id", (status,))
        else:
            rows = self._all("SELECT * FROM jobs ORDER BY id")
        return [self._job(r) for r in rows]

    def claim_next_job(self) -> Job | None:
        """Oldest queued job -> running (attempts+1). One heavy job at a time is the
        worker's contract; this just hands out the next one atomically."""
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if not r:
                return None
            self._conn.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE id=?",
                (now_iso(), r["id"]),
            )
            self._conn.commit()
            return self._job(
                self._conn.execute("SELECT * FROM jobs WHERE id=?", (r["id"],)).fetchone()
            )

    def finish_job(self, job_id: int, ok: bool, error: str | None = None) -> None:
        self._exec(
            "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
            ("done" if ok else "failed", error, now_iso(), job_id),
        )

    # -- recovery -------------------------------------------------------
    def recover(self) -> dict[str, int]:
        """Startup recovery: anything left 'running' (killed mid-job) goes back to 'queued'.
        Stage artifacts are cached, so the re-run resumes where it stopped."""
        with self._lock:
            c = self._conn.execute(
                "UPDATE clips SET status='queued', updated_at=? WHERE status='running'",
                (now_iso(),),
            ).rowcount
            j = self._conn.execute(
                "UPDATE jobs SET status='queued', updated_at=? WHERE status='running'",
                (now_iso(),),
            ).rowcount
            self._conn.commit()
        return {"clips": c, "jobs": j}
