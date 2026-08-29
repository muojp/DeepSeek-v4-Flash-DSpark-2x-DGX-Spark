"""Append-only JSONL writer with daily rotation and free-space guard."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class TraceWriter:
    """Thread-safe. ``write(rec)`` stamps ``t``/``m``/``host`` when absent,
    buffers, flushes every ``flush_s`` and fsyncs every ``fsync_s``.
    Rotates to ``<prefix>-YYYYMMDD.jsonl`` (UTC) on day change.
    Refuses to open when free space < ``min_free_start``; pauses writing
    (drops records, counts them) below ``min_free_run`` and resumes when
    space returns."""

    def __init__(self, directory: str | os.PathLike, prefix: str, host: str,
                 flush_s: float = 1.0, fsync_s: float = 10.0,
                 min_free_start: int = 2 << 30, min_free_run: int = 500 << 20,
                 clock=time.time, mono=time.monotonic, utcnow=None):
        self.dir = Path(directory)
        self.prefix = prefix
        self.host = host
        self.flush_s = flush_s
        self.fsync_s = fsync_s
        self.min_free_start = min_free_start
        self.min_free_run = min_free_run
        self._clock = clock
        self._mono = mono
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._fh = None
        self._day = None
        self._last_flush = 0.0
        self._last_fsync = 0.0
        self._last_space_check = 0.0
        self.paused = False
        self.dropped = 0
        self.written = 0
        self.bytes = 0
        self.dir.mkdir(parents=True, exist_ok=True)
        free = self.free_bytes()
        if free is not None and free < self.min_free_start:
            raise RuntimeError(f"refusing to start: {free / 2**20:.0f} MB free in {self.dir} "
                               f"(< {self.min_free_start / 2**20:.0f} MB)")
        self._open_for_today()

    # ── internals ────────────────────────────────────────────────────
    def free_bytes(self) -> int | None:
        try:
            st = os.statvfs(self.dir)
            return st.f_bavail * st.f_frsize
        except OSError:
            return None

    def _day_str(self) -> str:
        return self._utcnow().strftime("%Y%m%d")

    @property
    def path(self) -> Path:
        return self.dir / f"{self.prefix}-{self._day}.jsonl"

    def _open_for_today(self):
        day = self._day_str()
        if self._fh is not None and day == self._day:
            return
        self._close_locked()
        self._day = day
        self._fh = open(self.path, "ab", buffering=1 << 16)

    def _close_locked(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def _check_space(self, now: float):
        if now - self._last_space_check < 30:
            return
        self._last_space_check = now
        free = self.free_bytes()
        if free is None:
            return
        if self.paused and free >= self.min_free_run * 2:
            self.paused = False
        elif not self.paused and free < self.min_free_run:
            self.paused = True

    # ── public ───────────────────────────────────────────────────────
    def write(self, rec: dict) -> bool:
        rec.setdefault("t", self._clock())
        rec.setdefault("m", self._mono())
        rec.setdefault("host", self.host)
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
        data = line.encode("utf-8")
        with self._lock:
            now = self._mono()
            self._check_space(now)
            if self.paused:
                self.dropped += 1
                return False
            try:
                self._open_for_today()
                self._fh.write(data)
                self.written += 1
                self.bytes += len(data)
                if now - self._last_flush >= self.flush_s:
                    self._fh.flush()
                    self._last_flush = now
                    if now - self._last_fsync >= self.fsync_s:
                        os.fsync(self._fh.fileno())
                        self._last_fsync = now
                return True
            except OSError:
                self.dropped += 1
                return False

    def flush(self):
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except OSError:
                    pass

    def close(self):
        with self._lock:
            self._close_locked()

    def stats(self) -> dict:
        return {"written": self.written, "dropped": self.dropped, "bytes": self.bytes,
                "paused": self.paused, "path": str(self.path)}


def iter_jsonl(path: str | os.PathLike):
    """Yield parsed records; a truncated last line is skipped silently."""
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except ValueError:
                continue
