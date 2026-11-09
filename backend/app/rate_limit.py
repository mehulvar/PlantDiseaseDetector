"""Small in-memory sliding-window rate limiter (per process)."""

from __future__ import annotations

import threading
import time
from collections import deque

MAX_TRACKED_CLIENTS = 10_000


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = self._clock()
        with self._lock:
            if client not in self._hits and len(self._hits) >= MAX_TRACKED_CLIENTS:
                self._evict(now)
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True

    def _evict(self, now: float) -> None:
        """Drop clients with no recent hits; if still full, drop the oldest-seen half."""
        stale = [c for c, h in self._hits.items() if not h or now - h[-1] >= self._window]
        for client in stale:
            del self._hits[client]
        if len(self._hits) >= MAX_TRACKED_CLIENTS:
            for client in list(self._hits)[: MAX_TRACKED_CLIENTS // 2]:
                del self._hits[client]
