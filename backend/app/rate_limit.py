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
            if len(self._hits) > MAX_TRACKED_CLIENTS:
                self._hits.clear()
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True
