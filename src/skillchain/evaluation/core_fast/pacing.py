from __future__ import annotations

import threading
import time
from collections.abc import Callable


class StartPacer:
    """Thread-safe even spacing for starts of real provider calls."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self._interval
        if delay:
            self._sleeper(delay)


__all__ = ["StartPacer"]
