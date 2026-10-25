"""A minimal per-domain rate limiter.

This is a research assistant, not a crawler: retrieval is sequential and the
limiter only has to guarantee a minimum spacing between two requests to the
same domain. `sleep_fn`/`time_fn` are injectable so tests can verify the
computed delay without actually sleeping.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional


class DomainRateLimiter:
    def __init__(
        self,
        *,
        min_delay_seconds: float,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_delay = max(0.0, min_delay_seconds)
        self._sleep = sleep_fn
        self._time = time_fn
        self._last_request_at: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str) -> float:
        """Block until `min_delay_seconds` have passed since the last request
        to `domain`. Returns the number of seconds actually slept (for tests)."""
        if self._min_delay <= 0:
            return 0.0

        with self._lock:
            now = self._time()
            last: Optional[float] = self._last_request_at.get(domain)
            if last is None:
                self._last_request_at[domain] = now
                return 0.0

            remaining = self._min_delay - (now - last)
            self._last_request_at[domain] = last + self._min_delay if remaining > 0 else now

        if remaining > 0:
            self._sleep(remaining)
            return remaining
        return 0.0
