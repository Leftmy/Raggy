import asyncio
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """A process-local sliding-window limiter for single-instance deployments."""

    def __init__(self):
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._next_cleanup = 0.0

    async def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        async with self._lock:
            if now >= self._next_cleanup:
                expired_keys = [
                    event_key
                    for event_key, event_times in self._events.items()
                    if not event_times or event_times[-1] <= cutoff
                ]
                for event_key in expired_keys:
                    del self._events[event_key]
                self._next_cleanup = now + window_seconds
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            return True
