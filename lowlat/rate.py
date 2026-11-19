"""CBR pacer with NO catch-up bursts.

Invariant: consecutive packet admissions are spaced so that
    next_admit_time = max(now, next_admit_time) + n / rate
i.e. the instantaneous rate NEVER exceeds `rate`; idle time is NOT saved up
(no deficit recovery). Therefore the delivered rate over ANY window is
<= rate + one_packet/window. This is what a real constant-bit-rate link does
and is what keeps a live stream's bandwidth graph flat and honest.
"""
from __future__ import annotations

import time


class TokenBucket:
    def __init__(self, rate_bps: float, burst: float | None = None):
        self.rate = float(rate_bps)
        # `burst` = small initial head-start (bytes) allowed to go out immediately
        self.burst = float(burst if burst is not None else rate_bps * 0.05)
        now = time.monotonic()
        self._next = now - self.burst / self.rate  # tiny head start
        self._t0 = now
        self._sent = 0.0

    def wait_take(self, n: float, timeout: float = 120.0) -> bool:
        if n <= 0:
            return True
        now = time.monotonic()
        start = self._next if self._next > now else now
        dur = n / self.rate
        self._next = start + dur
        delay = start - now
        if delay > 0:
            capped = min(delay, timeout)
            time.sleep(capped)
            if capped < delay:
                self._sent += n
                return False
        self._sent += n
        return True

    def try_take(self, n: float) -> bool:
        now = time.monotonic()
        if now >= self._next:
            self._next = max(now, self._next) + n / self.rate
            self._sent += n
            return True
        return False

    def available(self) -> float:
        now = time.monotonic()
        head = max(0.0, self._next - now)
        return head * 0.0 + (self.burst if head <= 0 else 0.0)

    def sent_bytes(self) -> float:
        return self._sent

    def elapsed(self) -> float:
        return time.monotonic() - self._t0
