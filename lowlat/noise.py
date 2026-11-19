"""Deterministic (seed-defined) noise — the ONLY way 'pure noise' fits in 10 KB/s.

A frame is a pure function of a 64-bit seed:  frame = N(seed, h, w).
Both endpoints share this function, so the sender transmits only the seed
(8 bytes/frame) and the receiver regenerates a *bit-identical* frame.

Lossless for noise *defined* by the generator (synthetic / PRNG / test-pattern).
Impossible for physical-entropy noise (sensor/urandom): full entropy, no codec
can shrink it (Shannon bound).

Counter-based splitmix64 over byte-chunks → reproducible on any machine and
fast enough for 1080p@30 (hashes 1/8 as many words as bytes).
"""
from __future__ import annotations

import numpy as np

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_IDX_CACHE: dict[int, np.ndarray] = {}


def _idx(n: int) -> np.ndarray:
    a = _IDX_CACHE.get(n)
    if a is None:
        a = np.arange(n, dtype=np.uint64)
        if len(_IDX_CACHE) > 8:
            _IDX_CACHE.clear()
        _IDX_CACHE[n] = a
    return a


def _splitmix64_inplace(z: np.ndarray) -> None:
    z += _GOLD
    z ^= z >> np.uint64(30)
    z *= _M1
    z ^= z >> np.uint64(27)
    z *= _M2
    z ^= z >> np.uint64(31)


def noise_frame(seed: int, h: int, w: int, salt: int = 0, out: np.ndarray | None = None) -> np.ndarray:
    """Deterministic uniform RGB noise (h,w,3) uint8. Identical on any machine."""
    nbytes = h * w * 3
    nwords = (nbytes + 7) // 8
    z = _idx(nwords).copy()
    base = np.uint64((int(seed) * 0x100000001B3 + int(salt)) & 0xFFFFFFFFFFFFFFFF)
    z ^= base
    _splitmix64_inplace(z)
    raw = z.view(np.uint8)[:nbytes]
    if out is None:
        out = np.empty((h, w, 3), dtype=np.uint8)
    out.reshape(-1)[:] = raw
    return out


class SeededNoiseSource:
    """Endless 1080p noise stream from a seed counter. frame(i)=N(base+i)."""

    def __init__(self, h: int, w: int, base_seed: int = 0):
        self.h, self.w = h, w
        self.base_seed = base_seed
        self._buf = np.empty((h, w, 3), dtype=np.uint8)

    def seed_for(self, i: int) -> int:
        return (int(self.base_seed) + int(i)) & 0xFFFFFFFFFFFFFFFF

    def frame_into(self, i: int, out: np.ndarray | None = None) -> np.ndarray:
        return noise_frame(self.seed_for(i), self.h, self.w, out=out if out is not None else self._buf)

    def frame(self, i: int) -> np.ndarray:
        return noise_frame(self.seed_for(i), self.h, self.w)
