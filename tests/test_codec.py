import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lowlat.ccodec import CCodec
from lowlat.vnc_scenario import generate_vnc_layered, generate_article
from lowlat.rate import TokenBucket
from lowlat.noise import noise_frame


def _roundtrip(frames, h, w, budget=-1, cache=512):
    enc = CCodec(h, w, cache_cap=cache)
    dec = CCodec(h, w, cache_cap=cache)
    p, _ = enc.encode(frames[0], keyframe=True, budget=-1)
    assert np.array_equal(dec.decode(p), frames[0])
    violations = 0
    for i in range(1, len(frames)):
        p, _ = enc.encode(frames[i], keyframe=False, budget=budget)
        emask = enc.lossy_mask().copy()
        r = dec.decode(p)
        for ty in range(enc.nt):
            for tx in range(enc.nx):
                if emask[ty, tx]:
                    continue
                y0, x0 = ty * 32, tx * 32
                y1, x1 = min(y0 + 32, h), min(x0 + 32, w)
                if not np.array_equal(frames[i][y0:y1, x0:x1], r[y0:y1, x0:x1]):
                    violations += 1
    return violations, enc, dec


def test_lossless_desktop():
    h, w = 480, 854
    bf, _ = generate_vnc_layered(n_frames=20, h=h, w=w, seed=1, activity="typical")
    v, _, _ = _roundtrip(bf, h, w, budget=-1)
    assert v == 0


def test_adaptive_never_touches_lossless_tiles():
    h, w = 480, 854
    bf, _ = generate_vnc_layered(n_frames=20, h=h, w=w, seed=1, activity="busy")
    v, _, _ = _roundtrip(bf, h, w, budget=100)  # tight -> forces degradation
    assert v == 0


def test_cache_reduces_size_and_is_lossless():
    import cv2
    h, w = 240, 320
    fa = np.full((h, w, 3), (40, 60, 90), np.uint8)
    cv2.rectangle(fa, (20, 20), (250, 200), (200, 200, 200), -1)
    fb = fa.copy()
    cv2.rectangle(fb, (20, 20), (250, 200), (160, 200, 160), -1)
    seq = [fa, fa, fb, fb, fa, fb, fa, fb, fa, fb]

    def total(cap):
        e = CCodec(h, w, cache_cap=cap)
        d = CCodec(h, w, cache_cap=cap)
        t = 0
        for i, f in enumerate(seq):
            p, _ = e.encode(f, keyframe=(i == 0), budget=-1)
            assert np.array_equal(d.decode(p), f)
            t += len(p)
        return t

    assert total(512) < total(0) * 0.6  # cache should >~1.6x smaller


def test_global_scroll_lossless():
    h, w = 480, 854
    art, _ = generate_article(n_frames=30, h=h, w=w, seed=0, scroll_speed=2.0)
    v, _, _ = _roundtrip(art, h, w, budget=-1)
    assert v == 0


def test_pacer_rate():
    import time
    b = TokenBucket(50_000, burst=256)
    t0 = time.perf_counter()
    for _ in range(20):
        b.wait_take(250)
    el = time.perf_counter() - t0
    rate = 20 * 250 / el
    assert 0.7 * 50_000 <= rate <= 1.2 * 50_000


def test_seeded_noise_deterministic():
    a = noise_frame(123, 240, 320)
    b = noise_frame(123, 240, 320)
    c = noise_frame(124, 240, 320)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
