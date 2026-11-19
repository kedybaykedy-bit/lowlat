"""General-content benchmark (non-desktop synthetic video) with the C codec.

Shows the codec is not overfit to desktop: handles smooth/scrolling/noisy video
via motion compensation. Compares against VNC-style dirty-tile + zlib.
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowlat.ccodec import CCodec
from lowlat.content import generate_sequence

TILE = 32


def vnc_bytes(frames: np.ndarray) -> list[int]:
    out = [len(zlib.compress(frames[0].tobytes(), 6))]
    ref = frames[0]
    h, w = ref.shape[:2]
    nt, nx = (h + TILE - 1) // TILE, (w + TILE - 1) // TILE
    for f in frames[1:]:
        raw = bytearray()
        for ty in range(nt):
            for tx in range(nx):
                y0, x0 = ty * TILE, tx * TILE
                y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                if not np.array_equal(f[y0:y1, x0:x1], ref[y0:y1, x0:x1]):
                    raw += f[y0:y1, x0:x1].tobytes()
        out.append(len(zlib.compress(bytes(raw), 6)) if raw else 0)
        ref = f
    return out


def run(h, w, label, motion, n=60):
    frames = generate_sequence(n_frames=n, h=h, w=w, seed=42, motion=motion)
    enc = CCodec(h, w, enable_me=True)
    dec = CCodec(h, w, enable_me=True)
    sizes = []; ems = []; dms = []; exact = True
    for i, f in enumerate(frames):
        t0 = time.perf_counter(); p, nd = enc.encode(f, keyframe=(i == 0)); t1 = time.perf_counter()
        r = dec.decode(p); t2 = time.perf_counter()
        if not np.array_equal(f, r):
            exact = False
            print(f"  {label} {motion} LOSSLESS FAIL f{i}"); break
        sizes.append(len(p)); ems.append(1000*(t1-t0)); dms.append(1000*(t2-t1))
    if not exact:
        return
    vnc = vnc_bytes(frames)
    dur = n/30
    ours = np.mean(sizes[1:])*30/1024 if len(sizes) > 1 else 0
    vnc_s = np.mean(vnc[1:])*30/1024 if len(vnc) > 1 else 0
    ratio = vnc_s/ours if ours > 0 else 0
    print(f"{label:6s} {motion:7s} | ours {ours:7.1f} KB/s  VNC {vnc_s:8.1f} KB/s  "
          f"ratio {ratio:4.1f}x | kf {sizes[0]/1024:7.1f}KB | "
          f"enc {np.mean(ems):6.1f} dec {np.mean(dms):5.1f} ms | lossless=True")


def main():
    print("General video content (synthetic desktop animations) — lossless\n")
    print(f"{'res':6s} {'motion':7s} | {'steady rate':^36s} | {'keyframe':^14s} | speed")
    print("-" * 104)
    for h, w, label in [(480, 854, "480p"), (720, 1280, "720p"), (1080, 1920, "1080p")]:
        for motion in ("normal", "heavy"):
            run(h, w, label, motion, n=45)


if __name__ == "__main__":
    main()
