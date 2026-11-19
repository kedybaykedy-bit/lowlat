"""Final benchmark: lossless layered-cursor codec vs VNC-style (dirty-rect + zlib).

Compares achieved bitrate, encode/decode time, and size vs a VNC/ZRLE-like baseline
that bakes the cursor into pixels and zlib-compresses changed 32x32 tiles.
"""
from __future__ import annotations

import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowlat.ccodec import CCodec
from lowlat.vnc_scenario import generate_vnc, generate_vnc_layered, composite_cursor

TILE = 32


def vnc_bytes(frames: np.ndarray) -> list[int]:
    """VNC/ZRLE-like: keyframe zlib; inter = changed tiles (raw RGB) + zlib."""
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
                a = f[y0:y1, x0:x1]
                b = ref[y0:y1, x0:x1]
                if not np.array_equal(a, b):
                    raw += bytes([tx & 0xFF, ty & 0xFF])
                    raw += a.tobytes()
        out.append(len(zlib.compress(bytes(raw), 6)) if raw else 0)
        ref = f
    return out


def bench(h, w, label, activity, rate_bps=10 * 1024, n=90):
    base_frames, cursors = generate_vnc_layered(n_frames=n, h=h, w=w, seed=1, activity=activity)
    true_frames = generate_vnc(n_frames=n, h=h, w=w, seed=1, activity=activity)
    enc = CCodec(h, w, enable_me=True)
    dec = CCodec(h, w, enable_me=True)

    sizes = []; ems = []; dms = []; exact = True
    for i in range(n):
        t0 = time.perf_counter()
        p, nd = enc.encode(base_frames[i], keyframe=(i == 0))
        t1 = time.perf_counter()
        cpos = cursors[i]
        packet = p + (b"\x00" if cpos is None else b"\x01" + struct.pack("<HH", cpos[0], cpos[1]))
        rec = dec.decode(p)
        rec_c = None if packet[len(p)] == 0 else struct.unpack("<HH", packet[len(p)+1:len(p)+5])
        dms.append(1000 * (time.perf_counter() - t1))
        disp = composite_cursor(rec, rec_c)
        if not np.array_equal(disp, true_frames[i]):
            exact = False; break
        sizes.append(len(packet)); ems.append(1000 * (t1 - t0))

    t0 = time.perf_counter()
    vnc = vnc_bytes(true_frames)
    vnc_wall = time.perf_counter() - t0

    dur = n / 30.0
    ours_steady = np.mean(sizes[1:]) * 30 / 1024
    vnc_steady = np.mean(vnc[1:]) * 30 / 1024
    ours_total = sum(sizes) / 1024
    vnc_total = sum(vnc) / 1024
    ratio = vnc_steady / ours_steady if ours_steady > 0 else 0
    print(f"{label:6s} {activity:8s} | ours {ours_steady:6.2f} KB/s  VNC {vnc_steady:8.1f} KB/s  "
          f"ratio {ratio:6.1f}x | kf {sizes[0]/1024:6.1f}KB vs {vnc[0]/1024:6.1f}KB | "
          f"enc {np.mean(ems):5.1f}ms dec {np.mean(dms):4.1f}ms | lossless={exact}")
    return ours_steady, vnc_steady, exact


def main():
    print("Lossless desktop streaming: our codec vs VNC-style baseline")
    print("(same content; VNC bakes cursor into pixel tiles, ours uses cursor layer)\n")
    print(f"{'res':6s} {'activity':8s} | {'steady rate':^34s} | {'keyframe':^20s} | speed | check")
    print("-" * 118)
    configs = [
        (480, 854, "480p", "idle"), (480, 854, "480p", "typical"),
        (720, 1280, "720p", "typical"),
        (1080, 1920, "1080p", "typical"),
        (1440, 2560, "2K", "idle"), (1440, 2560, "2K", "typical"),
        (1440, 2560, "2K", "busy"),
    ]
    all_ok = True
    for h, w, label, act in configs:
        try:
            _, _, ok = bench(h, w, label, act)
            all_ok = all_ok and ok
        except Exception as e:
            print(f"{label} {act}: ERROR {e}")
    print("-" * 118)
    print("all lossless:", all_ok)


if __name__ == "__main__":
    main()
