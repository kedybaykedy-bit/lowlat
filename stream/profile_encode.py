#!/usr/bin/env python3
"""Profile ScreenEncoder.encode / _encode_inner hot path.

Runs cProfile over the first N frames (default 250) of out/demo/screen_60s.mp4
with the same Budget as the 60s run, prints top 25 functions by cumulative
and by total time, plus a few standalone micro-benchmarks of the suspect
components (webp tile/frame encode, dirty detection, probes, row-runs).

Usage:  ~/.venv/bin/python stream/profile_encode.py [n_frames]
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import (Budget, ScreenEncoder, TILE, DIRTY_D, _webp,  # noqa: E402
                   _webp_dec)

SRC = Path(__file__).resolve().parent.parent / "out/demo/screen_60s.mp4"
W, H, FPS = 1920, 1080, 30.0


def load_frames(n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(SRC))
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def make_encoder() -> ScreenEncoder:
    # exactly the Budget from the 60s run (rate=15KB/s, duration=60)
    return ScreenEncoder(W, H, FPS,
                         Budget(total=900 * 1024, burst=160 * 1024,
                                rate=15 * 1024, duration=60.0,
                                tail_rate=10 * 1024))


def run_encode(frames: list[np.ndarray]) -> tuple[list[bytes], float, ScreenEncoder]:
    enc = make_encoder()
    t0 = time.perf_counter()
    pkts = [enc.encode(f, i) for i, f in enumerate(frames)]
    dt = time.perf_counter() - t0
    return pkts, dt, enc


def profiled(frames: list[np.ndarray]):
    enc = make_encoder()
    pr = cProfile.Profile()
    pr.enable()
    pkts = [enc.encode(f, i) for i, f in enumerate(frames)]
    pr.disable()
    return pkts, pr, enc


def top_stats(pr, n: int = 25) -> tuple[str, str]:
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(n)
    cum = s.getvalue()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(n)
    tot = s.getvalue()
    return cum, tot


def micro(frames: list[np.ndarray], pkts: list[bytes]) -> None:
    """Standalone timings of the individual suspects (median of repeats)."""
    rng = np.random.default_rng(0)
    f0, f1 = frames[0], frames[1]

    def med(fn, reps: int) -> float:
        xs = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(xs))

    tile = f0[:TILE, :TILE].copy()
    noise_tile = np.clip(tile.astype(np.int16) +
                         rng.integers(-6, 7, tile.shape), 0, 255).astype(np.uint8)
    blob = _webp(f0, 82)

    print("=== micro-benchmarks (median ms / call) ===")
    print(f"webp q85  32x32 tile (static)     : {med(lambda: _webp(tile, 85), 200):7.3f}")
    print(f"webp q85  32x32 tile (changed)    : {med(lambda: _webp(noise_tile, 85), 200):7.3f}")
    print(f"webp q82  1920x1080 full frame    : {med(lambda: _webp(f0, 82), 10):7.3f}")
    print(f"webp q85  1920x1080 full frame    : {med(lambda: _webp(f0, 85), 10):7.3f}")
    print(f"webp q82  1920x540 band (crop)    : {med(lambda: _webp(f0[:540], 82), 10):7.3f}")
    print(f"webp q82  480x270 quarter probe   : {med(lambda: _webp(f0[::4, ::4], 82), 30):7.3f}")
    print(f"webp decode 1920x1080 (blob only) : {med(lambda: _webp_dec(blob), 20):7.3f}")

    a = f0.astype(np.int16)
    b = f1.astype(np.int16)
    print(f"astype int16 x2 (1920x1080)       : {med(lambda: (f0.astype(np.int16), f1.astype(np.int16)), 50):7.3f}")
    print(f"int16 subtract + abs (no max)     : {med(lambda: np.abs(a - b), 50):7.3f}")
    print(f"CURRENT: np.abs(int16).max(axis=2): {med(lambda: np.abs(a - b).max(axis=2), 50):7.3f}")

    # candidate replacements for the diff+max hotspot (must yield the SAME mask/values)
    def cand_absdiff_max3():
        d = cv2.absdiff(f0, f1)
        return np.maximum(np.maximum(d[:, :, 0], d[:, :, 1]), d[:, :, 2])
    def cand_absdiff_cv2max():
        d = cv2.absdiff(f0, f1)
        return cv2.max(cv2.max(d[:, :, 0], d[:, :, 1]), d[:, :, 2])
    def cand_absdiff_npmax():
        return cv2.absdiff(f0, f1).max(axis=2)
    print(f"  cand cv2.absdiff + np.maximum x3 : {med(cand_absdiff_max3, 50):7.3f}")
    print(f"  cand cv2.absdiff + cv2.max x3    : {med(cand_absdiff_cv2max, 50):7.3f}")
    print(f"  cand cv2.absdiff + np.max(axis=2): {med(cand_absdiff_npmax, 50):7.3f}")
    # equivalence: the candidate must yield bit-identical `d` to the current code
    d_cur = np.abs(a - b).max(axis=2)
    for _nm, _fn in (("np.maximum x3", cand_absdiff_max3),
                     ("cv2.max x3", cand_absdiff_cv2max)):
        _dc = _fn()
        print(f"  EQUIVALENT d ({_nm}): {np.array_equal(d_cur, _dc)}"
              f"  max|delta|={int(np.abs(d_cur.astype(np.int32) - _dc.astype(np.int32)).max())}")

    # row-runs: current `j in dirty` list scan vs set membership
    e_rs = make_encoder()
    e_rs.sent_src = f0.copy()
    for n_tiles in (200, 1000, 1859):
        dirty_rs = list(range(0, e_rs.ntiles, max(1, e_rs.ntiles // n_tiles)))[:n_tiles]
        dset = set(dirty_rs)
        def with_set(dset=dset, dirty_rs=dirty_rs):
            runs, seen = [], set()
            tw = e_rs.tw
            for i in dirty_rs:
                if i in seen:
                    continue
                ty, tx = divmod(i, tw)
                run = [i]
                seen.add(i)
                for dd in (1, -1):
                    tx2 = tx + dd
                    while 0 <= tx2 < tw:
                        j = ty * tw + tx2
                        if j in dset and j not in seen:
                            run.append(j)
                            seen.add(j)
                            tx2 += dd
                        else:
                            break
                runs.append(run)
        t_cur = med(lambda: e_rs._row_runs(dirty_rs), 20)
        t_set = med(with_set, 20)
        print(f"_row_runs n={n_tiles:4d}: current list-scan {t_cur:7.3f} ms"
              f"  | set-membership {t_set:7.3f} ms")

    d = np.abs(a - b).max(axis=2)
    big = d > DIRTY_D
    th, tw = -(-H // TILE), -(-W // TILE)
    ph, pw = th * TILE, tw * TILE
    pad_b = np.zeros((ph, pw), bool)
    pad_d = np.zeros((ph, pw), np.int16)
    pad_b[:H, :W] = big
    pad_d[:H, :W] = d
    print(f"tile counts reshape+sum (bool)    : {med(lambda: pad_b.reshape(th, TILE, tw, TILE).sum(axis=(1, 3)), 50):7.3f}")
    print(f"tile sums where+reshape (int16)    : {med(lambda: np.where(pad_b, pad_d, 0).reshape(th, TILE, tw, TILE).sum(axis=(1, 3)), 50):7.3f}")

    # full dirty detection as coded in ScreenEncoder._dirty_tiles
    def full_dirty():
        dd = np.abs(f0.astype(np.int16) - f1.astype(np.int16)).max(axis=2)
        bb = dd > DIRTY_D
        pb = np.zeros((ph, pw), bool)
        pb[:H, :W] = bb
        pd = np.zeros((ph, pw), np.int16)
        pd[:H, :W] = dd
        cnt = pb.reshape(th, TILE, tw, TILE).sum(axis=(1, 3))
        s = np.where(pb, pd, 0).reshape(th, TILE, tw, TILE).sum(axis=(1, 3))
        return np.where(cnt >= 6, s / np.maximum(cnt, 1), 0.0)
    print(f"FULL _dirty_tiles-equivalent       : {med(full_dirty, 50):7.3f}")

    # probes: fresh encoder state each rep (bypass the _probe_for identity cache)
    e = make_encoder()
    e.sent_src = f0.copy()
    dirty_p = list(range(0, e.ntiles, 33))[:60]   # 60 dirty tiles spread over the frame
    def probe_rep():
        e._probe_for = None
        e._build_probes(f1, dirty_p)
    print(f"_build_probes (60 dirty tiles)     : {med(probe_rep, 30):7.3f}")

    # row-runs over realistic dirty lists (note: `j in dirty` is a list scan)
    e0 = make_encoder()
    e0.sent_src = f0.copy()
    dirty_rr = list(range(0, e0.ntiles, 7))[:200]
    print(f"_row_runs (200 dirty tiles)        : {med(lambda: e0._row_runs(dirty_rr), 100):7.3f}")
    dirty_rr2 = list(range(0, e0.ntiles, 2))[:1000]
    print(f"_row_runs (1000 dirty tiles)       : {med(lambda: e0._row_runs(dirty_rr2), 20):7.3f}")

    # encode-loop fixed overhead: python call + budget + stats
    print(f"(for reference) mean pkt size first {len(frames)} frames: "
          f"{np.mean([len(p) for p in pkts]) if pkts else float('nan'):.0f} B")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    micro_only = "--micro" in sys.argv
    n = int(args[0]) if args else 250
    frames = load_frames(n)
    print(f"loaded {len(frames)} frames {frames[0].shape}")
    if micro_only:
        micro(frames, [])
        return

    pkts, plain, enc = run_encode(frames)
    print(f"\n=== plain run: {len(frames)} frames in {plain:.2f}s "
          f"= {1000*plain/len(frames):.1f} ms/frame "
          f"({len(frames)/plain:.1f} fps) ===")
    st = enc.stats
    print("modes:", {k: st.get(k, 0) for k in
                     ("key", "skip", "frozen", "crop", "tiles")},
          "bytes:", st["bytes"], "dirty_sum:", st["dirty_sum"])

    pkts2, pr, enc2 = profiled(frames)
    assert [len(p) for p in pkts] == [len(p) for p in pkts2], "profiled run diverged!"
    print(f"cProfile total: {pstats.Stats(pr).total_tt:.2f}s")

    cum, tot = top_stats(pr, 80)
    print("\n=== ALL functions by CUMULATIVE time ===")
    print(cum)
    print("=== ALL functions by TOTAL (self) time ===")
    print(tot)

    micro(frames, pkts)


if __name__ == "__main__":
    main()
