#!/usr/bin/env python3
"""Change-to-display latency: how long until a source change shows up
in the decode.

Per 32x32 tile we track freshness (decode matches current source within
tol). A stale run = the viewer stares at wrong content for that long.
Events are bucketed by how many tiles changed in the source at the run
start: <=12 tiles ~= interactive (typing, status), larger = bulk.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

H, W, T = 1080, 1920, 32
TH, TW = 34, 60
VALID = np.full((TH, TW), T * T, np.float64)
VALID[33, :] = (H - 33 * T) * T  # last row is 24 px tall


def tile_mean(d: np.ndarray) -> np.ndarray:
    """Mean of a HxW diff image per tile (padded to grid)."""
    dp = np.zeros((TH * T, TW * T), np.int16)
    dp[:H, :W] = d
    s = dp.reshape(TH, T, TW, T).sum(axis=(1, 3))
    return s / VALID


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoded", required=True)
    ap.add_argument("--src", default="out/demo/screen_60s.mp4")
    ap.add_argument("--tol", type=float, default=8.0)
    ap.add_argument("--small", type=int, default=12,
                    help="<= this many changed tiles => interactive event")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dc = cv2.VideoCapture(args.decoded)
    sc = cv2.VideoCapture(args.src)
    fresh = np.ones((TH, TW), bool)
    run = np.zeros((TH, TW), np.int32)
    run_nchg = np.full((TH, TW), -1, np.int32)  # change size at run START
    events: list[tuple[float, int, float]] = []  # (dur_s, nchg_at_start, t0)
    prev_src = None
    t = 0.0
    n = 0
    while True:
        okd, fd = dc.read()
        okc, fc = sc.read()
        if not (okd and okc):
            break
        # freshness of decode vs current source
        d_dec = np.abs(fd.astype(np.int16) - fc.astype(np.int16)).max(axis=2)
        now_fresh = tile_mean(d_dec) <= args.tol
        # source change size this frame
        nchg = -1
        if prev_src is not None:
            d_src = np.abs(fc.astype(np.int16) - prev_src.astype(np.int16)).max(axis=2)
            nchg = int((tile_mean(d_src) > args.tol).sum())
        prev_src = fc.copy()

        # stale runs: fresh -> stale starts a run, stale -> fresh ends it
        started = fresh & ~now_fresh
        ended = ~fresh & now_fresh
        for ty, tx in zip(*np.where(started)):
            run[ty, tx] = 1
            run_nchg[ty, tx] = max(nchg, 0)
        for ty, tx in zip(*np.where(ended)):
            dur = max(int(run[ty, tx]), 1) / 30.0
            events.append((dur, int(run_nchg[ty, tx]), t - dur))
            run[ty, tx] = 0
            run_nchg[ty, tx] = -1
        # continue stale runs (started ones already at 1)
        run = np.where(now_fresh, 0, np.maximum(run, 1) + (~fresh & ~now_fresh))
        fresh = now_fresh
        t += 1 / 30.0
        n += 1
    dc.release()
    sc.release()

    # unclosed runs at end
    for ty, tx in zip(*np.where(~fresh)):
        dur = max(int(run[ty, tx]), 1) / 30.0
        events.append((dur, -1, t - dur))

    if not events:
        print("no events")
        return
    durs = np.array([e[0] for e in events])
    small = np.array([e[0] for e in events if 0 <= e[1] <= args.small])
    large = np.array([e[0] for e in events if e[1] > args.small])

    def desc(a: np.ndarray, label: str) -> None:
        if a.size == 0:
            print(f"{label}: none")
            return
        print(f"{label}: n={a.size}  p50={np.percentile(a,50):.2f}s  "
              f"p90={np.percentile(a,90):.2f}s  p99={np.percentile(a,99):.2f}s  "
              f"max={a.max():.2f}s  mean={a.mean():.2f}s")

    print(f"frames={n}  events={len(events)}")
    desc(durs, "ALL stale runs                 ")
    desc(small, f"interactive (<= {args.small} tile change)")
    desc(large, "bulk (large change)            ")
    buckets = [0.0, 0.1, 0.3, 1.0, 3.0, 6.0]
    hist, _ = np.histogram(durs, buckets + [1e9])
    labels = ["<0.1s", "0.1-0.3s", "0.3-1s", "1-3s", "3-6s", ">6s"]
    print("histogram:", ", ".join(f"{l}:{c}" for l, c in zip(labels, hist)))
    worst = sorted(events, key=lambda e: -e[0])[:10]
    print("worst 10: " + "; ".join(
        f"{d:.1f}s@t={t0:.1f}(chg={c})" for d, c, t0 in worst))
    if args.out:
        Path(args.out).write_text(
            "\n".join(f"{d:.3f}\t{c}\t{t0:.2f}" for d, c, t0 in events))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
