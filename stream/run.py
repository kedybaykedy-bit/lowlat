#!/usr/bin/env python3
"""Run the recording through LOWLAT-S and report rate/quality."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import Budget, ScreenDecoder, ScreenEncoder  # noqa: E402


def _staleness(psnrs: list, fps: float, thr: float = 18.0) -> dict:
    """Windows where the decode is visibly stale (PSNR < thr)."""
    if not psnrs:
        return {}
    ps = np.asarray(psnrs)
    stale = ps < thr
    runs, cur, start = [], 0, 0
    for i, s in enumerate(stale):
        if s:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur:
                runs.append((start, cur))
            cur = 0
    if cur:
        runs.append((start, cur))
    long_runs = [(round(a / fps, 1), round(d / fps, 1))
                 for a, d in runs if d / fps > 1.0]
    return dict(threshold_db=thr,
                stale_pct=round(100 * float(stale.mean()), 1),
                stale_total_s=round(float(stale.sum()) / fps, 1),
                windows_gt1s=long_runs,
                longest_s=round(max((d for _, d in runs), default=0) / fps, 1))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 * 255.0 / m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="out/stream")
    ap.add_argument("--rate", type=float, default=15.0, help="KB/s target")
    ap.add_argument("--burst", type=float, default=160.0, help="KB initial burst (keyframe headroom)")
    ap.add_argument("--total", type=float, default=0, help="KB total cap (0=rate*duration+burst)")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n_total = min(n_total, args.max_frames)
    ok, f0 = cap.read()
    if not ok:
        raise SystemExit("cannot read")
    h, w = f0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    duration = n_total / fps
    total_kb = args.total or (args.rate * duration)   # strict average cap
    budget = Budget(total=total_kb * 1024, burst=args.burst * 1024,
                    rate=args.rate * 1024, duration=duration,
                    tail_rate=(args.rate * 1024) * 2 / 3)
    enc = ScreenEncoder(w, h, fps, budget)
    dec = ScreenDecoder(w, h)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    vw = None
    if args.save_video:
        vw = cv2.VideoWriter(str(outdir / "decoded.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    per_frame = []
    psnrs = []
    latencies = []          # frames of freeze before a change shows up
    pending_change = False
    t0 = time.perf_counter()
    n = 0
    while n < n_total:
        ok, frame = cap.read()
        if not ok:
            break
        pkt = enc.encode(frame, n)
        rec = dec.decode(pkt)
        if rec is None:
            rec = frame  # safety
        # wire model: SKIP means "hold" -> absence of update on the wire (0 bytes)
        wire = 0 if pkt[0] == 0 else len(pkt)
        per_frame.append(wire)
        if n > 0:
            p = psnr(frame, rec)
            psnrs.append(p)
            if pending_change:
                latencies.append(1)
                pending_change = False
        # detect if this source frame differed from what decoder shows but was frozen
        if pkt[0] == 0 and enc.sent_src is not None:
            d = np.abs(frame.astype(np.int16) - enc.sent_src.astype(np.int16)).max()
            if d > 3:
                pending_change = True
        if vw is not None:
            vw.write(rec)
        n += 1
        if n % 300 == 0:
            el = time.perf_counter() - t0
            print(f"  f{n}/{n_total}  {budget.spent/1024:.0f}KB/{total_kb:.0f}KB "
                  f"({100*budget.spent/budget.total:.0f}%)  {el:.0f}s", flush=True)
    cap.release()
    if vw:
        vw.release()
    wall = time.perf_counter() - t0

    sent = np.array(per_frame)
    # rate over the video duration (as if streamed live at fps)
    rate_kbps = sent.sum() / duration / 1024
    st = enc.stats
    result = dict(
        source=args.src, frames=n, fps=fps, duration_s=round(duration, 2),
        resolution=[w, h],
        budget_kb=dict(total=round(total_kb, 1), burst=args.burst, rate=args.rate),
        bytes_total=int(sent.sum()),
        rate_kbps=round(rate_kbps, 2),
        within_budget=bool(sent.sum() <= budget.total + 1),
        modes={k: st.get(k, 0) for k in ("key", "skip", "full", "crop", "tiles", "frozen")},
        quality_hist={str(k): v for k, v in sorted(st.get("q_hist", {}).items())},
        psnr_mean=round(float(np.mean(psnrs)), 2) if psnrs else None,
        psnr_p5=round(float(np.percentile(psnrs, 5)), 2) if psnrs else None,
        psnr_min=round(float(np.min(psnrs)), 2) if psnrs else None,
        staleness=_staleness(psnrs, fps),
        wall_s=round(wall, 1),
        encode_x_realtime=round(duration / wall, 1) if wall else None,
        per_frame_bytes=per_frame,
    )
    (outdir / "stats.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame_bytes"}, indent=2))


if __name__ == "__main__":
    main()
