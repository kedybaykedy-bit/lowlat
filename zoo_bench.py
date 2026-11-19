"""Tile 'Me at the zoo' into a 1080p mosaic, stream it through LOWLAT, save everything.

For each grid (gx x gy) we build a 1920x1080 mosaic of the source video
(aspect-preserved, letterboxed inside each cell), then:
  input_1080p.mp4        – the mosaic fed to the codec
  lowlat_lossless.mp4    – LOWLAT decoded, bit-exact (lossless mode)
  lowlat_adaptive.mp4    – LOWLAT decoded under a 10 KB/s budget (degrade non-text)
  stats.json             – per-frame sizes, bitrates, cache hits, lossy tiles, timings

Usage:
  python zoo_bench.py --src out/zoo/jNQXAC9IVRw_ia.mp4 --grids 2x2,3x3,4x3
  python zoo_bench.py ... --max-frames 60         # quick
  python zoo_bench.py ... --only-lossless
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lowlat.ccodec import CCodec

OUT_W, OUT_H = 1920, 1080


def load_frames(src: str, max_frames: int = 0):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def build_mosaic(frame: np.ndarray, gx: int, gy: int) -> np.ndarray:
    """Place the 4:3 source in an gx*gy grid filling 1920x1080, aspect preserved,
    black bars inside cells, centered."""
    canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    cw, ch = OUT_W // gx, OUT_H // gy
    sh, sw = frame.shape[:2]
    scale = min(cw / sw, ch / sh)
    tw, th = max(1, int(sw * scale)), max(1, int(sh * scale))
    tile = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    for j in range(gy):
        for i in range(gx):
            cx, cy = i * cw, j * ch
            ox, oy = cx + (cw - tw) // 2, cy + (ch - th) // 2
            canvas[oy:oy + th, ox:ox + tw] = tile
    return canvas


def write_video(path: str, frames, fps: float):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def run_grid(src, frames_src, fps, gx, gy, outdir: Path, budget_bps: int,
             only_lossless: bool, start: float):
    tag = f"{gx}x{gy}"
    print(f"\n=== grid {tag} ({gx*gy} tiles) ===", flush=True)
    t0 = time.perf_counter()
    mosaics = [build_mosaic(f, gx, gy) for f in frames_src]
    print(f"  built {len(mosaics)} mosaics in {time.perf_counter()-t0:.1f}s", flush=True)

    h, w = OUT_H, OUT_W
    write_video(str(outdir / f"zoo_{tag}_input_1080p.mp4"), mosaics, fps)
    print(f"  wrote input video", flush=True)

    # ---------- lossless ----------
    enc = CCodec(h, w, cache_cap=1024)
    dec = CCodec(h, w, cache_cap=1024)
    t0 = time.perf_counter()
    p, _ = enc.encode(mosaics[0], keyframe=True, budget=-1)
    d0 = dec.decode(p)
    kf_bytes = len(p)
    lossless_sizes = []
    exact = np.array_equal(mosaics[0], d0)
    lost_frames = []  # stored to write output video
    lost_frames.append(d0)
    enc_ms = []; dec_ms = []
    for i in range(1, len(mosaics)):
        ta = time.perf_counter()
        p, _ = enc.encode(mosaics[i], keyframe=False, budget=-1)
        tb = time.perf_counter()
        r = dec.decode(p)
        tc = time.perf_counter()
        if not np.array_equal(mosaics[i], r):
            exact = False
        lost_frames.append(r)
        lossless_sizes.append(len(p))
        enc_ms.append(1000 * (tb - ta)); dec_ms.append(1000 * (tc - tb))
        if (i % 40 == 0):
            print(f"    lossless f{i}/{len(mosaics)} bytes={len(p)}", flush=True)
    wall = time.perf_counter() - t0
    write_video(str(outdir / f"zoo_{tag}_lowlat_lossless.mp4"), lost_frames, fps)
    dur = len(mosaics) / fps
    ll_total = kf_bytes + sum(lossless_sizes)
    ll_rate = ll_total / dur
    ll_steady = (sum(lossless_sizes) / max(1, len(lossless_sizes))) if lossless_sizes else 0
    ll_steady_rate = ll_steady * fps
    # amortized: keyframe spread over 60 s (a long-lived session) + steady inter rate
    ll_amort = (kf_bytes / 60.0 + ll_steady) * fps
    print(f"  LOSSLESS: exact={exact} total={ll_total/1024/1024:.2f}MB "
          f"kf={kf_bytes/1024:.0f}KB steady={ll_steady_rate/1024:.1f}KB/s "
          f"amort(60s)={ll_amort/1024:.1f}KB/s enc={np.mean(enc_ms):.0f}ms dec={np.mean(dec_ms):.0f}ms wall={wall:.0f}s", flush=True)

    stats = dict(
        grid=tag, tiles=gx * gy, resolution=[w, h], fps=fps, frames=len(mosaics),
        duration_s=round(dur, 2),
        lossless=dict(keyframe_bytes=kf_bytes,
                      total_bytes=ll_total,
                      rate_kbps_with_kf=round(ll_rate / 1024, 3),
                      steady_kbps=round(ll_steady_rate / 1024, 3),
                      amortized_60s_kbps=round(ll_amort / 1024, 3),
                      per_frame_mean=int(ll_steady),
                      bit_exact=bool(exact),
                      enc_ms_mean=round(float(np.mean(enc_ms)), 2) if enc_ms else 0,
                      dec_ms_mean=round(float(np.mean(dec_ms)), 2) if dec_ms else 0),
        adaptive=None,
    )

    # ---------- adaptive @ budget ----------
    if not only_lossless:
        perframe = max(64, int(budget_bps / fps))
        e2 = CCodec(h, w, cache_cap=1024)
        d2 = CCodec(h, w, cache_cap=1024)
        p, _ = e2.encode(mosaics[0], keyframe=True, budget=-1)  # keyframe always full
        ad_kf_bytes = len(p)
        ad_frames = [d2.decode(p)]
        ad_sizes = []
        lossy_counts = []
        nonlossy_viol = 0
        t0 = time.perf_counter()
        for i in range(1, len(mosaics)):
            p, _ = e2.encode(mosaics[i], keyframe=False, budget=perframe)
            emask = e2.lossy_mask().copy()
            r = d2.decode(p)
            # verify non-lossy tiles exact
            for ty in range(e2.nt):
                for tx in range(e2.nx):
                    if emask[ty, tx]:
                        continue
                    y0, x0 = ty * 32, tx * 32
                    y1, x1 = min(y0 + 32, h), min(x0 + 32, w)
                    if not np.array_equal(mosaics[i][y0:y1, x0:x1], r[y0:y1, x0:x1]):
                        nonlossy_viol += 1
            ad_frames.append(r)
            ad_sizes.append(len(p))
            lossy_counts.append(int(emask.sum()))
            if i % 40 == 0:
                print(f"    adaptive f{i}/{len(mosaics)} bytes={len(p)} lossy_tiles={int(emask.sum())}", flush=True)
        wall = time.perf_counter() - t0
        write_video(str(outdir / f"zoo_{tag}_lowlat_adaptive.mp4"), ad_frames, fps)
        ad_total = ad_kf_bytes + sum(ad_sizes)
        ad_rate = ad_total / dur
        ad_steady = (np.mean(ad_sizes) if ad_sizes else 0) * fps
        ad_amort = (ad_kf_bytes / 60.0 + (np.mean(ad_sizes) if ad_sizes else 0)) * fps
        stats["adaptive"] = dict(budget_kbps=round(budget_bps / 1024, 2),
                                 per_frame_budget=perframe,
                                 keyframe_bytes=ad_kf_bytes,
                                 total_bytes=ad_total,
                                 rate_kbps_with_kf=round(ad_rate / 1024, 3),
                                 steady_kbps=round(ad_steady / 1024, 3),
                                 amortized_60s_kbps=round(ad_amort / 1024, 3),
                                 per_frame_mean=int(np.mean(ad_sizes)) if ad_sizes else 0,
                                 lossy_tiles_mean=round(float(np.mean(lossy_counts)), 1) if lossy_counts else 0,
                                 lossy_tiles_max=int(np.max(lossy_counts)) if lossy_counts else 0,
                                 total_tiles=e2.nt * e2.nx,
                                 nonlossy_tile_violations=nonlossy_viol,
                                 enc_ms_mean=round(1000 * wall / max(1, len(ad_sizes)), 2),
                                 wall_s=round(wall, 1))
        print(f"  ADAPTIVE: steady={stats['adaptive']['steady_kbps']:.2f}KB/s "
              f"lossy_tiles_mean={np.mean(lossy_counts):.0f}/{e2.nt*e2.nx} nonlossy_viol={nonlossy_viol}", flush=True)

    stats["elapsed_s"] = round(time.perf_counter() - start, 1)
    with open(outdir / f"zoo_{tag}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  stats -> zoo_{tag}_stats.json", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/zoo/jNQXAC9IVRw_ia.mp4")
    ap.add_argument("--grids", default="2x2,3x3,4x3")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--budget", type=float, default=10 * 1024)
    ap.add_argument("--only-lossless", action="store_true")
    args = ap.parse_args()

    outdir = Path("out/zoo")
    outdir.mkdir(parents=True, exist_ok=True)
    frames, fps = load_frames(args.src, args.max_frames)
    print(f"loaded {len(frames)} frames @ {fps:.2f}fps from {args.src}", flush=True)

    start = time.perf_counter()
    summary = []
    for g in args.grids.split(","):
        gx, gy = map(int, g.split("x"))
        summary.append(run_grid(args.src, frames, fps, gx, gy, outdir, int(args.budget),
                                args.only_lossless, start))

    print("\n===== SUMMARY =====")
    for s in summary:
        l = s["lossless"]
        a = s["adaptive"]
        print(f"{s['grid']:4s} tiles={s['tiles']:2d}  lossless steady={l['steady_kbps']:8.1f}KB/s (exact={l['bit_exact']})"
              + (f"  adaptive steady={a['steady_kbps']:7.1f}KB/s lossy={a['lossy_tiles_mean']:.0f}/{a['total_tiles']} viol={a['nonlossy_tile_violations']}" if a else ""))
    with open(outdir / "zoo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
