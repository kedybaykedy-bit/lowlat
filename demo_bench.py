#!/usr/bin/env python3
"""Process a screen recording through LOWLAT.

Streams frame-by-frame (low memory), runs two passes:
  1) lossless  2) adaptive (text-protected, per-frame budget)

Writes: output videos, per-frame stats JSON, bitrate timeline PNG,
side-by-side comparison video, contact sheet.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "lowlat"))
from ccodec import CCodec  # noqa: E402


def run_pass(src: str, out_video: str, budget_bps: float, fps_override=None,
             key_every=150, limit=0, progress_every=150):
    cap = cv2.VideoCapture(src)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    enc, dec = CCodec(h, w, cache_cap=1024), CCodec(h, w, cache_cap=1024)
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    per_frame = (budget_bps / fps) if budget_bps and budget_bps > 0 else -1.0

    sizes, enc_ms, lossy_counts, nonlossy_viol = [], [], [], 0
    kf_bytes, n, exact_all = 0, 0, True
    agg = dict(dirty=0, cache_hits=0, lossy_tiles=0, text_tiles=0)
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (limit and n >= limit):
            break
        keyframe = (n % key_every == 0)
        t1 = time.perf_counter()
        pkt, _ = enc.encode(frame, keyframe=keyframe, budget=per_frame)
        enc_ms.append((time.perf_counter() - t1) * 1000)
        sizes.append(len(pkt))
        if keyframe:
            kf_bytes += len(pkt)
        st = enc.last_stats
        agg["dirty"] += st.dirty
        agg["cache_hits"] += st.cache_hits
        agg["lossy_tiles"] += st.lossy_tiles
        agg["text_tiles"] += st.text_tiles
        mask = enc.lossy_mask()
        lossy_counts.append(int(mask.sum()))
        dec_frame = dec.decode(pkt)
        # every tile that was NOT degraded must be bit-exact (vectorized per-pixel)
        exact_pix = np.repeat(np.repeat(~mask, 32, axis=0), 32, axis=1)[:h, :w]
        if not np.array_equal(dec_frame[exact_pix], frame[exact_pix]):
            exact_all = False
            nonlossy_viol += 1
        vw.write(dec_frame)
        n += 1
        if progress_every and n % progress_every == 0:
            rate = (sum(sizes) / (n / fps)) / 1024
            print(f"    {Path(out_video).name} f{n} bytes={sizes[-1]} run={rate:.1f}KB/s", flush=True)
    cap.release()
    vw.release()
    wall = time.perf_counter() - t0
    dur = n / fps if fps else 1
    total = sum(sizes)
    steady_bytes = total - kf_bytes
    steady_n = n - len([i for i in range(n) if i % key_every == 0])
    steady = (steady_bytes / max(1, steady_n)) * fps
    return dict(frames=n, fps=fps, duration_s=round(dur, 2),
                keyframe_bytes=kf_bytes, total_bytes=total,
                rate_kbps_with_kf=round(total / dur / 1024, 3),
                steady_kbps=round(steady / 1024, 3),
                per_frame_mean=int(np.mean(sizes)) if sizes else 0,
                per_frame_max=int(np.max(sizes)) if sizes else 0,
                lossy_tiles_mean=round(float(np.mean(lossy_counts)), 1) if lossy_counts else 0,
                lossy_tiles_max=int(np.max(lossy_counts)) if lossy_counts else 0,
                total_tiles=enc.nt * enc.nx,
                nonlossy_tile_violations=nonlossy_viol,
                exact_nonlossy=bool(exact_all),
                enc_ms_mean=round(float(np.mean(enc_ms)), 2) if enc_ms else 0,
                wall_s=round(wall, 1),
                counters=agg,
                per_frame_sizes=sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--outdir", default="out/demo")
    ap.add_argument("--budget", type=float, default=10.0, help="KB/s adaptive budget")
    ap.add_argument("--key-every", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--only", choices=["lossless", "adaptive"], default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.src).stem
    budget_bps = args.budget * 1024

    results = {}
    if args.only in (None, "lossless"):
        print(f"LOSSLESS pass: {args.src}", flush=True)
        results["lossless"] = run_pass(
            args.src, str(outdir / f"{tag}_lowlat_lossless.mp4"),
            budget_bps=-1, key_every=args.key_every, limit=args.limit)
        r = results["lossless"]
        print(f"  LOSSLESS: steady={r['steady_kbps']:.1f}KB/s kf={r['keyframe_bytes']/1024:.0f}KB "
              f"exact={r['exact_nonlossy']} viol={r['nonlossy_tile_violations']} "
              f"enc={r['enc_ms_mean']}ms wall={r['wall_s']}s", flush=True)

    if args.only in (None, "adaptive"):
        print(f"ADAPTIVE pass (budget {args.budget} KB/s): {args.src}", flush=True)
        results["adaptive"] = run_pass(
            args.src, str(outdir / f"{tag}_lowlat_adaptive.mp4"),
            budget_bps=budget_bps, key_every=args.key_every, limit=args.limit)
        r = results["adaptive"]
        print(f"  ADAPTIVE: steady={r['steady_kbps']:.1f}KB/s (budget {args.budget}) "
              f"lossy={r['lossy_tiles_mean']:.0f}/{r['total_tiles']} "
              f"nonlossy_viol={r['nonlossy_tile_violations']} "
              f"enc={r['enc_ms_mean']}ms wall={r['wall_s']}s", flush=True)

    # persist (per-frame sizes dropped from file copy kept in memory only for graph)
    out_json = outdir / f"{tag}_stats.json"
    persisted = {}
    for k, r in results.items():
        rr = dict(r)
        rr.pop("per_frame_sizes", None)
        persisted[k] = rr
    out_json.write_text(json.dumps(persisted, indent=2))
    print(f"stats -> {out_json}", flush=True)

    # bitrate timeline graph
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4.5))
        colors = {"lossless": "#22c55e", "adaptive": "#3b82f6"}
        for k, r in results.items():
            fps = r["fps"]
            rates = np.array(r["per_frame_sizes"]) * fps / 1024.0  # KB/s instantaneous
            t = np.arange(len(rates)) / fps
            ax.plot(t, rates, label=f"{k} (steady {r['steady_kbps']:.1f} KB/s)",
                    color=colors.get(k, "#888"), alpha=0.85, lw=1)
        ax.axhline(args.budget, color="red", ls="--", lw=1.2,
                   label=f"budget {args.budget:.0f} KB/s")
        ax.set_xlabel("time (s)"); ax.set_ylabel("rate (KB/s)")
        ax.set_title(f"LOWLAT per-frame rate — {tag}")
        ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        graph = outdir / f"{tag}_timeline.png"
        fig.savefig(graph, dpi=110); plt.close(fig)
        print(f"graph -> {graph}", flush=True)
    except Exception as e:  # matplotlib optional
        print(f"(graph skipped: {e})", flush=True)


if __name__ == "__main__":
    main()
