#!/usr/bin/env python3
"""Build comparison artifacts for a LOWLAT-S run:
   side-by-side video, contact sheet, rate timeline, quality timeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def psnr(a, b) -> float:
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 * 255.0 / m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original recording")
    ap.add_argument("--decoded", required=True, help="decoded.mp4 from run.py")
    ap.add_argument("--stats", required=True, help="stats.json from run.py")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(args.stats).read_text())

    c1 = cv2.VideoCapture(args.src)
    c2 = cv2.VideoCapture(args.decoded)
    fps = c1.get(cv2.CAP_PROP_FPS) or 30
    n = min(int(c1.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(c2.get(cv2.CAP_PROP_FRAME_COUNT)))
    ok, f0 = c1.read()
    h, w = f0.shape[:2]
    c1.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---- per-frame PSNR + side-by-side video ----
    sw, sh = w // 2, h // 2
    vw = cv2.VideoWriter(str(outdir / "compare_sbs.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (sw * 2, sh))
    psnrs = []
    frame_bytes = stats.get("per_frame_bytes", [])
    idxs_sheet = [int(n * f) for f in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
    sheet_frames = {i: [] for i in idxs_sheet}
    for i in range(n):
        ok1, a = c1.read()
        ok2, b = c2.read()
        if not (ok1 and ok2):
            break
        p = psnr(a, b)
        psnrs.append(p)
        row = np.hstack([
            cv2.resize(a, (sw, sh)),
            cv2.resize(b, (sw, sh)),
        ])
        cv2.putText(row, f"SOURCE  f{i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(row, f"LOWLAT-S  PSNR={p:.1f}dB", (sw + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 120), 2, cv2.LINE_AA)
        vw.write(row)
        if i in sheet_frames:
            sheet_frames[i].append(cv2.resize(row, (sw, sh // 2)))
    c1.release()
    c2.release()
    vw.release()

    # ---- contact sheet ----
    rows = [sheet_frames[i] for i in idxs_sheet if sheet_frames[i]]
    if rows:
        # each entry is a row of 7 images stacked vertically per source? simplify:
        pass
    # simpler: horizontal strip of 7 sbs thumbnails
    thumbs = []
    for i in idxs_sheet:
        if sheet_frames.get(i):
            thumbs.append(sheet_frames[i][0])
    if thumbs:
        thumbs = [cv2.resize(t, (480, int(t.shape[0] * 480 / t.shape[1])))
                  for t in thumbs]
        rh = max(t.shape[0] for t in thumbs)
        thumbs = [t if t.shape[0] == rh else
                  cv2.copyMakeBorder(t, 0, rh - t.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
                  for t in thumbs]
        rows = [np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)]
        maxw = max(r.shape[1] for r in rows)
        rows = [r if r.shape[1] == maxw else
                cv2.copyMakeBorder(r, 0, 0, 0, maxw - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for r in rows]
        sheet = np.vstack(rows)
        cv2.imwrite(str(outdir / "contact_sheet.png"), sheet)

    # ---- rate + quality timeline ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        if frame_bytes:
            t = np.arange(len(frame_bytes)) / fps
            ax1.plot(t, np.array(frame_bytes) / 1024, lw=0.7, color="#3b82f6",
                     label="bytes/frame")
            # sliding 1s rate
            win = int(fps)
            if len(frame_bytes) >= win:
                k = np.ones(win) / win
                rate = np.convolve(frame_bytes, k, mode="valid") * fps / 1024
                ax1.plot(t[win - 1:], rate, lw=1.4, color="#ef4444",
                         label="rate (1s window)")
            ax1.axhline(stats["budget_kb"]["rate"], color="green", ls="--",
                        lw=1, label=f"target {stats['budget_kb']['rate']} KB/s")
            ax1.set_ylabel("KB/s")
            ax1.legend(loc="upper right", fontsize=9)
            ax1.set_title(f"LOWLAT-S rate — {stats['rate_kbps']:.2f} KB/s avg "
                          f"({stats['bytes_total']/1024:.0f} KB total, "
                          f"budget {stats['budget_kb']['total']:.0f} KB)")
            ax1.grid(alpha=0.3)
        if psnrs:
            ax2.plot(np.arange(len(psnrs)) / fps, psnrs, lw=0.7, color="#22c55e")
            ax2.set_ylabel("PSNR (dB)")
            ax2.set_xlabel("time (s)")
            ax2.grid(alpha=0.3)
            ax2.set_title(f"decoded vs source — mean {np.mean(psnrs):.1f} dB, "
                          f"p5 {np.percentile(psnrs, 5):.1f} dB")
        fig.tight_layout()
        fig.savefig(outdir / "timeline.png", dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"(timeline skipped: {e})")

    print(json.dumps({
        "psnr_mean": round(float(np.mean(psnrs)), 2) if psnrs else None,
        "psnr_p5": round(float(np.percentile(psnrs, 5)), 2) if psnrs else None,
        "rate_kbps": stats.get("rate_kbps"),
        "bytes_total": stats.get("bytes_total"),
        "modes": stats.get("modes"),
        "artifacts": [p.name for p in sorted(outdir.iterdir())],
    }, indent=2))


if __name__ == "__main__":
    main()
