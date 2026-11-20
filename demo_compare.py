#!/usr/bin/env python3
"""Build side-by-side video + contact sheet: INPUT | LOSSLESS | ADAPTIVE."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def label(img, text, color):
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original recording")
    ap.add_argument("--lossless", required=True)
    ap.add_argument("--adaptive", required=True)
    ap.add_argument("--outdir", default="out/demo")
    ap.add_argument("--scale", type=float, default=0.5, help="panel scale")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tag = Path(args.src).stem
    paths = [args.src, args.lossless, args.adaptive]
    caps = [cv2.VideoCapture(p) for p in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
    w = int(caps[0].get(3) * args.scale)
    h = int(caps[0].get(4) * args.scale)
    colors = [(0, 255, 255), (0, 255, 120), (80, 160, 255)]
    names = ["INPUT", "LOSSLESS", "ADAPTIVE"]

    sbs_path = outdir / f"{tag}_compare_sbs.mp4"
    vw = cv2.VideoWriter(str(sbs_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 3, h))
    n = 0
    while True:
        frames = []
        for c in caps:
            ok, f = c.read()
            if not ok:
                frames = None
                break
            frames.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
        if frames is None or (args.limit and n >= args.limit):
            break
        row = np.hstack([label(f.copy(), t, col) for f, t, col in zip(frames, names, colors)])
        vw.write(row)
        n += 1
    vw.release()
    for c in caps:
        c.release()
    print(f"sbs -> {sbs_path} ({n} frames)")

    # contact sheet: sample 6 time points
    idxs = [int(n * f) for f in (0.08, 0.24, 0.40, 0.56, 0.72, 0.88) if int(n * f) < n]
    rows = []
    for p, name, col in zip(paths, names, colors):
        c = cv2.VideoCapture(p)
        cells = []
        for i in idxs:
            c.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = c.read()
            if ok:
                f = cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA)
                cells.append(label(f, name, col))
        c.release()
        if cells:
            rows.append(np.hstack(cells))
    if rows:
        sheet = np.vstack(rows)
        sheet_path = outdir / f"{tag}_contact_sheet.png"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"sheet -> {sheet_path} {sheet.shape}")


if __name__ == "__main__":
    main()
