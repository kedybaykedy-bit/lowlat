"""Comprehensive benchmark: all scenarios × resolutions, with cache + adaptive."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowlat.ccodec import CCodec
from lowlat.vnc_scenario import generate_vnc_layered, generate_article, generate_stress


def measure(frames, h, w, budget, label):
    enc = CCodec(h, w, cache_cap=512)
    dec = CCodec(h, w, cache_cap=512)
    t0 = time.perf_counter()
    p, _ = enc.encode(frames[0], keyframe=True, budget=-1)
    dec.decode(p)
    kf = len(p)
    sizes = []
    hits = []
    approx = []
    viol = 0
    ecost = 0.0
    for i in range(1, len(frames)):
        ta = time.perf_counter()
        p, _ = enc.encode(frames[i], keyframe=False, budget=budget)
        emask = enc.lossy_mask().copy()
        tb = time.perf_counter()
        r = dec.decode(p)
        tc = time.perf_counter()
        ecost += (tb - ta); 
        sizes.append(len(p)); hits.append(enc.last_stats.cache_hits); approx.append(int(emask.sum()))
        # verify non-lossy tiles exact
        for ty in range(enc.nt):
            for tx in range(enc.nx):
                if emask[ty, tx]:
                    continue
                y0, x0 = ty * 32, tx * 32
                y1, x1 = min(y0 + 32, h), min(x0 + 32, w)
                if not np.array_equal(frames[i][y0:y1, x0:x1], r[y0:y1, x0:x1]):
                    viol += 1
    s = np.array(sizes)
    wall = time.perf_counter() - t0
    rate = (s.mean() * 30 / 1024) if s.size else 0
    print(f"  {label:26s} {rate:7.2f} KB/s  p95={np.percentile(s,95)/1024:5.1f}KB  "
          f"kf={kf/1024:6.1f}KB  hits={np.mean(hits):4.1f}  approx={np.mean(approx):5.1f}  "
          f"viol={viol}  enc={1000*ecost/max(1,len(s)):4.1f}ms")
    return dict(rate=rate, viol=viol, kf=kf)


def main():
    print("=" * 104)
    print("LOWLAT benchmark — lossless-by-default, text-protected adaptive degradation")
    print("=" * 104)
    for h, w, res in [(480, 854, "480p"), (720, 1280, "720p"), (1080, 1920, "1080p")]:
        print(f"\n### {res} ({w}x{h}) ###")
        for act in ("idle", "typical", "busy"):
            bf, _ = generate_vnc_layered(n_frames=90, h=h, w=w, seed=1, activity=act)
            measure(bf, h, w, -1, f"desktop {act} lossless")
            measure(bf, h, w, int(10 * 1024 / 30), f"desktop {act} @10KB/s")
        art, _ = generate_article(n_frames=90, h=h, w=w, seed=0, scroll_speed=1.5)
        measure(art, h, w, -1, "article scroll lossless")
        measure(art, h, w, int(10 * 1024 / 30), "article scroll @10KB/s")
        st, _ = generate_stress(n_frames=60, h=h, w=w, seed=0, video_frac=0.30)
        measure(st, h, w, -1, "video+text lossless")
        measure(st, h, w, int(10 * 1024 / 30), "video+text @10KB/s")


if __name__ == "__main__":
    main()
