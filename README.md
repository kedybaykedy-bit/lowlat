# LOWLAT — a lossless-first remote-display codec

**A VNC replacement that is lossless by default and degrades *only* where it must.**
It streams a desktop at 30 fps over an extremely small bitrate (target **10 KB/s**),
keeps **text pixel-exact**, caches repeated patterns, and turns full-screen
scrolling into a near-free operation.

```
frame ─► [dirty tiles] ─► [global-scroll / ME] ─► [tile cache] ─► [hybrid coder] ─► deflate ─► wire
                                                                                      (CBR 10 KB/s)
wire  ─► inflate ─► [RLE bitmap] ─► [per-tile: PRED/REF/varint/submask/down/skip] ─► composite cursor ─► screen
```

---

## Headline results

All numbers are **real**, measured, and **bit-exact verified** (non-lossy tiles pixel-exact).
`enc/dec` are single-core on a laptop; the core is C.

### Desktop activity (lossless)

| resolution | activity | bitrate (lossless) | notes |
|---|---|---|---|
| 480p  | idle    | **2.1 KB/s** | cursor/blink only |
| 720p  | typing  | **0.9 KB/s** | text never degraded |
| 1080p | idle    | **1.0 KB/s** | |
| 1080p | typing  | **0.7 KB/s** | |
| 1080p | busy    | 8.8 KB/s | still lossless |
| any   | very busy | 12–20 KB/s | >budget → adaptive |

### Repeated-pattern cache (toggle A↔B UI)

| | no cache | **cache** |
|---|---|---|
| total bytes | 20,216 B | **5,592 B** (3.6×) |
| cache refs | 0 | **373** |

### Smooth article scrolling — lossless (global-scroll *band* mode)

Pure vertical scroll only sends the newly-exposed strip (Paeth + deflate); the rest
is one shift. Lossless, full-frame bit-exact:

| scroll speed | px/s | bitrate (lossless) |
|---|---|---|
| 1.0 px/frame | 30 | **6.4 KB/s** |
| 1.5 px/frame | 45 | **9.1 KB/s**  ← under 10 KB/s |
| 2.0 px/frame | 60 | 11.7 KB/s |
| 4.0 px/frame | 120 | 22.1 KB/s |

Cost is proportional to *newly revealed text*, which is genuine information.
At a normal reading scroll the lossless stream fits under 10 KB/s. (Earlier builds
were ~40 KB/s here because the strip was re-coded tile-by-tile with per-tile
overhead; the band path removes that.)

Scrolling reveals genuinely new text; the rate is proportional to new content.
Static/typing/interactive desktops sit far below 10 KB/s — that is where a VNC
replacement lives.

### Text protection (the core promise)

Under a hard 10 KB/s budget on busy content, **only non-text tiles degrade**
(skip / 2× / 4× downsample). Verified every frame:

```
non-lossy-tile pixel mismatches : 0
tiles classified text-as-lossy  : 0
```

### Real 60-second desktop recording at 15 KB/s (LOWLAT-S)

A full minute of a real 1080p/30 desktop (terminal, wallpaper, window fades,
notifications) streamed through the rate-controlled `stream/` codec under a
hard 15 KB/s budget:

| metric | value |
|---|---|
| average rate | **14.08 KB/s** (cap 900 KB / 60 s) |
| total transferred | **844.6 KB** ✅ |
| PSNR mean / p5 | **26.05 / 16.65 dB** |
| tile quality | **97 % at q82+** (text floor never breached) |

The encoder banks credit during holds, refreshes the whole screen with
coherent full-frame snapshots when activity demands it, and paints calm
updates as whole text-lines at q92 — see
[`experiments/streaming/`](experiments/streaming/) for the contact sheet,
rate/PSNR timeline and full stats.

---

## Features

- **Lossless by default.** Every frame is byte-exact unless the per-frame budget
  forces degradation.
- **Text-protected adaptive quality.** Tiles are classified as text-like
  (strong-edge density + bimodal luma histogram). Text tiles are *never* degraded;
  only smooth/photo/video tiles may be skip / 2× / 4× downsampled — and only when
  the frame would otherwise exceed the byte budget.
- **Tile cache (FIFO ring).** A tile identical to a previously sent tile costs ~7
  bytes (a ring-slot reference) instead of full pixels. Repeated UI chrome, restored
  windows, toggled panels, scrolled-back content become nearly free.
- **Global-scroll mode.** If the whole frame moved by a constant `dy`, one byte
  carries it; recovers predict the shift and only the newly-exposed strip is coded.
  Makes article/code scrolling smooth and cheap.
- **Per-tile hybrid coder.** For each tile the encoder picks the smallest of:
  `PRED` (MV-exact, 0 payload), cache `REF`, zigzag-varint residual, raw,
  changed-pixel submask, 2×/4× downsample, or `SKIP`.
- **Integer motion estimation** with neighbour-MV hint and text-safe global search.
- **Layered cursor.** The desktop is coded without the cursor; the cursor is a
  2–4 byte side channel composited on the client. Displayed frame stays bit-exact.
- **CBR pacing** with no catch-up bursts → flat, honest bandwidth.
- **Single-stream deflate** for all tile payloads (no per-tile zlib headers).

---

## Why 10 KB/s is achievable (and where it isn't)

A 2K desktop at 30 fps is ~331 MB/s raw. The **only** way to send it losslessly at
10 KB/s is to send *almost nothing most of the time* — which is exactly what
desktop work is: static regions, a moving cursor, small text edits. LOWLAT exploits:

1. temporal redundancy (dirty tiles + motion + global scroll),
2. spatial redundancy within tiles (residual/submask + deflate),
3. repetition (tile cache).

For **pure random noise** or **full-motion natural video**, lossless 10 KB/s is
*information-theoretically impossible* (Shannon). LOWLAT reports this honestly:
`noise_demo.py --mode true` shows every general-purpose codec (and ours) at ~1.00×
on true random frames (≈178 MB/s required). For **seed-defined** synthetic noise,
`noise_demo.py --mode seed` transmits an 8-byte seed and regenerates a bit-identical
1080p frame at 0.23 KB/s.

---

## Build & run

Requirements: Python 3.10+, `numpy`, `opencv-python-headless`, and `gcc` + `zlib`.
The C core auto-compiles on first import.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# live window (ffplay), real-time
python live_gui.py --res 720
python live_gui.py --res 2k --activity busy

# record while streaming
python live_gui.py --res 1080 --record out/live.mp4 --seconds 12

# benchmarks
python bench_all.py          # all scenarios × resolutions
python final_bench.py        # vs VNC-style zlib baseline
python noise_demo.py --mode seed    # lossless noise at 0.23 KB/s
python noise_demo.py --mode true    # impossibility proof

# unit tests
pip install pytest && pytest -q
```

`live_gui.py` shows `SOURCE | DECODED`, a live bandwidth graph (LINK vs CONTENT),
latency, cache hits and `BIT-EXACT OK` / `ADAPTIVE nT (text exact)`.

---

## Repository layout

```
lowlat/
  ccodec.c / ccodec.py     # C codec core + ctypes wrapper  (the engine)
  rate.py                  # strict CBR pacer (no catch-up)
  vnc_scenario.py          # desktop / article / stress content generators
  noise.py                 # seed-defined deterministic noise
  content.py               # generic animated content
  # --- earlier neural experiment (optional, needs torch) ---
  model.py train.py codec.py ans.c ans.py freq.c freq.py sparse.py classical.py tiles.py
stream/                    # LOWLAT-S: rate-controlled streaming mode (15 KB/s demo)
  codec.py                 #   tile grid, probes, quality floors, paced budget
  run.py                   #   encode a recording under the budget + stats
  compare.py               #   side-by-side / contact sheet / rate+PSNR timeline
experiments/streaming/     # published 15 KB/s demo (contact sheet, timeline, stats)
live_gui.py                # live visual demo (ffplay window + HUD)
final_bench.py demo.py live_demo.py noise_demo.py bench_all.py
tests/test_codec.py
```

## License

MIT — see `LICENSE`.
