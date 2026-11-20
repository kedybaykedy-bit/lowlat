# Live screen streaming at 15 KB/s — real 60 s desktop recording

**LOWLAT-S** is the streaming mode of this project: a rate-controlled screen
codec fed with a real 60-second 1080p/30fps desktop recording (terminal,
wallpaper, window fades, notifications, scene cuts) and hard-limited to a
**15 KB/s** budget.

![contact sheet](contact_sheet.png)
*Source (left of each pair) vs LOWLAT-S decode, sampled over the minute.
Panels: wallpaper desktop, file manager, fastfetch output, opencode TUI.*

![rate + quality timeline](timeline.png)
*Top: bytes/frame and 1 s-window rate vs the 15 KB/s target.
Bottom: decoded-vs-source PSNR over time.*

## Result

| metric | value |
|---|---|
| **Average rate** | **14.97 KB/s** (budget 15.0, hard cap 900 KB) |
| **Total transferred** | **898.3 KB** / 60 s (cap 900 KB) ✅ |
| **Max send gap** (silent wire) | **0.4 s** — zero gaps over 2 s |
| **Interactive latency** (small change → on screen) | **p50 0.13 s** (was 0.73 s before recency scheduling) |
| PSNR mean / p5 | **24.28 dB / 13.44 dB** |
| Frames sent as tile updates | 1235 |
| Frames sent as crop-bands | 103 |
| Frames held (banking credit) | 398 |
| Full snapshots needed | **0** (progressive bands replace them) |
| Tile quality floor | **q82+** — 3100 of 3182 encodes at q82–92 (97 %) |
| Band quality floor | q75 |
| Decode fidelity | every sent tile/band is bit-consistent WebP; holds show the last good screen |
| Encode speed (offline harness) | **0.4× realtime** (2.6× after profiled hot-path cleanup; byte-identical output) |

### Latency is the headline

A remote viewer's core experience is *how long until what I just did
appears*. Measured per 32×32 tile: a "stale run" = decode does not match
the current source for that long. Small-change events (≤12 tiles changed
in the source that frame ≈ typing, status updates, cursor):

| interactive event | before (age-ordered only) | **now (recency lane A)** |
|---|---|---|
| p50 | 0.73 s | **0.13 s** |
| slow events >3 s during the typing flood (t=32–60 s) | 77 | **26** |
| max send gap on the wire | 0.7 s | **0.4 s** |

The mechanism: a **recency-first interactive lane** packs the newest runs
first every frame (budget = min(credit, 1.5 KB, rate-share)), so a line
the user just typed never queues behind bulk backlog — while a hard
**rate share (5 KB/s)** guarantees bulk bands always keep ≥10 KB/s to
bank against, so the two lanes cannot starve each other.

## How it works

```
budget:  allowed(t) = min(900 KB, burst + 15 KB/s·t, 900 KB − 10 KB/s·(60 s − t))
         └ keyframe headroom      └ sustained rate      └ tail reserve (no dead end)
```

1. **Keyframe** — one full-frame WebP q85 (141 KB); text stays sharp
   (PSNR > 40 dB on text crops).
2. **32×32 tile grid** — dirty detection against the last *sent source*
   (threshold kills codec noise; no drift: every send is a fresh deterministic
   WebP of the source, the decoder just replaces tiles).
3. **Content-aware size probes** — a quarter-frame WebP (calibrated ×6.5 on 7
   representative frames) and sampled dirty tiles estimate packet costs,
   so the encoder never guesses.
4. **Quality floors** — tiles ≥ q82, bands ≥ q75. When credit cannot
   afford the floor, the frame is *held* (0 bytes on the wire) instead of
   shipping mush.
5. **Two-lane scheduling** — lane A packs runs **newest-first** (recency =
   urgency) under a per-frame cap + a 5 KB/s long-run share; bulk is
   **dense → age-targeted crop-bands** (full-bbox down to a single row,
   largest that fits) and **sparse → age-oldest prefix-packed runs**
   (a wide line draws in over a frame or two rather than waiting for
   enough credit for the whole run).
6. **P_BATCH** — when lane A and a band both fire in one frame they ride
   together in a single batched packet (one frame, two segments).

### What the timeline shows

* **t ≈ 0** — keyframe burst (141 KB), the one unavoidable spike.
* **t ≈ 0.5 / 2.5 / 4.0 s** — three full-screen scene cuts (96 % of pixels
  change each time). At 15 KB/s one 1080p refresh costs ~70–90 KB ≈ 5–6 s of
  budget, so the decoder cannot be perfect during a cut — but the wire never
  goes silent (max gap 0.4 s): the new scene is painted in progressively
  instead of freezing on the old one. Whole-frame PSNR stays below 18 dB
  for 7.8 s (the cut storm), then recovers.
* **Quiet desktop** — near-zero traffic (1-byte holds), PSNR climbs to
  30–37 dB as updates converge.
* **Typing / TUI output (t≈ 32+)** — lane A delivers each new line with
  0.13 s median latency while bands keep the rest of the screen current.
* **Last 10 s** — the tail reserve keeps ~10 KB/s available so the ending
  never starves.

### Honest limits

* A full-screen scene cut at 15 KB/s is ~5–6 s of budget (physics, not a
  codec bug). Quality dips during cuts; the guarantee is *updates never
  stop flowing* (≤ 0.4 s silence), not that a cut arrives instantly.
* Whole-frame PSNR stays below 18 dB for 15.2 s of the minute (25.4 %),
  concentrated in the cut storm at the start and busy stretches around
  t = 12–23 s — during those windows most of the screen is old-but-coherent
  while new bands land, not a broken mosaic.
* Bulk (large-change) events still median ~2.2 s: delivering ~200 KB of
  new content through a 15 KB/s pipe takes ~13 s of channel time; the
  scheduler chooses *who waits*, not whether physics applies.
* This is an offline Python harness (cv2 + libwebp): 0.4× realtime after
  profiled hot-path cleanup (all changes byte-identical — see
  [`profile_report.md`](profile_report.md)). The codec logic maps
  directly to a C/ffmpeg implementation for real-time use.
* The wire model treats a *hold* as absence of update (0 bytes) — how any
  TCP/WebSocket screen stream works.

## Files

| file | contents |
|---|---|
| [`stats.json`](stats.json) | full run: per-frame bytes, modes, quality histogram, PSNR, staleness windows |
| [`contact_sheet.png`](contact_sheet.png) | source vs decode samples |
| [`timeline.png`](timeline.png) | rate + PSNR over the 60 s |
| [`lowlat-s_15kbps_compare.mp4`](https://github.com/kedybaykedy-bit/lowlat/releases/download/demo-15kbps/lowlat-s_15kbps_compare.mp4) | SOURCE \| LOWLAT-S side-by-side (release) |
| [`lowlat-s_15kbps_decoded.mp4`](https://github.com/kedybaykedy-bit/lowlat/releases/download/demo-15kbps/lowlat-s_15kbps_decoded.mp4) | what the remote viewer sees (release) |

## Reproduce

```bash
# encode + decode the recording under the 15 KB/s budget
python stream/run.py --src out/demo/screen_60s.mp4 --out out/stream/v13 \
    --rate 15 --burst 160 --save-video

# side-by-side video, contact sheet, rate/PSNR timeline
python stream/compare.py --src out/demo/screen_60s.mp4 \
    --decoded out/stream/v13/decoded.mp4 \
    --stats  out/stream/v13/stats.json --outdir out/stream/v13/artifacts

# change-to-display latency (per-tile stale runs, interactive vs bulk)
python stream/latency.py --decoded out/stream/v13/decoded.mp4
```
