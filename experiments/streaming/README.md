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
| **Average rate** | **14.92 KB/s** (budget 15.0, hard cap 900 KB) |
| **Total transferred** | **895.0 KB** / 60 s (cap 900 KB) ✅ |
| **Max send gap** (silent wire) | **0.7 s** — zero gaps over 2 s |
| PSNR mean / p5 | **24.5 dB / 13.2 dB** |
| Frames sent as tile updates | 651 |
| Frames sent as crop-bands | 141 |
| Frames held (banking credit) | 895 |
| Full snapshots needed | **0** (progressive bands replace them) |
| Tile quality floor | **q82+** (691× q82–92; text never mushed) |
| Band quality floor | q75 |
| Decode fidelity | every sent tile/band is bit-consistent WebP; holds show the last good screen |

The headline number is the **send gap**: the wire never goes silent for more
than 0.7 s, so the viewer always sees *something* arriving. A previous
snapshot-banking build reached a higher mean PSNR (26.1 dB) only by freezing
the stream for up to 8.4 s while it banked a full frame — exactly the
multi-second dead air this build eliminates.

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
   representative frames) and sampled dirty tiles estimate packet costs
   within ~15 ms, so the encoder never guesses.
4. **Quality floors** — tiles ≥ q82, bands ≥ q75. When credit cannot
   afford the floor, the frame is *held* (0 bytes on the wire) instead of
   shipping mush.
5. **Dirty-age scheduling** — every tile carries how long it has been
   continuously dirty; runs and bands are served oldest-first, so a change
   that just arrived cannot be starved by newer churn. Wide lines are
   *prefix-packed*: a text line draws in over a frame or two rather than
   waiting for enough credit for the whole run.
6. **Progressive crop-bands** — dense changes (≥ 35 % of the bbox) are sent
   as whole horizontal bands of the changed region, age-targeted so the
   longest-waiting rows paint first, largest band that fits the current
   credit (full-bbox down to a single row). This replaces both scattered
   tile storms and multi-second snapshot holds: the viewer watches the
   update arrive.

### What the timeline shows

* **t ≈ 0** — keyframe burst (141 KB), the one unavoidable spike.
* **t ≈ 0.5 / 2.5 / 4.0 s** — three full-screen scene cuts (96 % of pixels
  change each time). At 15 KB/s one 1080p refresh costs ~70–90 KB ≈ 5–6 s of
  budget, so the decoder cannot be perfect during a cut — but this build
  keeps the wire busy painting the new scene in (max gap 0.7 s) instead of
  freezing on the old one. PSNR dips to ~6 dB mid-cut and recovers as bands
  land.
* **Quiet desktop** — near-zero traffic (1-byte holds), PSNR climbs to
  30–37 dB as updates converge.
* **Activity bursts** — dirty-age bands + row-runs stream continuously; the
  1 s-window rate line shows brief spikes above target when saved credit is
  spent. The 60 s average stays 14.92 KB/s and the hard cap (900 KB) is
  never crossed.
* **Last 10 s** — the tail reserve keeps ~10 KB/s available so the ending
  never starves.

### Honest limits

* A full-screen scene cut at 15 KB/s is ~5–6 s of budget (physics, not a
  codec bug). Quality dips during cuts; the guarantee is that *updates never
  stop flowing* (≤ 0.7 s silence), not that a cut arrives instantly.
* Whole-frame PSNR stays below 18 dB for 15.9 s of the minute (26.5 %),
  concentrated in the cut storm at the start and busy stretches around
  t = 12–23 s — during those windows most of the screen is old-but-coherent
  while new bands land, not a broken mosaic.
* This is an offline Python harness (cv2 + libwebp): ~0.1× realtime encode.
  The codec logic (probes ≈ 15 ms, tile WebP ≈ 0.5 ms) maps directly to a
  C/ffmpeg implementation for real-time use.
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
python stream/run.py --src out/demo/screen_60s.mp4 --out out/stream/v9 \
    --rate 15 --burst 160 --save-video

# side-by-side video, contact sheet, rate/PSNR timeline
python stream/compare.py --src out/demo/screen_60s.mp4 \
    --decoded out/stream/v9/decoded.mp4 \
    --stats  out/stream/v9/stats.json --outdir out/stream/v9/artifacts
```
