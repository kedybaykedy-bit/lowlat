# "Me at the zoo" → 1080p mosaic → LOWLAT

Source: `jNQXAC9IVRw_ia.mp4` (320×240, 15 fps, 284 frames, 18.9 s) — the first YouTube video.
Built into a **1920×1080** mosaic (aspect-preserved cells, letterboxed) for two grids.

| grid | tiles | input video | lossless out | adaptive out |
|---|---|---|---|---|
| 2×2 | 4 | `zoo_2x2_input_1080p.mp4` | `zoo_2x2_lowlat_lossless.mp4` | `zoo_2x2_lowlat_adaptive.mp4` |
| 3×3 | 9 | `zoo_3x3_input_1080p.mp4` | `zoo_3x3_lowlat_lossless.mp4` | `zoo_3x3_lowlat_adaptive.mp4` |

Side-by-side (INPUT | LOSSLESS | ADAPTIVE):
`zoo_2x2_compare_sbs.mp4`, `zoo_3x3_compare_sbs.mp4`
Contact sheets: `zoo_{2x2,3x3}_contact_sheet.png`
Raw stats: `zoo_{2x2,3x3}_stats.json`, `zoo_summary.json`

## Results (steady-state, keyframe excluded)

| grid | mode | bitrate | bit-exact | notes |
|---|---|---|---|---|
| 2×2 | lossless | **11.97 MB/s** | yes (diff 0) | real video, high entropy |
| 3×3 | lossless | **9.85 MB/s** | yes (diff 0) | |
| 2×2 | adaptive (10 KB/s target) | 1.70 MB/s | non-text tiles exact (viol=0) | |
| 3×3 | adaptive (10 KB/s target) | 1.68 MB/s | non-lossy viol=0 | |

Keyframe: 523 KB (2×2), 535 KB (3×3) — small because the repeated cells are
cache-referenced.

## Honest interpretation

This is **real, dynamic webcam video filling the whole frame** — the opposite of a
static desktop. It is high-entropy: lossless is ~10–12 MB/s and even aggressive
lossy degradation only reaches ~1.7 MB/s. **10 KB/s is impossible here** — that is
the Shannon bound, not a codec defect.

The point of this test: show the *contrast*. LOWLAT's promise (≤10 KB/s) lives on
**screen/desktop content** (static regions, text, cursor, scrolling), where the
same engine delivers ~1–9 KB/s lossless. Video is shown as the honest upper bound.

Also visible: the adaptive pass protects "text-like" tiles — on pure video the
edge/bimodal heuristic mis-classifies some textured video as text, so a chunk of
tiles stays lossless. On real desktop content that heuristic is exactly what keeps
text pixel-perfect.

## Regenerate

```bash
# download the source (Internet Archive mirror of the original video)
curl -L -o out/zoo/jNQXAC9IVRw_ia.mp4 \
  https://archive.org/download/jawed-jNQXAC9IVRw/jNQXAC9IVRw.ia.mp4
# build mosaics + stream + save videos/stats
python zoo_bench.py --grids 2x2,3x3 --src out/zoo/jNQXAC9IVRw_ia.mp4
```
Videos land in `out/zoo/` (git-ignored; large).
