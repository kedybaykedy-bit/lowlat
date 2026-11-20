# LOWLAT-S `ScreenEncoder.encode` hot-path profile — speedup report

**Scope:** `stream/codec.py` → `ScreenEncoder.encode` / `_encode_inner` only.
**Input:** first 250 frames of `out/demo/screen_60s.mp4` (1920×1080, 30 fps),
keyframe + inter frames, with the production `Budget(total=900*1024, burst=160*1024,
rate=15*1024, duration=60.0, tail_rate=10*1024)`.
**Tool:** `stream/profile_encode.py` (cProfile over 250 frames + standalone micro-benchmarks).
**Raw numbers:** re-run `~/.venv/bin/python stream/profile_encode.py 250`.

---

## 0. Method caveats (read first)

1. **`stream/codec.py` was edited 3× by a concurrent session while I profiled it.**
   Primary profile = **sha `3aeccdf8` (mtime 15:33)**. The hot-path lines are
   byte-identical in every version I saw (verified by grep at 15:54, sha `34c2380e`):
   `np.abs(frame.astype(np.int16) - …).max(axis=2)` (line 180→182),
   `if j in dirty` list scan (line 218→220), `_build_probes` quarter probe (144→146),
   dense-band probe (423→427). Decision logic changed between versions (mode mix
   differs), so **all ratios below are tied to the sha noted, but the hotspot
   structure is stable** — re-verified once more at 16:04 (line numbers drift, the
   three hotspot expressions are still verbatim present).
2. **Machine contention.** Load average was 1.6–10 during the session. Absolute ms
   vary up to ~1.5× between runs; the *share* of each component is stable. Where a
   number moved between runs I give the range.
3. **Determinism verified.** Same input → byte-identical packet stream across runs
   (the script asserts packet lengths match between its plain and profiled pass;
   an explicit double-run test also matched). So all "wire-safe" claims below are
   about code paths that provably compute the same values, not about luck.
4. Earlier runs (pre-15:21 codec) measured a different mode mix (172 freezes vs 59);
   they are quoted only where marked *[old codec]*.

---

## 1. Headline

| metric | value |
|---|---|
| **plain encode wall (250 frames, load ≈1.6)** | **142.6 ms/frame = 7.0 fps** (35.66 s) |
| same pass under cProfile | 134.4 ms/frame (33.59 s) → profiler overhead ≈ 0 |
| other runs (same code) | 158 ms (load ~2), 219 ms (load spike) → contention band |
| first 60 frames (burst phase) | **282 ms/frame** (3.5 fps) — big bands while burst credit lasts |
| mode mix (250 f) | key 1, frozen 59, crop 27, tiles 163 |
| traffic | 291 303 B total, mean 1 164 B/pkt |
| dirty tiles | 349 294 over 249 frames = **1 397/frame of 2 040** |
| dense (band) branch entered | **233 of 249 frames** (93 %) |

For reference, the stated 0.37 s/frame full-run baseline is **not** all encoder
(see §5: `run.py` adds PSNR + decode + mp4 write on top).

---

## 2. cProfile — top functions (250 frames, 33.589 s total)

### 2a. By cumulative time

```
 ncalls  tottime  cumtime  function
      1    0.000   33.589  <listcomp> (encode loop)                     100 %
    250    0.011   33.589  codec.py:256 encode                           100 %
    250    0.129   33.578  codec.py:262 _encode_inner                    100 %
    249    3.529   19.071  codec.py:174 _dirty_tiles                     56.8 %
  25658   14.119   14.119  {method 'reduce' of numpy.ufunc}              42.0 %
  22609    0.009   12.863  {method 'max' of numpy.ndarray}               38.3 %
   4385    0.012   10.923  codec.py:89 _webp                             32.5 %
   4385   10.907   10.907  {imencode}                                    32.5 %
    249    0.009    3.932  codec.py:134 _build_probes                    11.7 %
    249    3.077    3.142  codec.py:202 _row_runs                         9.4 %
    747    1.375    1.375  {method 'astype'}                              4.1 %
    498    0.002    1.255  {method 'sum'}                                 3.7 %
    249    0.006    0.388  codec.py:146 <listcomp> (8 tile samples)       1.2 %
    175    0.003    0.126  codec.py:343 _pack (tile sends)                0.4 %
    249    0.040    0.075  codec.py:315 <listcomp> (run-age)              0.2 %
    249    0.002    0.067  codec.py:126 _bbox_of                          0.2 %
 699758    0.058    0.058  {list.append}                                  0.2 %
 367657    0.035    0.035  {divmod}                                       0.1 %
     28    0.000    0.032  codec.py:96 _webp_dec / {imdecode}             0.1 %
 349294    0.032    0.032  {set.add}                                      0.1 %
    498    0.026    0.026  {numpy.zeros}                                  0.1 %
    498    0.020    0.023  {method 'argsort'}                             0.1 %
    233    0.001    0.012  numpy.unique (dense branch)                    0.0 %
```

### 2b. By total (self) time

```
  25658   14.119  {method 'reduce'}        <- np.abs(int16).max(axis=2) + tile sums
   4385   10.907  {imencode}               <- every WebP encode
    249    3.529  codec.py:174 _dirty_tiles <- Python/np.where/abs-sub/argsort loop
    249    3.077  codec.py:202 _row_runs    <- `j in dirty` list scans
    747    1.375  {astype}                  <- 2× int16 casts per frame
    250    0.129  codec.py:262 _encode_inner
 699758    0.058  {list.append}
    249    0.040  codec.py:315 <listcomp>
 367657    0.035  {divmod}
     28    0.032  {imdecode}
 349294    0.032  {set.add}
    498    0.026  {numpy.zeros}
    498    0.020  {argsort}
   4385    0.012  {imencode} wrapper / tobytes 0.005
```

---

## 3. Estimated ms/frame breakdown (baseline 142.6 ms/frame)

Shares are from the profile (exact); the WebP sub-split is *inferred* from call
counts × micro unit costs (±20 %), the totals are exact.

| component | ms/frame | share | measured how |
|---|---:|---:|---|
| **`_dirty_tiles` total** | **76.6** | **54 %** | profile cumtime |
| ↳ `np.abs(int16 diff).max(axis=2)` | 51.7 | 36 % | `{max}` cumtime 12.863 s |
| ↳ `astype(int16)` ×2 | 5.5 | 4 % | `{astype}` 1.375 s |
| ↳ tile counts + `np.where(…).sum` | 5.0 | 3.5 % | `{sum}` 1.255 s |
| ↳ Python self (abs/sub/np.where/argsort scan) | 14.2 | 10 % | `_dirty_tiles` tottime |
| **WebP encodes, all** | **43.7** | **31 %** | `_webp` cumtime 10.923 s / 4 385 calls |
| ↳ `_build_probes`: 1 quarter-crop probe (480×270) | 14.2 | 10 % | micro 15.4 ms × 249 |
| ↳ `_build_probes`: 8 sampled tile probes | 1.2 | 1 % | listcomp 0.388 s |
| ↳ dense-band `frac=1.0` probe — **exact duplicate of the quarter probe** | ~13 | ~9 % | 233 calls × ~15 ms |
| ↳ sub-frac band probes (0.5 … 0.03) | ~6 | ~4 % | ~1 100 calls × 1–4 ms |
| ↳ **actual band encodes that produce bytes** (27 crop packets) | ~9–12 | ~7 % | 27 × 98–119 ms |
| ↳ tile sends (`_pack`, ~700 tiles) + keyframe | 1.3 | 1 % | `_pack` 0.126 s |
| **`_row_runs` (Python `j in dirty` list scan)** | **12.6** | **9 %** | profile cumtime 3.142 s |
| decision Python (`_encode_inner` self + listcomps + `_bbox_of`) | ~3 | 2 % | tottime + small entries |
| `_webp_dec` (key + 27 crop decodes) | 0.13 | 0 % | 0.032 s |
| **total (profiled)** | **≈134** | | plain wall 142.6 |

**The three facts that drive everything:**

1. **36 % of all runtime is one numpy call**: `np.abs(a-b).max(axis=2)` on a
   (1080,1920,3) int16 array — 51.7 ms. numpy's generic reduction over the last
   axis is ~10× slower than an elementwise pass over the same data (see §4:
   `cv2.absdiff + np.maximum×3` does the identical job in 5.9–7.5 ms).
2. **31 % is WebP encoding, of which ~60 % is *probing*** (size estimation that
   produces no bytes): ≈30 ms/frame. Of that, **~13 ms/frame is computed twice**
   — `_build_probes` encodes `frame[bbox][::4,::4]` at q82, then the dense band
   loop with `frac=1.0` re-encodes *the same pixels at the same quality*
   (the bbox from `_bbox_of` is always tile-aligned, so `band == frame[y0:y1,x0:x1]`).
3. **9 % is a Python O(n·m) list scan** in `_row_runs`: `if j in dirty` where
   `dirty` is a list of ~1 400 ints.

---

## 4. Micro-benchmarks (median ms/call)

Two runs (profile pass at load ≈1.6, dedicated micro pass at load ≈3.6 — the
second column is the contended one; both agree on ratios):

| operation | load ≈1.6 | load ≈3.6 |
|---|---:|---:|
| `cv2.imencode` webp q85, 32×32 tile (static) | 0.151 | 0.170 |
| `cv2.imencode` webp q85, 32×32 tile (changed) | 0.184 | 0.190 |
| `cv2.imencode` webp q82, **1920×1080 full frame** | **188.5** | 202.1 |
| `cv2.imencode` webp q85, 1920×1080 full frame | 195.1 | 212.0 |
| `cv2.imencode` webp q82, 1920×540 band | 98.2 | 119.4 |
| `cv2.imencode` webp q82, **480×270 quarter probe** | **15.4** | 17.1 |
| `cv2.imdecode` webp 1920×1080 (blob only) | 17.3 | 22.0 |
| `astype(int16)` ×2 on 1080p | 5.66 | 5.35 |
| int16 subtract + abs (no max) | 7.04 | 7.14 |
| **CURRENT `np.abs(int16).max(axis=2)`** | **60.1** | **61.3** |
| ⟶ cand: `cv2.absdiff` + `np.maximum` ×3 | **5.85** | **7.49** |
| ⟶ cand: `cv2.absdiff` + `cv2.max` ×3 | — | 7.53 |
| ⟶ cand: `cv2.absdiff` + `.max(axis=2)` (still numpy max!) | 54.6 | 58.9 |
| **cand equivalence vs current `d`** | — | **`True`, max\|delta\|=0** (both cands) |
| tile counts `reshape+sum` (bool) | — | 2.77 |
| tile sums `np.where(…)+reshape+sum` (int16) | 4.16 | 7.56 |
| FULL `_dirty_tiles` equivalent | 69.2 | 78.5 |
| `_build_probes` (60 dirty tiles) | 16.4 | 18.6 |
| `_row_runs`, 200 dirty tiles: **current / set** | — | 0.773 / **0.107** |
| `_row_runs`, 1 000 dirty tiles: current / set | 19.2 | 19.66 / **0.614** |
| `_row_runs`, 1 859 dirty tiles: current / set | — | 19.16 / **0.616** |

Notes:
* The **bottleneck inside the diff is `ndarray.max(axis=2)` itself** (≈54 ms), not
  the subtract/abs (7 ms) — proof: swapping only the front half to `cv2.absdiff`
  while keeping `.max(axis=2)` saves nothing (54.6 ms).
* **`cv2.reduce(..., dim=2)` does NOT reduce across channels** (OpenCV only reduces
  rows/cols) — that candidate is invalid, discarded.
* The two surviving candidates were verified **bit-identical** to the current `d`
  (`np.array_equal == True`, `max|delta| == 0`) on real frame pairs → identical
  dirty masks, identical magnitudes, identical sort order → **identical wire bytes**.

### Harness context (why the full run is 0.37 s/frame, not 0.14)

Measured separately (same machine):

| run.py component | ms/frame |
|---|---:|
| `run.psnr()` — float64 PSNR over 1080p×3 | **50.6** |
| staleness check `np.abs(int16).max()` (frozen frames only, 24 % of frames) | 38.0 (→ ~9 amortized) |
| `ScreenDecoder.decode` over the first-60-frame packets (tile-dense, burst phase) | **up to 502** (front-loaded; later frames are far cheaper) |
| encoder alone, first 60 frames (burst: big bands) | 282 |
| encoder alone, frames 0–249 | 142.6 |

So the 11-min/1800-frame wall ≈ encoder 0.14–0.28 + PSNR 0.05 + decode (heavy early) + mp4 write + staleness. **The harness is roughly a third of the wall clock and is pure Python-side waste too.**

---

## 5. Ranked speedup proposals

Gain figures are **ms/frame saved against the 142.6 ms baseline**, grounded in the
numbers above. "Wire risk" = could the packet bytes change?

### Tier A — zero wire risk (provably identical output), do all four

| # | What to change | Gain | Why it's safe | Size |
|---|---|---|---|---|
| **A1** | `_dirty_tiles`: replace `np.abs(frame.astype(np.int16) - sent_src.astype(np.int16)).max(axis=2)` with `d = cv2.absdiff(frame, sent_src)` then `np.maximum(np.maximum(d[...,0], d[...,1]), d[...,2])` | **−59 ms (−41 %)** → 83 ms left | **Verified bit-identical `d`** (max\|delta\|=0 on real pairs). Same values ⇒ same `_dirty2d`, same magnitudes, same `argsort` ⇒ same packets. Also removes both `astype` casts (5.5 ms) and works in uint8 (halves the follow-up `np.where` traffic). `cv2.absdiff` releases the GIL. | 5 LOC |
| **A2** | `_row_runs`: build `dirty_set = set(dirty)` once, test `j in dirty_set` instead of `j in dirty` | **−12 ms (−8.6 %)** | Set membership returns True exactly where list membership does; iteration order and run construction untouched → identical `runs` → identical packets. Measured 19.2 → 0.62 ms at n=1 859 (**31×**). | 3 LOC |
| **A3** | dense-band loop, `frac >= 1.0`: use `probe = self._probe_q82` instead of re-encoding `band[::4,::4]` at q82 | **−11…−13 ms (−8 %)** | `_bbox_of` always returns tile-aligned coords, so the `frac=1.0` band is *the same pixel region* `_build_probes` already encoded at the same q → identical probe length → identical `est` → identical band/quality choice. 233 duplicate encodes/250 frames. | 3 LOC |
| **A4** | `_dirty_tiles` tail: (a) keep the diff in uint8 (comes free with A1) so `np.where(big, d, 0)` moves to 2.1 MB instead of 4.2 MB; (b) replace the Python `for flat in order:` scan with a vectorized filter (`order[mag_flat[order] > 0]` → `divmod` once on the array) — argsort is called identically, order preserved | **−4…−6 ms (−3…−4 %)** | Same formulas, same `argsort` input, same output list order. | 10–15 LOC |
| | **Tier A total** | **−87 ms → ≈55 ms/frame (2.6×, ~18 fps)** | | ~25 LOC |

### Tier B — safe *if* you prove it byte-identical (cheap to check: run 250 frames
through both versions and `==` the packet list; `stream/compare.py` already exists)

| # | What to change | Gain | Risk | Size |
|---|---|---|---|---|
| **B1** | Skip the sub-frac band probes whose `est` cannot fit: compute a *conservative lower bound* on `probe` (e.g. smallest possible q82 WebP for that region) and `continue` before encoding. Today the loop probes all 5 sub-fracs on essentially every dense frame even when credit is a few hundred bytes. | −4…−6 ms | **Medium** — a too-tight bound would skip a band that *would* have fit → the row stays dirty one more frame → different bytes. Must be validated byte-exactly. | 6 LOC |
| **B2** | Don't run `_build_probes` on frames that will freeze anyway (credit below the smallest possible packet). | −0…−16 ms (only on the 59/250 frozen frames) | **Medium** — today's decision uses *fresh* estimates; skipping means the freeze decision uses the *previous* frame's `_probe_tile82`. A flip (fresh says "1 tile fits", stale says "no") changes bytes. Validate byte-exactly. | 5 LOC |
| **B3** | Cache the quarter probe keyed by an exact hash of `frame[bbox][::4,::4]` (392 KB hash ≈ 0.2 ms vs 14 ms encode) | −0…−14 ms, **hit rate likely ≈0** | None on hit (identical input ⇒ identical length) — but the source is H.264-compressed, so pixels differ every frame and exact hits should be rare. Low expected value; measure before building. | 8 LOC |

### Tier C — algorithmic (largest remaining wins, but *can* change wire output)

| # | What to change | Gain | Risk | Size |
|---|---|---|---|---|
| **C1** | Probe the band battery at 1/8 instead of 1/4 (`frame[::8,::8]` = 240×135 ≈ 4 ms instead of 15 ms), or probe only every 2nd frame and reuse while the bbox is unchanged | −8…−12 ms | **High** — different `est` → different q/band selection → different bytes *and* different PSNR/staleness stats. Needs full A/B on the recording (byte compare + `stats.json` diff). | 10 LOC + A/B |
| **C2** | Cut the probe battery to one probe/frame (reuse `_probe_q82` for the whole bbox, derive sub-band costs by area scaling) | −15…−20 ms | High (same as C1) | 20 LOC + A/B |

### Tier D — parallelism (GIL-aware)

| # | What to change | Gain | Notes | Size |
|---|---|---|---|---|
| **D1** | `ThreadPoolExecutor` for the probe battery + tile/band encodes (`cv2.imencode` **releases the GIL**) | −3…−8 ms | **Limited**: probes depend on `dirty`, which needs the numpy diff first (numpy does *not* release the GIL), so only the Python decision work (`_row_runs`, age/sort ≈ 3 ms) can overlap probes. Same inputs, same assembly order → output unchanged. Worth it only after Tier A. | 30 LOC |
| **D2** | **Decoder**: thread-pool `_webp_dec` over the tiles of one packet | **−300…−450 ms/frame in the burst phase** (harness only) | Tile decodes are independent and `imdecode` releases the GIL → near-linear scaling. Doesn't touch encoder wire output at all. Biggest single lever on the *run.py wall clock*. | 15 LOC |
| **D3** | `run.py` PSNR: use `cv2.PSNR` / uint32 MSE instead of float64 `np.mean((a-b)**2)` | −45…−50 ms/frame (harness only) | Pure measurement change; reported PSNR differs by <0.01 dB. | 5 LOC |
| **D4** | `multiprocessing` for the encoder | **not recommended** | `sent_src`/`recon`/`age` must update serially *between* frames (frame N+1's dirty is vs. post-send-N source), so frame-level pipelining is illegal; a 6 MB frame +6 MB `sent_src` per hop costs more than the work saved. | — |

---

## 6. Projection and verdict

### Encode-only, 1080p, measured on this machine (load ≈1.6)

| stage | ms/frame | fps |
|---|---:|---:|
| today (sha `3aeccdf8`) | 142.6 | 7.0 |
| + Tier A (zero-risk, ~25 LOC) | **≈55** | **≈18** |
| + Tier B (validated) | ≈48 | ≈21 |
| + Tier D1 threading | ≈45 | ≈22 |
| + Tier C (probe reduction, validated) | **≈32–35** | **≈29–31** |
| idle-machine factor (~1.3–1.5× from the contention band) | ÷1.4 | — |

Composition after Tier A: WebP ≈43 ms (78 % — of which probes ≈30 ms, real
encodes ≈13 ms), dirty detection ≈18 ms, Python ≈4 ms.

### Can Python hit 30 fps 1080p?

**Yes — but not with the encoder optimizations alone, and not because of Python.**

* The zero-risk Python work gets **2.6× (7 → 18 fps)** for ~25 lines. That is the
  free 60 % of the profile.
* After that the encoder is **78 % libwebp** (already native code). A Rust/C port
  of the hot loop would reclaim only the residual Python/numpy work — the
  `max(axis=2)` and `_row_runs` items Tier A *already* removes in Python. Porting
  without Tier A ≈ 50 ms/frame; porting *with* Tier A ≈ 45–50 ms → **roughly the
  same as optimized Python.** The port is therefore neither necessary for the big
  win nor sufficient for 30 fps.
* The real remaining barrier is **the probing strategy**: ~30 ms/frame of q82
  WebP encodes used only to guess packet sizes, plus ~10 ms of actual band
  encodes. Those cost the same in any language (libwebp is libwebp). Cutting them
  (Tier B/C) is what moves 18 → 30 fps.
* Hard floors that no language fixes: a 1920×1080 WebP at q82 costs **188–202 ms**
  and a 1920×540 band **98–119 ms** on this box. The burst phase (first ~60 frames,
  measured 282 ms/frame) cannot run at 30 fps while full-bbox snapshots are being
  encoded synchronously — that's a pacing/design issue, not a language issue.
* And the stated 0.37 s/frame full-run wall is ~⅓ harness (PSNR 50 ms + decode
  + mp4 write + staleness 9 ms amortized). **Fixing D2/D3 alone removes ~100 ms
  of wall clock without touching the codec.**

### Bottom line

1. **Do Tier A now** (≈25 LOC, provably byte-identical): 142.6 → ~55 ms/frame,
   7 → 18 fps. Biggest single item: replace the numpy int16 `abs().max(axis=2)`
   with `cv2.absdiff + np.maximum×3` (−41 %, verified identical output).
2. **Tier B/C (probe reduction) is required for 30 fps** and needs a byte-exact
   A/B harness, not a port.
3. **A Rust/C port of the encoder loop is NOT required and would not reach 30 fps
   by itself** — after Tier A the encoder is dominated by libwebp calls that are
   already native. Port only if you also want margin, or to serve the *decoder*
   (which is the actual wall-clock hog during the burst phase).
4. If "10×" is judged on `run.py`'s 11-minute wall, the cheapest 10× is
   **Tier A + D2 + D3** (encoder 2.6×, decoder threading, cheap PSNR) — all
   Python, none of it changes a single byte on the wire.
