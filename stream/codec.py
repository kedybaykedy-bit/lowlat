#!/usr/bin/env python3
"""LOWLAT-S: rate-controlled screen streaming codec (target 15 KB/s).

Design (measured on the 60s screen recording):
  - Keyframe: full-frame WebP (text stays sharp at q85+, PSNR>40dB on text).
  - Inter frames: 32x32 tile grid; dirty detection vs last SENT SOURCE
    (threshold kills x264 noise; no drift: every sent tile is a fresh
    deterministic WebP of the source, decoder just replaces tiles).
  - Modes: SKIP (hold) / TILES (whole row-runs) / CROP (bbox snapshot -
    only the changed region, the cheap coherent refresh).
  - Rate budget: allowed(t) = min(TOTAL, BURST + RATE*t); TOTAL = rate*duration
    so the 60s average never exceeds the target. A frame is sent only if its
    ESTIMATED cost fits credit; quality ladder down first, freeze last.
  - Size cache: observed WebP sizes per quality -> frozen frames cost only
    the dirty-mask computation (~50ms), never a wasted encode.
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import cv2
import numpy as np

P_SKIP = 0
P_KEY = 1
P_FULL = 2          # whole-frame snapshot (legacy)
P_TILES = 3
P_CROP = 4          # bbox snapshot: only the changed region (v4)

TILE = 32
DIRTY_PX_MIN = 6      # changed pixels (|d|>3) inside a tile to call it dirty
DIRTY_D = 3           # per-pixel threshold: kills x264 noise
QUALITY_LADDER = (92, 88, 85, 82, 80, 75, 70, 65, 60, 55, 50)
Q_FLOOR_TILE = 82     # text tiles never sent below this (readability)
Q_FLOOR_FULL = 75     # full-frame snapshots floor
HOLD_ACTIVITY = 0.15  # frame-to-frame churn -> hold, save for a full snapshot
HOLD_BACKLOG = 0.30   # big backlog vs sent_src -> same

# Calibration: est_full(q) = quarter82_probe * _FULL_F[q]  (measured on 7
# representative frames; full/quarter ratio is content-stable ~6.5x)
_FULL_F = {92: 11.45, 88: 8.54, 85: 7.25, 82: 6.84, 80: 6.29,
           75: 5.54, 70: 5.30, 65: 5.17, 60: 5.03, 55: 4.96, 50: 4.75}
# est_tiles(q) = sample_mean82 * n * _TILE_SCALE[q] (median tile sizes)
_TILE_SCALE = {92: 1.21, 88: 1.10, 85: 1.04, 82: 1.00, 80: 0.99,
               75: 0.95, 70: 0.94, 65: 0.93, 60: 0.91, 55: 0.90, 50: 0.88}
_EST_SAFETY = 1.15


@dataclass
class Budget:
    """Paced token budget with a tail reserve.

    allowed(t) = min(total,
                     burst + rate*t,                    # keyframe headroom + rate
                     total - tail_rate*(duration - t))  # reserve for the final seconds

    All three terms are non-decreasing in t, so credit never shrinks.
    Guarantees: kf fits at t=0 (burst), sustained ~rate early, and at least
    tail_rate per second remains available for the last seconds (no dead tail),
    hitting exactly `total` at t=duration.
    """
    total: float
    burst: float
    rate: float            # bytes per second
    duration: float = 0.0  # seconds (0 = no tail reserve)
    tail_rate: float = 0.0 # bytes per second reserved for the end
    t0: float = 0.0
    spent: float = 0.0

    def allowed(self, t: float) -> float:
        t = max(0.0, t - self.t0)
        a = min(self.total, self.burst + self.rate * t)
        if self.duration > 0 and self.tail_rate > 0:
            a = min(a, self.total - self.tail_rate * max(0.0, self.duration - t))
        return max(a, 0.0)

    def credit(self, t: float) -> float:
        return self.allowed(t) - self.spent

    def pay(self, n: float) -> None:
        self.spent += n


def _webp(img: np.ndarray, q: int) -> bytes:
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, int(q)])
    if not ok:
        raise RuntimeError("webp encode failed")
    return buf.tobytes()


def _webp_dec(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("webp decode failed")
    return img


class ScreenEncoder:
    def __init__(self, w: int, h: int, fps: float, budget: Budget):
        self.w, self.h, self.fps = w, h, fps
        self.budget = budget
        self.tw = (w + TILE - 1) // TILE
        self.th = (h + TILE - 1) // TILE
        self.ntiles = self.tw * self.th
        self.sent_src: np.ndarray | None = None
        self.prev_src: np.ndarray | None = None   # previous SOURCE frame (activity ref)
        self.recon: np.ndarray | None = None
        self._probe_for = None
        self._probe_q82 = 0
        self._probe_tile82 = 200.0
        self._full_block = 0   # do not attempt full below this credit
        # per-tile staleness: frames since that tile's content was last sent.
        # Bands target the OLDEST dirty rows so no region can be pinned
        # (the v6 bug: top-of-bbox bands re-hit a churny top forever while
        # the bottom stayed 100% stale for 10+ seconds).
        self.age = np.zeros(self.ntiles, np.uint32)
        self.stats = dict(frames=0, skip=0, key=0, full=0, tiles=0,
                          frozen=0, bytes=0, dirty_sum=0, q_hist={})

    # ---------- estimates: content-aware via cheap probes ----------
    def _bbox_of(self, dirty: list[int]) -> tuple[int, int, int, int]:
        """Pixel bbox (x0,y0,x1,y1) covering all dirty tiles, clipped."""
        tys = [i // self.tw for i in dirty]
        txs = [i % self.tw for i in dirty]
        y0, y1 = min(tys) * TILE, min((max(tys) + 1) * TILE, self.h)
        x0, x1 = min(txs) * TILE, min((max(txs) + 1) * TILE, self.w)
        return x0, y0, x1, y1

    def _build_probes(self, frame: np.ndarray, dirty: list[int]) -> None:
        """Quarter-crop WebP probe + sampled dirty-tile sizes (both at q82).
        ~15ms total; makes size estimates content-aware instead of static priors."""
        if self._probe_for is frame:
            return
        x0, y0, x1, y1 = self._bbox_of(dirty)
        self._crop_geom = (x0, y0, x1, y1)
        crop_tiles = (int(np.ceil((y1 - y0) / TILE)) *
                      int(np.ceil((x1 - x0) / TILE)))
        self._crop_fill = len(dirty) / max(1, crop_tiles)
        self._probe_q82 = len(_webp(frame[y0:y1, x0:x1][::4, ::4], 82))
        step = max(1, len(dirty) // 8)
        samples = [len(_webp(self._tile(frame, dirty[i]), 82))
                   for i in range(0, len(dirty), step)][:8]
        self._probe_tile82 = float(np.mean(samples)) if samples else 200.0
        self._probe_for = frame

    def _est_full(self, q: int) -> int:
        base = getattr(self, "_probe_q82", None)
        if base is None:
            return 120_000
        return int(base * _FULL_F[q] * _EST_SAFETY)

    def _est_tile(self, q: int) -> int:
        base = getattr(self, "_probe_tile82", None)
        if base is None:
            return 350
        return int(base * _TILE_SCALE.get(q, 1.0) * _EST_SAFETY)

    def _note_full(self, q: int, n: int, failed: bool = False) -> None:
        if failed:
            # exact learning for this attempt; block re-attempts below actual
            self._full_block = max(getattr(self, "_full_block", 0), int(n * 1.02))
        else:
            self._full_block = 0

    def _note_tile(self, q: int, n: int) -> None:
        pass  # probe-based estimates are already per-frame fresh

    # ---------- helpers ----------
    def _dirty_tiles(self, frame: np.ndarray) -> list[int]:
        """Dirty tiles sorted by change magnitude (biggest first).
        Also stores self._dirty2d (bool th x tw)."""
        if self.sent_src is None:
            self._dirty2d = np.ones((self.th, self.tw), dtype=bool)
            return list(range(self.ntiles))
        d = np.abs(frame.astype(np.int16) - self.sent_src.astype(np.int16)).max(axis=2)
        big = d > DIRTY_D
        ph, pw = self.th * TILE, self.tw * TILE
        padded_d = np.zeros((ph, pw), np.int16)
        padded_d[: self.h, : self.w] = d
        padded_b = np.zeros((ph, pw), bool)
        padded_b[: self.h, : self.w] = big
        counts = padded_b.reshape(self.th, TILE, self.tw, TILE).sum(axis=(1, 3))
        # magnitude = mean of changed-pixel deltas per tile (0 if none)
        sums = np.where(padded_b, padded_d, 0).reshape(
            self.th, TILE, self.tw, TILE).sum(axis=(1, 3))
        mag = np.where(counts >= DIRTY_PX_MIN, sums / np.maximum(counts, 1), 0.0)
        self._dirty2d = counts >= DIRTY_PX_MIN
        order = np.argsort(-mag, axis=None)
        out = []
        for flat in order:
            r, c = divmod(int(flat), self.tw)
            if mag[r, c] <= 0:
                break
            out.append(r * self.tw + c)
        return out

    def _row_runs(self, dirty: list[int]) -> list[list[int]]:
        """Group magnitude-sorted dirty tiles into contiguous horizontal runs
        (per row). Sending whole runs keeps text lines coherent - no holes."""
        runs: list[list[int]] = []
        seen: set[int] = set()
        for i in dirty:
            if i in seen:
                continue
            ty, tx = divmod(i, self.tw)
            run = [i]
            seen.add(i)
            # extend right/left within the row
            for d in (1, -1):
                tx2 = tx + d
                while 0 <= tx2 < self.tw:
                    j = ty * self.tw + tx2
                    if j in dirty and j not in seen:
                        run.append(j)
                        seen.add(j)
                        tx2 += d
                    else:
                        break
            runs.append(run)
        return runs

    def _count_dirty(self, a: np.ndarray, b: np.ndarray) -> int:
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > DIRTY_D
        ph, pw = self.th * TILE, self.tw * TILE
        pad = np.zeros((ph, pw), bool)
        pad[: self.h, : self.w] = d
        counts = pad.reshape(self.th, TILE, self.tw, TILE).sum(axis=(1, 3))
        return int((counts >= DIRTY_PX_MIN).sum())

    def _tile(self, img: np.ndarray, i: int) -> np.ndarray:
        ty, tx = divmod(i, self.tw)
        y0, x0 = ty * TILE, tx * TILE
        return img[y0:min(y0 + TILE, self.h), x0:min(x0 + TILE, self.w)]

    def _apply_tiles(self, frame: np.ndarray, idxs: list[int]) -> None:
        for i in idxs:
            ty, tx = divmod(i, self.tw)
            y0, x0 = ty * TILE, tx * TILE
            y1, x1 = min(y0 + TILE, self.h), min(x0 + TILE, self.w)
            t = self._tile(frame, i)
            self.recon[y0:y1, x0:x1] = t
            self.sent_src[y0:y1, x0:x1] = t
            self.age[i] = 0

    def _freeze(self) -> bytes:
        self.stats["frozen"] += 1
        self.stats["skip"] += 1
        return bytes([P_SKIP])

    # ---------- main ----------
    def encode(self, frame: np.ndarray, fi: int) -> bytes:
        try:
            return self._encode_inner(frame, fi)
        finally:
            self.prev_src = frame

    def _encode_inner(self, frame: np.ndarray, fi: int) -> bytes:
        t = fi / self.fps
        self.stats["frames"] += 1
        credit = self.budget.credit(t)

        # --- keyframe (mandatory) ---
        if self.sent_src is None:
            q = 85
            while q >= 50:
                blob = _webp(frame, q)
                if len(blob) <= max(credit, 0) or q == 50:
                    break
                q -= 5
            self._note_full(q, len(blob))
            pkt = bytes([P_KEY]) + blob
            self.sent_src = frame.copy()
            self.recon = _webp_dec(blob)
            self.age[:] = 0
            self.budget.pay(len(pkt))
            self.stats["key"] += 1
            self.stats["bytes"] += len(pkt)
            self.stats["q_hist"][q] = self.stats["q_hist"].get(q, 0) + 1
            return pkt

        # --- dirty tiles vs last sent source ---
        dirty = self._dirty_tiles(frame)
        # dirty-age: frames each tile has been continuously dirty (0 when
        # clean). Unlike send-age this is not inflated by long clean spells,
        # so it is a true "how long has this update been waiting" signal.
        self.age = np.where(self._dirty2d.ravel(), self.age + 1, 0).astype(np.uint32)
        self.stats["dirty_sum"] += len(dirty)
        if not dirty:
            self.stats["skip"] += 1
            return bytes([P_SKIP])
        n = len(dirty)
        self._build_probes(frame, dirty)

        # --- decide (v4) ---
        # 1) bbox snapshot: WebP of ONLY the changed region - much cheaper
        #    than a whole frame when the change is localized, so credit
        #    lasts longer and stale windows shrink. Only when the change
        #    is dense inside its bbox (otherwise rows are cheaper).
        # 2) all dirty tiles if they fit completely.
        # 3) hold (bank credit) ONLY for dense big/churny changes where a
        #    snapshot is the right eventual answer.
        # 4) otherwise stream whole row-runs (coherent lines, no holes).
        dense = self._crop_fill >= 0.35
        use_snap = False
        q_send: int | None = None

        if dense:
            # --- v5: progressive bands, no multi-second holds ---
            # send the LARGEST tile-aligned top band of the bbox that fits
            # credit, at the best quality that fits. The band shrinks the
            # remaining bbox top-down, so the viewer watches the update
            # paint in instead of staring at a frozen screen.
            x0, y0, x1, y1 = self._crop_geom
            bh_total = y1 - y0
            if credit >= self._full_block:
                # full-bbox first when affordable; partial bands target the
                # OLDEST dirty rows (age map) so churn at the top cannot
                # pin the band forever while the bottom rots stale.
                dirty_arr = np.asarray(dirty, dtype=np.int64)
                tys = dirty_arr // self.tw
                txs = dirty_arr % self.tw
                age2d = self.age.reshape(self.th, self.tw)
                best_ty, best_age = int(tys[0]), -1
                for ty in np.unique(tys):
                    m = tys == ty
                    a = int(age2d[ty, txs[m]].max())
                    if a > best_age:
                        best_age, best_ty = a, int(ty)
                bbox_ty0, bbox_ty1 = y0 // TILE, (y1 + TILE - 1) // TILE
                n_rows_bbox = bbox_ty1 - bbox_ty0
                for frac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03):
                    n_rows = max(1, int(n_rows_bbox * frac))
                    if frac >= 1.0:
                        ty0b, ty1b = bbox_ty0, bbox_ty1
                        bx0, bx1 = x0, x1
                    else:
                        ty0b = best_ty
                        if ty0b + n_rows > bbox_ty1:
                            ty0b = bbox_ty1 - n_rows
                        if ty0b < bbox_ty0:
                            ty0b = bbox_ty0
                        ty1b = min(ty0b + n_rows, bbox_ty1)
                        sel = (tys >= ty0b) & (tys < ty1b)
                        if not sel.any():
                            continue
                        bx0 = int(txs[sel].min()) * TILE
                        bx1 = min(int(txs[sel].max()) * TILE + TILE, self.w)
                    y0b, y1b = ty0b * TILE, min(ty1b * TILE, self.h)
                    band = frame[y0b:y1b, bx0:bx1]
                    probe = len(_webp(band[::4, ::4], 82))
                    for q in QUALITY_LADDER:
                        if q < Q_FLOOR_FULL:
                            break
                        est = int(probe * _FULL_F[q] * _EST_SAFETY) + 9
                        if est > credit:
                            continue
                        blob = _webp(band, q)
                        if len(blob) + 9 > credit:
                            # estimate missed: block retries until credit
                            # reaches what this content actually costs
                            self._full_block = int((len(blob) + 9) * 1.02)
                            break
                        pkt = (bytes([P_CROP])
                               + struct.pack("<HHHH", bx0, y0b,
                                             bx1 - bx0, y1b - y0b)
                               + blob)
                        rec = _webp_dec(blob)
                        self.recon[y0b:y1b, bx0:bx1] = rec
                        self.sent_src[y0b:y1b, bx0:bx1] = band
                        self.age.reshape(self.th, self.tw)[ty0b:ty1b,
                            bx0 // TILE:(bx1 + TILE - 1) // TILE] = 0
                        self._full_block = 0
                        q_send = q
                        use_snap = True
                        mode = "crop"
                        break
                    if use_snap:
                        break
            if not use_snap:
                return self._freeze()      # credit < smallest band: bank a beat
        else:
            # --- scattered change: whole row-runs (skip the gaps) ---
            q_tiles: int | None = None
            for q in QUALITY_LADDER:
                if q >= Q_FLOOR_TILE and n * (self._est_tile(q) + 4) + 2 <= credit:
                    q_tiles = q
                    break
            if q_tiles is not None:
                q_send = q_tiles
            else:
                for q in QUALITY_LADDER:
                    if (q >= Q_FLOOR_TILE
                            and credit >= 3 * (self._est_tile(q) + 4) + 3):
                        q_send = q
                        break
                if q_send is None:
                    return self._freeze()

        if not use_snap:
            # Starvation fix: try runs OLDEST-FIRST (age map, stable so
            # magnitude breaks ties). Previously runs were tried in pure
            # magnitude order: wide lines (est > typical credit) were
            # skipped every frame while smaller runs drained the credit,
            # so the same rows could stay stale for 20+ seconds.
            # Also: pack a run's PREFIX when the whole run doesn't fit
            # (the suffix stays dirty and continues next frame) instead of
            # dropping the whole run - a line draws in over a frame or two
            # rather than never arriving.
            runs = self._row_runs(dirty)
            if len(runs) > 1:
                run_age = np.array(
                    [int(self.age[run].max()) for run in runs], dtype=np.int64)
                runs = [runs[i] for i in np.argsort(-run_age, kind="stable")]
            sent_idxs: list[int] = []
            parts: list[bytes] = []
            total = 1 + 2                      # actual packet bytes so far
            for run in runs:
                if credit - total < 512:
                    break
                for i in run:
                    tb = _webp(self._tile(frame, i), q_send)
                    self._note_tile(q_send, len(tb))
                    if total + 4 + len(tb) > credit:
                        break                  # prefix sent; suffix next frame
                    total += 4 + len(tb)
                    parts.append(struct.pack("<HH", i, len(tb)))
                    parts.append(tb)
                    sent_idxs.append(i)
                if credit - total < 512:
                    break
            if not sent_idxs:
                return self._freeze()
            pkt = bytes([P_TILES]) + struct.pack("<H", len(sent_idxs)) + b"".join(parts)
            self._apply_tiles(frame, sent_idxs)
            mode = "tiles"

        self.budget.pay(len(pkt))
        self.stats["bytes"] += len(pkt)
        self.stats[mode] = self.stats.get(mode, 0) + 1
        self.stats["q_hist"][q_send] = self.stats["q_hist"].get(q_send, 0) + 1
        return pkt


class ScreenDecoder:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.tw = (w + TILE - 1) // TILE
        self.th = (h + TILE - 1) // TILE
        self.frame: np.ndarray | None = None

    def decode(self, pkt: bytes) -> np.ndarray | None:
        typ = pkt[0]
        if typ == P_SKIP:
            return self.frame
        if typ in (P_KEY, P_FULL):
            self.frame = _webp_dec(pkt[1:])
            return self.frame
        if typ == P_CROP:
            x, y, w, h = struct.unpack_from("<HHHH", pkt, 1)
            img = _webp_dec(pkt[9:])
            if self.frame is None:
                self.frame = np.zeros((self.h, self.w, 3), np.uint8)
            self.frame[y:y + h, x:x + w] = img
            return self.frame
        if typ == P_TILES:
            (n,) = struct.unpack_from("<H", pkt, 1)
            off = 3
            for _ in range(n):
                i, ln = struct.unpack_from("<HH", pkt, off)
                off += 4
                blob = pkt[off:off + ln]
                off += ln
                t = _webp_dec(blob)
                ty, tx = divmod(i, self.tw)
                y0, x0 = ty * TILE, tx * TILE
                y1, x1 = min(y0 + TILE, self.h), min(x0 + TILE, self.w)
                if self.frame is None:
                    self.frame = np.zeros((self.h, self.w, 3), np.uint8)
                self.frame[y0:y1, x0:x1] = t
            return self.frame
        raise RuntimeError(f"bad packet type {typ}")
