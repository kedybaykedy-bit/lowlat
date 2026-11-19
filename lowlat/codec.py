"""Lossless low-bitrate video codec: closed-loop inter pred + neural CDF + rANS.

Bitstream (per frame):
  u8  flags (bit0=keyframe)
  u16 n_dirty  (0 for pure skip)
  if keyframe: full raw path via tile skip-delta (or heavy inter from black)
  else:
    for each dirty tile (raster order, from dirty bitmap RLE):
      i8 mv_y, i8 mv_x   (clamped ±16)
      residuals coded with shared neural CDF via rANS
Dirty bitmap: RLE over nt*nx bools.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field

import numpy as np
import torch

from .ans import Decoder, DecoderB, Encoder, EncoderB, freqs_from_probs
from .model import N_BINS, RES_MIN, ResidualCDF, residual_to_bin
from .rate import TokenBucket
from .tiles import TILE, border_context, dirty_mask, motion_search, predict_tile


def _rle_bools(mask: np.ndarray) -> bytes:
    flat = mask.reshape(-1)
    out = bytearray()
    i = 0
    n = len(flat)
    while i < n:
        v = flat[i]
        j = i + 1
        while j < n and flat[j] == v:
            j += 1
        # pack run length (max 65535)
        run = j - i
        while run > 0:
            r = min(run, 65535)
            out.append(1 if v else 0)
            out += struct.pack("<H", r)
            run -= r
        i = j
    return bytes(out)


def _unrle_bools(data: bytes, count: int) -> np.ndarray:
    flat = np.zeros(count, dtype=bool)
    pos = 0
    idx = 0
    while idx < count and pos + 3 <= len(data):
        v = data[pos] != 0
        run = struct.unpack_from("<H", data, pos + 1)[0]
        pos += 3
        end = min(count, idx + run)
        flat[idx:end] = v
        idx = end
    return flat


@dataclass
class FrameStats:
    bytes: int = 0
    dirty: int = 0
    total_tiles: int = 0
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    waited_s: float = 0.0


class LowLatCodec:
    def __init__(
        self,
        h: int,
        w: int,
        model: ResidualCDF | None = None,
        device: str = "cpu",
        rate_bps: float = 10 * 1024,
        enable_me: bool = True,
        tile: int = TILE,
    ):
        self.h, self.w = h, w
        self.tile = tile
        self.nt = (h + tile - 1) // tile
        self.nx = (w + tile - 1) // tile
        self.device = device
        self.enable_me = enable_me
        self.model = model.to(device).eval() if model is not None else None
        self.bucket = TokenBucket(rate_bps, burst=rate_bps * 0.25)
        self.rate_bps = rate_bps

        self.recon = np.zeros((h, w, 3), dtype=np.uint8)
        self.prev_res = np.zeros((h, w, 3), dtype=np.float32)  # last residuals (scaled)
        self._freqs_cache: np.ndarray | None = None
        self._tile_coords = [
            (ty * tile, tx * tile)
            for ty in range(self.nt)
            for tx in range(self.nx)
        ]

    def reset(self) -> None:
        self.recon[:] = 0
        self.prev_res[:] = 0

    # ---------- neural helpers ----------
    @torch.no_grad()
    def _tile_logits(
        self,
        pred: np.ndarray,
        border: np.ndarray,
        prev_r: np.ndarray,
        motion_e: np.ndarray | None = None,
        pad_to: int | None = None,
    ) -> np.ndarray:
        """Return (h,w,3,N_BINS) logits. If pad_to set, input padded to pad_to² then sliced back."""
        assert self.model is not None
        device = next(self.model.parameters()).device
        hh, ww = pred.shape[:2]
        pred_n = pred.astype(np.float32) / 255.0
        bord_n = border.astype(np.float32)
        pr_n = prev_r.astype(np.float32) / 255.0
        me = np.abs(pred.astype(np.float32) - prev_r).mean(axis=2, keepdims=True) / 255.0
        ctx = np.concatenate(
            [pred_n, bord_n, pr_n, me.astype(np.float32)], axis=2
        ).transpose(2, 0, 1)
        if pad_to is not None and (hh != pad_to or ww != pad_to):
            ctx_p = np.zeros((10, pad_to, pad_to), dtype=np.float32)
            ctx_p[:, :hh, :ww] = ctx
            ctx = ctx_p
        ctx_t = torch.from_numpy(ctx).unsqueeze(0).to(device)
        logits = self.model(ctx_t)  # (1, 3*N_BINS, H, W)
        b, c, H, W = logits.shape
        logits = logits.view(b, 3, N_BINS, H, W)
        # (3, N_BINS, H, W) → (H, W, 3, N_BINS)
        out = logits[0].permute(2, 3, 0, 1).cpu().numpy().astype(np.float32)
        return out[:hh, :ww]

    @staticmethod
    def _freqs_from_logits(logits: np.ndarray) -> np.ndarray:
        """logits (..., C) → freqs (..., C) uint32 summing exactly to 1<<16 on last axis."""
        # float32 exp in-place; avoid copy when possible
        x = logits.astype(np.float32, copy=False)
        x = x - x.max(axis=-1, keepdims=True)
        np.exp(x, out=x)
        p = x
        s = p.sum(axis=-1, keepdims=True)
        p /= s
        shape = p.shape
        flat = p.reshape(-1, shape[-1])
        n_sym = flat.shape[-1]
        TOTAL = 65536
        remain = TOTAL - n_sym
        # vectorized floor + argmax dump
        scaled = flat * np.float32(remain)
        freqs = scaled.astype(np.int32) + 1
        need = (TOTAL - freqs.sum(axis=1)).astype(np.int32)
        am = np.argmax(flat, axis=1)
        np.add.at(freqs, (np.arange(flat.shape[0]), am), need)
        # rare safety
        sums = freqs.sum(axis=1)
        bad = np.where(sums != TOTAL)[0]
        for i in bad:
            row = freqs[i]
            d = TOTAL - int(row.sum())
            row[am[i]] += d
            freqs[i] = row
        return freqs.reshape(shape).astype(np.uint32)

    # ---------- encode / decode ----------
    def encode_frame(
        self, frame: np.ndarray, keyframe: bool = False, wait_rate: bool = True
    ) -> tuple[bytes, FrameStats]:
        import time as _t

        t0 = _t.perf_counter()
        st = FrameStats()
        assert frame.shape == (self.h, self.w, 3)

        if keyframe:
            self.reset()
            # Keyframe: zlib of raw RGB (lossless, fast, good for screen content).
            # Flag bit1 = zlib keyframe body follows header instead of tile path.
            import zlib as _zlib

            body = _zlib.compress(frame.tobytes(), 6)
            buf = bytearray()
            buf.append(0b11)  # keyframe | zlib
            buf += struct.pack("<I", len(body))
            buf += body
            waited = 0.0
            if wait_rate:
                import time as _t2

                tw = _t2.perf_counter()
                self.bucket.wait_take(len(buf), timeout=30.0)
                waited = _t2.perf_counter() - tw
            st.bytes = len(buf)
            st.dirty = self.nt * self.nx
            st.total_tiles = self.nt * self.nx
            st.waited_s = waited
            st.encode_ms = (_t.perf_counter() - t0) * 1000
            # encoder recon = frame
            self.recon = frame.copy()
            self.prev_res[:] = 0
            return bytes(buf), st

        ref = self.recon
        dmask = (
            dirty_mask(frame, ref, self.tile)
            if not keyframe
            else np.ones((self.nt, self.nx), dtype=bool)
        )

        buf = bytearray()
        flags = 1 if keyframe else 0
        buf.append(flags)
        n_dirty = int(dmask.sum())
        buf += struct.pack("<H", n_dirty)
        buf += _rle_bools(dmask)

        has_model = self.model is not None
        residuals_for_prev = np.zeros((self.h, self.w, 3), dtype=np.float32)
        new_recon = ref.copy()

        # Pass 1: predictions + residuals for all dirty tiles
        tiles = []  # (y0,x0,h,w,mv_y,mv_x,pred,resid)
        for ty in range(self.nt):
            for tx in range(self.nx):
                if not dmask[ty, tx]:
                    continue
                y0, x0 = ty * self.tile, tx * self.tile
                y1 = min(y0 + self.tile, self.h)
                x1 = min(x0 + self.tile, self.w)
                h, w = y1 - y0, x1 - x0
                cur = frame[y0:y1, x0:x1]

                if keyframe:
                    mv_y = mv_x = 0
                    pred = np.zeros_like(cur)
                else:
                    if self.enable_me:
                        mv_y, mv_x, _ = motion_search(cur, ref, y0, x0)
                        mv_y = int(np.clip(mv_y, -127, 127))
                        mv_x = int(np.clip(mv_x, -127, 127))
                    else:
                        mv_y = mv_x = 0
                    pred = predict_tile(ref, y0, x0, h, w, mv_y, mv_x)

                resid = cur.astype(np.int16) - pred.astype(np.int16)
                residuals_for_prev[y0:y1, x0:x1] = resid
                tiles.append((y0, x0, h, w, mv_y, mv_x, pred, resid))
                new_recon[y0:y1, x0:x1] = cur

        # Pass 2+3: chunked neural → ANS (never hold all logits)
        def _ctx_of(y0, x0, h, w, mv_y, mv_x, pred) -> np.ndarray:
            border = border_context(ref, y0, x0, h, w, mv_y, mv_x)
            pr = self.prev_res[y0 : y0 + h, x0 : x0 + w]
            me = np.abs(pred.astype(np.float32) - pr).mean(axis=2, keepdims=True) / 255.0
            ctx = np.concatenate(
                [
                    pred.astype(np.float32) / 255.0,
                    border,
                    np.clip(pr / 255.0, -1, 1),
                    me,
                ],
                axis=2,
            ).transpose(2, 0, 1)
            if h != self.tile or w != self.tile:
                ctx_p = np.zeros((10, self.tile, self.tile), dtype=np.float32)
                ctx_p[:, :h, :w] = ctx
                ctx = ctx_p
            return ctx

        chunk_sz = 16
        for i0 in range(0, len(tiles), chunk_sz):
            chunk_tiles = tiles[i0 : i0 + chunk_sz]
            logits_chunk = None
            if has_model:
                ctx_list = [_ctx_of(t[0], t[1], t[2], t[3], t[4], t[5], t[6]) for t in chunk_tiles]
                logits_chunk = self._batch_logits(ctx_list)
            for j, (y0, x0, h, w, mv_y, mv_x, pred, resid) in enumerate(chunk_tiles):
                buf += struct.pack("bb", mv_y, mv_x)
                if has_model:
                    lg = logits_chunk[j, :, :, :h, :w].transpose(2, 3, 0, 1)
                    freqs = self._freqs_from_logits(lg)
                    enc = EncoderB(hint=h * w * 3 * 2)
                    for ch in (2, 1, 0):
                        bins = (
                            resid[:, :, ch].astype(np.int32) - RES_MIN
                        ).clip(0, N_BINS - 1).reshape(-1)
                        frow_all = np.ascontiguousarray(
                            freqs[:, :, ch].reshape(-1, N_BINS)
                        )
                        csum = np.cumsum(frow_all, axis=1, dtype=np.uint64)
                        idx = np.arange(bins.size)
                        starts = np.where(
                            bins > 0, csum[idx, np.maximum(bins - 1, 0)], 0
                        ).astype(np.uint32)
                        fsel = frow_all[idx, bins].astype(np.uint32)
                        enc.put_batch(starts, fsel)
                    payload = enc.flush()
                    enc.close()
                    buf += struct.pack("<I", len(payload))
                    buf += payload
                else:
                    raw = self._pack_raw(resid)
                    buf += struct.pack("<I", len(raw))
                    buf += raw
            del logits_chunk

        # rate pacing
        waited = 0.0
        if wait_rate and len(buf) > 0:
            import time as _t2

            tw = _t2.perf_counter()
            ok = self.bucket.wait_take(len(buf), timeout=30.0)
            waited = _t2.perf_counter() - tw

        self.recon = new_recon
        self.prev_res = residuals_for_prev
        st.bytes = len(buf)
        st.dirty = n_dirty
        st.total_tiles = self.nt * self.nx
        st.waited_s = waited
        st.encode_ms = (_t.perf_counter() - t0) * 1000
        return bytes(buf), st

    def _prev_res_tile(self, y0: int, x0: int, h: int, w: int) -> np.ndarray:
        return self.prev_res[y0 : y0 + h, x0 : x0 + w]

    @torch.no_grad()
    def _batch_logits(self, ctx_list: list[np.ndarray]) -> np.ndarray:
        """Run model on stacked (N,10,TILE,TILE) contexts with chunking. Returns (N,3,C,T,T)."""
        batch = np.stack(ctx_list, axis=0)
        device = next(self.model.parameters()).device
        B = batch.shape[0]
        # keep peak logits ≈ 8M floats
        max_chunk = max(1, min(B, (8 << 20) // (3 * N_BINS * self.tile * self.tile)))
        outs = []
        for s in range(0, B, max_chunk):
            chunk = torch.from_numpy(batch[s : s + max_chunk]).to(device)
            logits_t = self.model(chunk)
            b = logits_t.shape[0]
            logits_t = logits_t.view(b, 3, N_BINS, logits_t.shape[2], logits_t.shape[3])
            outs.append(logits_t.cpu().numpy())
            del chunk, logits_t
        return np.concatenate(outs, axis=0)

    @staticmethod
    def _pack_raw(resid: np.ndarray) -> bytes:
        # zigzag + varint per channel, naive but ok
        r = resid.reshape(-1).astype(np.int32)
        zz = (r << 1) ^ (r >> 31)
        out = bytearray()
        for v in zz:
            v = int(v)
            while v >= 0x80:
                out.append((v & 0x7F) | 0x80)
                v >>= 7
            out.append(v)
        return bytes(out)

    @staticmethod
    def _unpack_raw(data: bytes, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int32)
        i = 0
        pos = 0
        while i < n:
            v = 0
            shift = 0
            while True:
                b = data[pos]
                pos += 1
                v |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            out[i] = v
            i += 1
        r = (out >> 1) ^ -(out & 1)
        return r

    def decode_frame(self, data: bytes) -> tuple[np.ndarray, FrameStats]:
        import time as _t

        t0 = _t.perf_counter()
        st = FrameStats()
        bio = io.BytesIO(data)
        flags = bio.read(1)[0]
        keyframe = bool(flags & 1)
        zlib_kf = bool(flags & 2)
        if zlib_kf:
            import zlib as _zlib

            (blen,) = struct.unpack("<I", bio.read(4))
            raw = _zlib.decompress(bio.read(blen))
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 3).copy()
            self.recon = frame
            self.prev_res[:] = 0
            st.bytes = len(data)
            st.dirty = self.nt * self.nx
            st.total_tiles = self.nt * self.nx
            st.decode_ms = (_t.perf_counter() - t0) * 1000
            return frame, st
        (n_dirty,) = struct.unpack("<H", bio.read(2))
        nt_total = self.nt * self.nx
        # read RLE until we have nt_total bools
        mask_flat = np.zeros(nt_total, dtype=bool)
        idx = 0
        while idx < nt_total:
            chunk = bio.read(3)
            if len(chunk) < 3:
                break
            v = chunk[0] != 0
            run = struct.unpack("<H", chunk[1:3])[0]
            end = min(nt_total, idx + run)
            mask_flat[idx:end] = v
            idx = end
        dmask = mask_flat.reshape(self.nt, self.nx)

        ref = self.recon.copy() if not keyframe else np.zeros_like(self.recon)
        new_recon = ref.copy()  # must NOT alias ref — inter pred uses start-of-frame ref
        residuals_for_prev = np.zeros((self.h, self.w, 3), dtype=np.float32)

        has_model = self.model is not None
        decode_ms = 0.0

        # Pass A: parse all dirty tiles (mv + payload bytes) — no logits yet
        parsed = []  # (y0,x0,h,w,mv_y,mv_x,payload)
        for ty in range(self.nt):
            for tx in range(self.nx):
                if not dmask[ty, tx]:
                    continue
                y0, x0 = ty * self.tile, tx * self.tile
                y1 = min(y0 + self.tile, self.h)
                x1 = min(x0 + self.tile, self.w)
                h, w = y1 - y0, x1 - x0
                mv_y, mv_x = struct.unpack("bb", bio.read(2))
                if keyframe:
                    mv_y = mv_x = 0
                (plen,) = struct.unpack("<I", bio.read(4))
                payload = bio.read(plen)
                parsed.append((y0, x0, h, w, mv_y, mv_x, payload))

        # Pass B+C: chunked logits → ANS decode (same chunking as encoder)
        def _pred_of(y0, x0, h, w, mv_y, mv_x):
            if keyframe:
                return np.zeros((h, w, 3), dtype=np.uint8)
            return predict_tile(ref, y0, x0, h, w, mv_y, mv_x)

        def _ctx_of(y0, x0, h, w, mv_y, mv_x, pred) -> np.ndarray:
            border = border_context(ref, y0, x0, h, w, mv_y, mv_x)
            pr = self.prev_res[y0 : y0 + h, x0 : x0 + w]
            me = np.abs(pred.astype(np.float32) - pr).mean(axis=2, keepdims=True) / 255.0
            ctx = np.concatenate(
                [
                    pred.astype(np.float32) / 255.0,
                    border,
                    np.clip(pr / 255.0, -1, 1),
                    me,
                ],
                axis=2,
            ).transpose(2, 0, 1)
            if h != self.tile or w != self.tile:
                ctx_p = np.zeros((10, self.tile, self.tile), dtype=np.float32)
                ctx_p[:, :h, :w] = ctx
                ctx = ctx_p
            return ctx

        chunk_sz = 16
        for i0 in range(0, len(parsed), chunk_sz):
            chunk_p = parsed[i0 : i0 + chunk_sz]
            logits_chunk = None
            if has_model:
                ctx_list = []
                for y0, x0, h, w, mv_y, mv_x, payload in chunk_p:
                    pred = _pred_of(y0, x0, h, w, mv_y, mv_x)
                    ctx_list.append(_ctx_of(y0, x0, h, w, mv_y, mv_x, pred))
                logits_chunk = self._batch_logits(ctx_list)
            for j, (y0, x0, h, w, mv_y, mv_x, payload) in enumerate(chunk_p):
                y1 = min(y0 + self.tile, self.h)
                x1 = min(x0 + self.tile, self.w)
                pred = _pred_of(y0, x0, h, w, mv_y, mv_x)
                if has_model:
                    lg = logits_chunk[j, :, :, :h, :w].transpose(2, 3, 0, 1)
                    freqs = self._freqs_from_logits(lg)
                    dec = DecoderB(payload)
                    resid = np.empty((h, w, 3), dtype=np.int32)
                    npx = h * w
                    for ch in (0, 1, 2):
                        frow_all = np.ascontiguousarray(
                            freqs[:, :, ch].reshape(npx, N_BINS)
                        )
                        syms = dec.symbol_batch(frow_all)
                        resid[:, :, ch] = syms.reshape(h, w).astype(np.int32) + RES_MIN
                    dec.close()
                else:
                    resid = self._unpack_raw(payload, h * w * 3).reshape(h, w, 3)

                tile_rec = pred.astype(np.int16) + resid.astype(np.int16)
                tile_rec = np.clip(tile_rec, 0, 255).astype(np.uint8)
                new_recon[y0:y1, x0:x1] = tile_rec
                residuals_for_prev[y0:y1, x0:x1] = resid.astype(np.float32)
            del logits_chunk

        self.recon = new_recon
        self.prev_res = residuals_for_prev
        st.bytes = len(data)
        st.dirty = int(dmask.sum())
        st.total_tiles = nt_total
        st.decode_ms = (_t.perf_counter() - t0) * 1000
        return self.recon.copy(), st


def ssim_exact(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))
