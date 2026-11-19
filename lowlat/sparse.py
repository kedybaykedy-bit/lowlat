"""Lossless sparse-tile coding: exploit that screen-content residuals are ~all ZERO.

Container (all methods):
    u8   method_id   0 = A change-mask, 1 = B zlib(cur bytes),
                     2 = C zigzag+varint(residual) + zlib
    u32  len         payload length, little endian
    ...  payload

Method A (change mask):
    payload = u8 mask_mode (0 = packbits, 1 = RLE; encoder picks the smaller)
              + mask bytes  (packbits: ceil(h*w/8) known from pred;
                             RLE: (u8 value, u16 run) pairs, self-delimiting
                             once h*w bits are filled)
              + zigzag+varint of the nonzero residual values (raster order,
              3 channels together per changed pixel).
Method B:
    payload = zlib(clip(pred+resid).tobytes())   # raw reconstructed bytes
Method C:
    payload = zlib(zigzag+varint of the FULL residual, h*w*3 values)

All methods are bit-exact: decode returns uint8 clip(pred + resid, 0, 255).
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

METHOD_A_MASK = 0
METHOD_B_ZLIB_CUR = 1
METHOD_C_VARINT_ZLIB = 2
_HDR = struct.Struct("<BI")  # method_id, payload length


# ---------- zigzag / varint (vectorized) ----------
def _zigzag_encode(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    return (v << 1) ^ (v >> 63)


def _zigzag_decode(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.int64)
    return (u >> 1) ^ -(u & 1)


def _varint_encode(vals: np.ndarray) -> bytes:
    """Unsigned LEB128. vals must be >= 0."""
    v = np.asarray(vals, dtype=np.int64)
    n = v.size
    if n == 0:
        return b""
    nb = np.ones(n, dtype=np.int64)  # bytes per value
    t = v.copy()
    while True:
        t >>= 7
        m = t != 0
        if not m.any():
            break
        nb += m
    offs = np.cumsum(nb) - nb
    total = int(nb.sum())
    el = np.repeat(np.arange(n, dtype=np.int64), nb)
    j = np.arange(total, dtype=np.int64) - offs[el]
    b = ((v[el] >> (7 * j)) & 0x7F).astype(np.uint8)
    b |= np.where(j + 1 < nb[el], 0x80, 0).astype(np.uint8)
    return b.tobytes()


def _varint_decode(data: bytes, n: int) -> np.ndarray:
    """Decode exactly n unsigned LEB128 values; data must be fully consumed."""
    if n == 0:
        if len(data):
            raise ValueError("varint stream: trailing bytes after 0 values")
        return np.zeros(0, dtype=np.int64)
    arr = np.frombuffer(data, dtype=np.uint8)
    ends = np.flatnonzero((arr & 0x80) == 0)
    if ends.size < n:
        raise ValueError("varint stream truncated")
    ends = ends[:n]
    last = int(ends[-1]) + 1
    if last != arr.size:
        raise ValueError("varint stream: trailing bytes")
    starts = np.empty(n, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    pos = np.arange(last, dtype=np.int64)
    el = np.searchsorted(starts, pos, side="right") - 1
    if el[0] < 0:
        raise ValueError("varint stream empty")
    j = pos - starts[el]
    val = (arr[:last].astype(np.int64) & 0x7F) << (7 * j)
    out = np.zeros(n, dtype=np.int64)
    np.add.at(out, el, val)
    return out


# ---------- helpers ----------
def _check(resid: np.ndarray, pred: np.ndarray) -> np.ndarray:
    resid = np.asarray(resid)
    pred = np.asarray(pred)
    if resid.ndim != 3 or resid.shape[2] != 3:
        raise ValueError(f"resid must be (h,w,3), got {resid.shape}")
    if pred.shape != resid.shape:
        raise ValueError(f"pred shape {pred.shape} != resid shape {resid.shape}")
    return resid.astype(np.int16, copy=False)


def _recon(resid: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """uint8 clip(pred + resid, 0, 255) — the bit-exact reconstruction."""
    out = pred.astype(np.int16) + resid.astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rle_encode_mask(mask: np.ndarray) -> bytes:
    flat = mask.reshape(-1)
    out = bytearray()
    i, n = 0, flat.size
    while i < n:
        v = bool(flat[i])
        j = i + 1
        while j < n and bool(flat[j]) == v:
            j += 1
        run = j - i
        while run > 0:
            r = min(run, 65535)
            out.append(1 if v else 0)
            out += struct.pack("<H", r)
            run -= r
        i = j
    return bytes(out)


def _rle_decode_mask(data: bytes, npx: int) -> tuple[np.ndarray, int]:
    """Decode runs until npx bits are filled. Returns (mask, bytes_consumed)."""
    flat = np.zeros(npx, dtype=bool)
    idx, pos = 0, 0
    while idx < npx:
        if pos + 3 > len(data):
            raise ValueError("method A RLE mask truncated")
        v = data[pos] != 0
        run = struct.unpack_from("<H", data, pos + 1)[0]
        pos += 3
        if run == 0 or idx + run > npx:
            raise ValueError("method A RLE mask bad run")
        flat[idx : idx + run] = v
        idx += run
    return flat, pos


def _encode_a(resid: np.ndarray) -> bytes:
    h, w, _ = resid.shape
    mask = np.any(resid != 0, axis=2)  # (h,w) True where ANY channel nonzero
    nz = resid[mask].reshape(-1)  # 3 values per changed pixel, raster order
    vals = _varint_encode(_zigzag_encode(nz))
    pb = np.packbits(mask.reshape(-1)).tobytes()  # ceil(h*w/8)
    rle = _rle_encode_mask(mask)
    if len(pb) <= len(rle):
        return b"\x00" + pb + vals
    return b"\x01" + rle + vals


def _decode_a(payload: bytes, pred: np.ndarray) -> np.ndarray:
    h, w, _ = pred.shape
    npx = h * w
    if not payload:
        raise ValueError("method A payload empty")
    mode, rest = payload[0], payload[1:]
    if mode == 0:
        mlen = (npx + 7) // 8
        if len(rest) < mlen:
            raise ValueError("method A payload shorter than mask")
        bits = np.unpackbits(np.frombuffer(rest[:mlen], dtype=np.uint8))[:npx]
        mask = bits.astype(bool)
        vstart = rest[mlen:]
    elif mode == 1:
        mask, used = _rle_decode_mask(rest, npx)
        vstart = rest[used:]
    else:
        raise ValueError(f"method A mask_mode {mode} unknown")
    k = int(mask.sum())
    vals = _zigzag_decode(_varint_decode(vstart, 3 * k))
    resid = np.zeros((npx, 3), dtype=np.int16)
    if k:
        resid[mask] = vals.astype(np.int16).reshape(k, 3)
    return _recon(resid.reshape(h, w, 3), pred)


def _encode_b(resid: np.ndarray, pred: np.ndarray) -> bytes:
    cur = _recon(resid, pred)
    return zlib.compress(cur.tobytes(), 6)


def _decode_b(payload: bytes, pred: np.ndarray) -> np.ndarray:
    raw = zlib.decompress(payload)
    h, w, _ = pred.shape
    if len(raw) != h * w * 3:
        raise ValueError("method B payload has wrong size")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()


def _encode_c(resid: np.ndarray) -> bytes:
    zz = _zigzag_encode(resid.reshape(-1))
    return zlib.compress(_varint_encode(zz), 6)


def _decode_c(payload: bytes, pred: np.ndarray) -> np.ndarray:
    h, w, _ = pred.shape
    raw = zlib.decompress(payload)
    zz = _varint_decode(raw, h * w * 3)
    resid = _zigzag_decode(zz).astype(np.int16).reshape(h, w, 3)
    return _recon(resid, pred)


# ---------- public API ----------
def encode_sparse_methods(resid: np.ndarray, pred: np.ndarray) -> dict[int, bytes]:
    """Encode `resid` under each method. Returns {method_id: payload} (no header)."""
    resid = _check(resid, pred)
    return {
        METHOD_A_MASK: _encode_a(resid),
        METHOD_B_ZLIB_CUR: _encode_b(resid, pred),
        METHOD_C_VARINT_ZLIB: _encode_c(resid),
    }


def _container(method_id: int, payload: bytes) -> bytes:
    return _HDR.pack(method_id, len(payload)) + payload


def encode_sparse_residual(resid: np.ndarray, pred: np.ndarray) -> bytes:
    """Best-of-A/B/C container for an int16 residual (actual = cur - pred)."""
    methods = encode_sparse_methods(resid, pred)
    mid, payload = min(methods.items(), key=lambda kv: len(kv[1]))
    return _container(mid, payload)


def encode_tile_auto(cur: np.ndarray, pred: np.ndarray) -> bytes:
    """Same container built directly from a uint8 tile and its prediction."""
    cur = np.asarray(cur, dtype=np.uint8)
    pred = np.asarray(pred)
    if cur.shape != pred.shape:
        raise ValueError(f"cur shape {cur.shape} != pred shape {pred.shape}")
    resid = cur.astype(np.int16) - pred.astype(np.int16)
    return encode_sparse_residual(resid, pred)


def decode_sparse_residual(data: bytes, pred: np.ndarray) -> np.ndarray:
    """Inverse of encode_sparse_residual / encode_tile_auto → (h,w,3) uint8."""
    if len(data) < _HDR.size:
        raise ValueError("container too short")
    mid, plen = _HDR.unpack_from(data, 0)
    payload = bytes(data[_HDR.size : _HDR.size + plen])
    if len(payload) != plen:
        raise ValueError("container payload truncated")
    if mid == METHOD_A_MASK:
        return _decode_a(payload, pred)
    if mid == METHOD_B_ZLIB_CUR:
        return _decode_b(payload, pred)
    if mid == METHOD_C_VARINT_ZLIB:
        return _decode_c(payload, pred)
    raise ValueError(f"unknown method_id {mid}")


__all__ = [
    "METHOD_A_MASK",
    "METHOD_B_ZLIB_CUR",
    "METHOD_C_VARINT_ZLIB",
    "encode_sparse_methods",
    "encode_sparse_residual",
    "encode_tile_auto",
    "decode_sparse_residual",
]
