"""ctypes wrapper for rANS + fast static/dynamic frequency helpers."""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "ans.c"
_LIB = _DIR / "libans.so"


def _ensure_lib() -> ctypes.CDLL:
    if not _LIB.exists() or _SRC.stat().st_mtime > _LIB.stat().st_mtime:
        subprocess.check_call(
            ["gcc", "-O3", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)]
        )
    return ctypes.CDLL(str(_LIB))


_lib = _ensure_lib()
_lib.enc_new.restype = ctypes.c_void_p
_lib.enc_new.argtypes = [ctypes.c_size_t]
_lib.enc_put.restype = ctypes.c_int
_lib.enc_put.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
_lib.enc_flush.restype = ctypes.c_int
_lib.enc_flush.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.enc_free.argtypes = [ctypes.c_void_p]
_lib.dec_new.restype = ctypes.c_void_p
_lib.dec_new.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.dec_slot.restype = ctypes.c_uint32
_lib.dec_slot.argtypes = [ctypes.c_void_p]
_lib.dec_advance.restype = ctypes.c_int
_lib.dec_advance.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
_lib.dec_free.argtypes = [ctypes.c_void_p]

TOTAL = 1 << 16


def freqs_from_probs(probs: np.ndarray, min_freq: int = 1) -> np.ndarray:
    """Quantize probabilities to integer freqs summing exactly to TOTAL."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 1e-12)
    p = p / p.sum()
    n = p.shape[-1]
    freqs = np.maximum((p * TOTAL).astype(np.int64), min_freq)
    # Fix sum
    diff = int(TOTAL - freqs.sum())
    if diff != 0:
        order = np.argsort(-p.ravel() if freqs.ndim > 1 else -p)
        if freqs.ndim > 1:
            flat = freqs.reshape(-1, n)
            ordr = np.argsort(-flat, axis=-1)
            # adjust per-row for 2D batch
            for row in range(flat.shape[0]):
                d = TOTAL - int(flat[row].sum())
                i = 0
                while d != 0:
                    idx = ordr[row, i % n]
                    if d > 0:
                        flat[row, idx] += 1
                        d -= 1
                    else:
                        if flat[row, idx] > min_freq:
                            flat[row, idx] -= 1
                            d += 1
                    i += 1
                    if i > 4 * n:
                        # force by moving from largest
                        break
            freqs = flat.reshape(freqs.shape)
        else:
            i = 0
            while diff != 0 and i < 8 * n:
                idx = order[i % n]
                if diff > 0:
                    freqs[idx] += 1
                    diff -= 1
                else:
                    if freqs[idx] > min_freq:
                        freqs[idx] -= 1
                        diff += 1
                i += 1
    # safety
    s = int(freqs.sum())
    if s != TOTAL:
        freqs.flat[0] += TOTAL - s
        if freqs.flat[0] < 1:
            freqs.flat[0] = 1
            # steal from max
            j = int(np.argmax(freqs))
            freqs[j] -= 1 + (1 - int(freqs.flat[0]))
            freqs[0] = 1
    return freqs.astype(np.uint32)


class Encoder:
    def __init__(self, hint: int = 1 << 16):
        self._h = _lib.enc_new(hint)

    def put(self, start: int, freq: int) -> None:
        r = _lib.enc_put(self._h, start, freq)
        if r != 0:
            raise RuntimeError("enc_put failed")

    def put_symbol(self, symbol: int, freqs: np.ndarray) -> None:
        start = int(freqs[:symbol].sum())
        self.put(start, int(freqs[symbol]))

    def flush(self) -> bytes:
        buf = ctypes.create_string_buffer(1 << 20)
        n = ctypes.c_size_t(0)
        # grow if needed
        for cap in (1 << 20, 1 << 22, 1 << 24, 1 << 26):
            buf = ctypes.create_string_buffer(cap)
            r = _lib.enc_flush(self._h, buf, cap, ctypes.byref(n))
            if r == 0:
                return buf.raw[: n.value]
        raise RuntimeError("enc_flush overflow")

    def close(self) -> None:
        if self._h:
            _lib.enc_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Decoder:
    def __init__(self, data: bytes):
        self._buf = data  # keep ref
        self._h = _lib.dec_new(data, len(data))
        if not self._h:
            raise ValueError("bad rANS stream")

    def symbol(self, freqs: np.ndarray) -> int:
        slot = _lib.dec_slot(self._h)
        # binary search cumulative
        csum = np.cumsum(freqs)
        idx = int(np.searchsorted(csum, slot, side="right"))
        idx = min(idx, len(freqs) - 1)
        start = int(csum[idx - 1]) if idx > 0 else 0
        _lib.dec_advance(self._h, start, int(freqs[idx]))
        return idx

    def close(self) -> None:
        if self._h:
            _lib.dec_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

_lib.enc_put_batch.restype = ctypes.c_int
_lib.enc_put_batch.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
_lib.dec_symbol_batch.restype = ctypes.c_int
_lib.dec_symbol_batch.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
]


class EncoderB(Encoder):
    """Batch put: starts/freqs arrays for symbols in original order (C reverses)."""

    def put_batch(self, starts: np.ndarray, freqs: np.ndarray) -> None:
        starts = np.ascontiguousarray(starts, dtype=np.uint32)
        freqs = np.ascontiguousarray(freqs, dtype=np.uint32)
        r = _lib.enc_put_batch(self._h, starts, freqs, len(starts))
        if r != 0:
            raise RuntimeError("enc_put_batch failed")


class DecoderB(Decoder):
    def symbol_batch(self, freqs: np.ndarray) -> np.ndarray:
        """freqs: (n, alphabet) uint32 rows summing to TOTAL. Returns (n,) symbols."""
        freqs = np.ascontiguousarray(freqs, dtype=np.uint32)
        n, A = freqs.shape
        out = np.empty(n, dtype=np.uint32)
        r = _lib.dec_symbol_batch(self._h, freqs, A, n, out)
        if r != 0:
            raise RuntimeError("dec_symbol_batch failed")
        return out
