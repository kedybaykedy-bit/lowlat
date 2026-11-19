"""ctypes wrapper for fast frequency quantization (largest remainder, sum = 65536).

C core: freq.c -> libfreq.so. Falls back to a pure-numpy implementation of the
same algorithm when the shared library cannot be loaded/built.
"""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "freq.c"
_LIB = _DIR / "libfreq.so"

TOTAL = 65536


def _load_lib() -> ctypes.CDLL | None:
    try:
        if not _LIB.exists() or _SRC.stat().st_mtime > _LIB.stat().st_mtime:
            subprocess.check_call(
                ["gcc", "-O3", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)]
            )
        lib = ctypes.CDLL(str(_LIB))
        lib.freqs_from_probs.argtypes = [
            np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS"),
        ]
        lib.freqs_from_probs.restype = None
        return lib
    except Exception:
        return None


_lib = _load_lib()
HAVE_C = _lib is not None


def _quantize_numpy(flat: np.ndarray) -> np.ndarray:
    """Pure-numpy largest-remainder quantization (mirrors freq.c exactly)."""
    p = np.where(flat >= 0, flat, np.float32(0))  # NaN/neg -> 0, like C
    n_rows, n_sym = p.shape
    target = TOTAL - n_sym
    ftarget = np.float32(target)
    scaled = p * ftarget
    np.minimum(scaled, ftarget, out=scaled)
    fl = np.floor(scaled)
    base = fl.astype(np.int64) + 1  # min freq 1
    frac = scaled - fl  # float32, in [0,1)
    rem = TOTAL - base.sum(axis=1)  # (n_rows,) int64
    order = np.argsort(frac, axis=1, kind="stable")  # ties -> lower index first
    out = base
    for i in range(n_rows):
        r = int(rem[i])
        if r > 0:
            cycles, r = divmod(r, n_sym)
            if cycles:
                out[i] += cycles
            if r:
                out[i, order[i, n_sym - r :]] += 1  # largest fracs last
        elif r < 0:
            need = -r
            while need > 0:
                moved = False
                for k in range(n_sym):
                    if need == 0:
                        break
                    j = int(order[i, k])
                    if out[i, j] > 1:
                        out[i, j] -= 1
                        need -= 1
                        moved = True
                if not moved:
                    break
    return out.astype(np.uint32)


def freqs_from_probs(probs: np.ndarray) -> np.ndarray:
    """Quantize probabilities (..., n_sym) to uint32 freqs; last axis sums to 65536."""
    p = np.ascontiguousarray(probs, dtype=np.float32)
    if p.ndim == 0:
        raise ValueError("probs must have at least 1 dimension")
    n_sym = int(p.shape[-1])
    if n_sym == 0:
        return p.astype(np.uint32)
    if n_sym > TOTAL:
        raise ValueError(f"n_sym {n_sym} > {TOTAL}: cannot quantize")
    flat = p.reshape(-1, n_sym)
    if _lib is not None:
        out = np.empty(flat.shape, dtype=np.uint32)
        _lib.freqs_from_probs(flat, flat.shape[0], n_sym, out)
    else:
        out = _quantize_numpy(flat)
    return out.reshape(p.shape)


def freqs_from_logits(logits: np.ndarray) -> np.ndarray:
    """logits (..., n_sym) -> uint32 freqs (..., n_sym), last axis sums to 65536.

    float32 exp/softmax (vectorized, same as codec.py) then C quantization.
    """
    x = np.asarray(logits, dtype=np.float32)
    if x.ndim == 0:
        raise ValueError("logits must have at least 1 dimension")
    if x.shape[-1] == 0 or x.shape[-1] > TOTAL:
        raise ValueError(f"bad n_sym {x.shape[-1]} for quantization")
    x = x - x.max(axis=-1, keepdims=True)  # new array; input untouched
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return freqs_from_probs(x)


__all__ = ["TOTAL", "HAVE_C", "freqs_from_logits", "freqs_from_probs"]
