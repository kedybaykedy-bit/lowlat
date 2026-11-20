"""ctypes wrapper for the C codec core (ccodec.c).

Features: dirty-tile inter coding, motion estimation, tile-cache references,
text-protected adaptive quality (degrade only non-text tiles under a byte budget).
"""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "ccodec.c"
_LIB = _DIR / "libccodec.so"


def _ensure() -> ctypes.CDLL:
    if not _LIB.exists() or _SRC.stat().st_mtime > _LIB.stat().st_mtime:
        subprocess.check_call(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC), "-lz"]
        )
    return ctypes.CDLL(str(_LIB))


_lib = _ensure()
_lib.ccodec_new.restype = ctypes.c_void_p
_lib.ccodec_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.ccodec_free.argtypes = [ctypes.c_void_p]
_lib.ccodec_reset.argtypes = [ctypes.c_void_p]
_lib.ccodec_encode.restype = ctypes.c_void_p
_lib.ccodec_encode.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,   # keyframe
    ctypes.c_int,   # budget (<=0 unlimited)
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.ccodec_decode.restype = ctypes.c_void_p
_lib.ccodec_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_lib.ccodec_get_stats.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.ccodec_lossy_mask.restype = ctypes.POINTER(ctypes.c_uint8)
_lib.ccodec_lossy_mask.argtypes = [ctypes.c_void_p]
_libc = ctypes.CDLL(None)
_lib.ccodec_dbg_methods.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

_libc.free.argtypes = [ctypes.c_void_p]


class Stats:
    __slots__ = ("bytes", "dirty", "cache_hits", "lossy_tiles", "text_tiles", "has_lossy")

    def __init__(self):
        self.bytes = self.dirty = self.cache_hits = 0
        self.lossy_tiles = self.text_tiles = self.has_lossy = 0

    def __repr__(self):
        return (f"Stats(bytes={self.bytes}, dirty={self.dirty}, hits={self.cache_hits}, "
                f"lossy={self.lossy_tiles}, text={self.text_tiles}, has_lossy={bool(self.has_lossy)})")


class CCodec:
    def __init__(self, h: int, w: int, tile: int = 32, enable_me: bool = True,
                 cache_cap: int = 512):
        self.h, self.w = h, w
        self.tile, self.nt, self.nx = tile, (h + tile - 1) // tile, (w + tile - 1) // tile
        self._h = _lib.ccodec_new(h, w, tile, 1 if enable_me else 0, cache_cap)
        if not self._h:
            raise MemoryError("ccodec_new failed")
        self.last_stats = Stats()

    def close(self) -> None:
        if self._h:
            _lib.ccodec_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def reset(self) -> None:
        _lib.ccodec_reset(self._h)

    def encode(self, frame: np.ndarray, keyframe: bool = False,
               budget: int = -1) -> tuple[bytes, int]:
        """Encode a frame. budget>0 = max target bytes for this frame; only
        non-text tiles may be degraded to fit. budget<=0 = pure lossless."""
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        assert frame.shape == (self.h, self.w, 3), frame.shape
        outlen = ctypes.c_int(0)
        nd = ctypes.c_int(0)
        ptr = _lib.ccodec_encode(
            self._h, frame.ctypes.data, 1 if keyframe else 0, int(budget),
            ctypes.byref(outlen), ctypes.byref(nd),
        )
        if not ptr:
            raise RuntimeError("ccodec_encode failed")
        try:
            data = ctypes.string_at(ptr, outlen.value)
        finally:
            _libc.free(ptr)
        self._pull_stats()
        self.last_stats.dirty = nd.value
        return data, nd.value

    def decode(self, data: bytes) -> np.ndarray:
        buf = ctypes.create_string_buffer(data, len(data))
        outlen = ctypes.c_int(0)
        ptr = _lib.ccodec_decode(self._h, buf, len(data), ctypes.byref(outlen))
        if not ptr:
            raise RuntimeError("ccodec_decode failed")
        try:
            arr = np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(outlen.value,)
            ).copy()
        finally:
            _libc.free(ptr)
        self._pull_stats()
        return arr.reshape(self.h, self.w, 3)

    def _pull_stats(self) -> None:
        b = ctypes.c_int(0); d = ctypes.c_int(0); h = ctypes.c_int(0)
        lo = ctypes.c_int(0); te = ctypes.c_int(0); hl = ctypes.c_int(0)
        _lib.ccodec_get_stats(self._h, ctypes.byref(b), ctypes.byref(d), ctypes.byref(h),
                              ctypes.byref(lo), ctypes.byref(te), ctypes.byref(hl))
        s = self.last_stats
        s.bytes, s.dirty, s.cache_hits = b.value, d.value, h.value
        s.lossy_tiles, s.text_tiles, s.has_lossy = lo.value, te.value, hl.value

    def dbg_methods(self) -> list:
        """Counts of chosen tile methods [0..7] for the last encoded frame."""
        arr = (ctypes.c_int * 8)()
        _lib.ccodec_dbg_methods(self._h, arr)
        return list(arr)

    def lossy_mask(self) -> np.ndarray:
        """Per-tile lossy flag (nt,nx) for the last decoded/encoded frame."""
        ptr = _lib.ccodec_lossy_mask(self._h)
        arr = np.ctypeslib.as_array(ptr, shape=(self.nt * self.nx,))
        return arr.reshape(self.nt, self.nx).astype(bool)
