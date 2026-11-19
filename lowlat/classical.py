"""Classical lossless tile coders: PNG, palette+RLE, Paeth predictor + zlib.

Bit-exact baselines for keyframe/intra tiles (no neural model involved).

Bitstreams:
  png_encode         -> PNG file bytes (cv2.imencode)
  palette_rle_encode -> [u8 mode][u16 n_colors][n_colors*3 palette][packbits indices]
                         mode 1 = palette, mode 2 = PNG fallback (>256 colors)
  encode_intra_tile  -> [u8 method][payload]
                         method 0 = PNG, 1 = palette_rle, 2 = Paeth+zlib
"""
from __future__ import annotations

import struct
import zlib

import cv2
import numpy as np

# encode_intra_tile method ids (first byte of bitstream)
METHOD_PNG = 0
METHOD_PALETTE = 1
METHOD_PAETH_ZLIB = 2
METHOD_NAMES = {METHOD_PNG: "png", METHOD_PALETTE: "palette_rle", METHOD_PAETH_ZLIB: "paeth_zlib"}

# palette_rle internal mode flags (first byte of its bitstream)
PAL_MODE_PALETTE = 1
PAL_MODE_PNG = 2  # >256 unique colors: rest of stream is raw PNG bytes


# ---------------------------------------------------------------- PNG
def png_encode(arr: np.ndarray) -> bytes:
    """Lossless PNG bytes for HxWx3 uint8."""
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {arr.dtype} {arr.shape}")
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr))
    if not ok:
        raise ValueError("cv2.imencode('.png') failed")
    return buf.tobytes()


def png_decode(data: bytes, shape: tuple) -> np.ndarray:
    """Inverse of png_decode; `shape` is (h, w, 3)."""
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cv2.imdecode failed")
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(f"decoded shape {arr.shape} != expected {shape}")
    return arr


# ---------------------------------------------------------------- packbits
def _packbits_encode(data: np.ndarray) -> bytes:
    """PackBits over 1-D uint8 (literal runs 1..128, repeat runs 2..128)."""
    out = bytearray()
    n = int(data.shape[0])
    i = 0
    while i < n:
        j = i
        while j < n and data[j] == data[i] and (j - i) < 128:
            j += 1
        run = j - i
        if run >= 3:
            out.append((1 - run) & 0xFF)
            out.append(int(data[i]))
            i = j
        else:
            start = i
            i += max(run, 1)
            while i < n and (i - start) < 128:
                j = i
                while j < n and data[j] == data[i] and (j - i) < 128:
                    j += 1
                r = j - i
                if r >= 3:
                    break
                take = min(r, 128 - (i - start))
                i += take
                if take < r:
                    break
            out.append(i - start - 1)
            out += data[start:i].tobytes()
    return bytes(out)


def _packbits_decode(buf: bytes, count: int) -> np.ndarray:
    out = bytearray()
    i = 0
    n = len(buf)
    while len(out) < count and i < n:
        h = buf[i]
        i += 1
        hs = h - 256 if h >= 128 else h
        if hs >= 0:
            k = hs + 1
            out += buf[i : i + k]
            i += k
        elif hs >= -127:
            k = 1 - hs
            if i >= n:
                raise ValueError("packbits: truncated repeat run")
            out += bytes([buf[i]]) * k
            i += 1
        # hs == -128: no-op
    if len(out) < count:
        raise ValueError(f"packbits: got {len(out)} bytes, need {count}")
    return np.frombuffer(bytes(out[:count]), dtype=np.uint8)


# ---------------------------------------------------------------- palette + RLE
def palette_rle_encode(tile: np.ndarray) -> bytes:
    """[u8 mode][u16 n_colors][palette][packbits(indices)] lossless.

    If the tile has >256 unique colors, falls back to mode 2 = PNG bytes.
    """
    if tile.dtype != np.uint8 or tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {tile.dtype} {tile.shape}")
    h, w, _ = tile.shape
    flat = np.ascontiguousarray(tile).reshape(-1, 3)
    keys = (
        (flat[:, 0].astype(np.int32) << 16)
        | (flat[:, 1].astype(np.int32) << 8)
        | flat[:, 2].astype(np.int32)
    )
    uniq, inv = np.unique(keys, return_inverse=True)
    n = int(uniq.shape[0])
    if n > 256:
        return bytes([PAL_MODE_PNG]) + png_encode(tile)
    palette = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(
        np.uint8
    )
    indices = inv.astype(np.uint8)
    out = bytearray()
    out.append(PAL_MODE_PALETTE)
    out += struct.pack("<H", n)
    out += palette.tobytes()
    out += _packbits_encode(indices)
    return bytes(out)


def palette_rle_decode(data: bytes, h: int, w: int) -> np.ndarray:
    """Inverse of palette_rle_encode; returns (h, w, 3) uint8."""
    if not data:
        raise ValueError("palette_rle: empty data")
    mode = data[0]
    if mode == PAL_MODE_PNG:
        return png_decode(data[1:], (h, w, 3))
    if mode != PAL_MODE_PALETTE:
        raise ValueError(f"palette_rle: unknown mode {mode}")
    if len(data) < 3:
        raise ValueError("palette_rle: truncated header")
    (n,) = struct.unpack_from("<H", data, 1)
    if n == 0 or len(data) < 3 + n * 3:
        raise ValueError("palette_rle: truncated palette")
    palette = np.frombuffer(data, dtype=np.uint8, count=n * 3, offset=3).reshape(n, 3)
    idx = _packbits_decode(data[3 + n * 3 :], h * w)
    return palette[idx].reshape(h, w, 3)


# ---------------------------------------------------------------- predictors
def paeth_predictor(a, b, c):
    """PNG Paeth predictor (elementwise; scalars or arrays). p=a+b-c."""
    ai = np.asarray(a, dtype=np.int32)
    bi = np.asarray(b, dtype=np.int32)
    ci = np.asarray(c, dtype=np.int32)
    p = ai + bi - ci
    pa = np.abs(p - ai)
    pb = np.abs(p - bi)
    pc = np.abs(p - ci)
    out = np.where((pa <= pb) & (pa <= pc), ai, np.where(pb <= pc, bi, ci))
    if np.isscalar(a) and np.isscalar(b) and np.isscalar(c):
        return int(out)
    return out


def pred_copy(ref_tile: np.ndarray, cur: np.ndarray | None = None) -> np.ndarray:
    """Identity predictor: predict ref_tile itself.

    With `cur` given returns the residual cur - ref; otherwise the prediction.
    """
    if cur is None:
        return ref_tile.copy()
    return cur.astype(np.int16) - ref_tile.astype(np.int16)


def pred_left(cur: np.ndarray) -> np.ndarray:
    """Left-neighbor prediction inside the tile (first column predicted as 0)."""
    pred = np.zeros_like(cur)
    if cur.shape[1] > 1:
        pred[:, 1:] = cur[:, :-1]
    return pred


def intra_residual(tile: np.ndarray) -> np.ndarray:
    """Paeth (left+top) intra residual for a keyframe tile -> int16 (h,w,3).

    Borders are treated as 0. Signed range [-255, 255].
    """
    t = tile.astype(np.int32)
    left = np.zeros_like(t)
    left[:, 1:] = t[:, :-1]
    top = np.zeros_like(t)
    top[1:, :] = t[:-1, :]
    tl = np.zeros_like(t)
    tl[1:, 1:] = t[:-1, :-1]
    pred = paeth_predictor(left, top, tl)
    return (t - pred).astype(np.int16)


def intra_recon(residual: np.ndarray) -> np.ndarray:
    """Inverse of intra_residual for uint8-wrapped (mod 256) residuals.

    Sequential Paeth reconstruction, anti-diagonal vectorized.
    """
    h, w, c = residual.shape
    r = residual.astype(np.int32)
    out = np.zeros((h, w, c), dtype=np.int32)
    for k in range(h + w - 1):
        y0 = max(0, k - (w - 1))
        y1 = min(h - 1, k)
        ys = np.arange(y0, y1 + 1)
        xs = k - ys
        xm = np.maximum(xs - 1, 0)
        ym = np.maximum(ys - 1, 0)
        left_ok = (xs > 0)[:, None]
        top_ok = (ys > 0)[:, None]
        tl_ok = left_ok & top_ok
        a = np.where(left_ok, out[ys, xm], 0)
        b = np.where(top_ok, out[ym, xs], 0)
        cc = np.where(tl_ok, out[ym, xm], 0)
        pred = paeth_predictor(a, b, cc)
        out[ys, xs] = (pred + r[ys, xs]) & 0xFF
    return out.astype(np.uint8)


# ---------------------------------------------------------------- intra tile
def encode_intra_tile(tile: np.ndarray) -> bytes:
    """Try [png, palette_rle, paeth+zlib], prefix the smallest with method id."""
    png = png_encode(tile)
    pal = palette_rle_encode(tile)
    wrapped = np.mod(intra_residual(tile), 256).astype(np.uint8)
    paeth = zlib.compress(wrapped.tobytes(), 6)
    method, payload = min(
        ((METHOD_PNG, png), (METHOD_PALETTE, pal), (METHOD_PAETH_ZLIB, paeth)),
        key=lambda cp: len(cp[1]),
    )
    return bytes([method]) + payload


def decode_intra_tile(data: bytes, h: int, w: int) -> np.ndarray:
    """Inverse of encode_intra_tile; returns (h, w, 3) uint8."""
    if not data:
        raise ValueError("decode_intra_tile: empty data")
    method = data[0]
    payload = data[1:]
    if method == METHOD_PNG:
        return png_decode(payload, (h, w, 3))
    if method == METHOD_PALETTE:
        return palette_rle_decode(payload, h, w)
    if method == METHOD_PAETH_ZLIB:
        raw = zlib.decompress(payload)
        need = h * w * 3
        if len(raw) != need:
            raise ValueError(f"paeth_zlib: got {len(raw)} bytes, need {need}")
        resid = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        return intra_recon(resid)
    raise ValueError(f"decode_intra_tile: unknown method {method}")


# ---------------------------------------------------------------- tests
def _tile_desktop(h: int = 64, w: int = 64) -> np.ndarray:
    """Flat-color window + text edges (screen-content-like)."""
    t = np.full((h, w, 3), 32, np.uint8)  # flat desktop
    cv2.rectangle(t, (2, 2), (w - 3, h - 3), (240, 240, 240), -1)  # window body
    cv2.rectangle(t, (2, 2), (w - 3, 18), (60, 100, 170), -1)  # title bar
    cv2.putText(
        t, "editor.txt", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_8
    )
    cv2.rectangle(t, (2, 19), (w - 3, 30), (200, 200, 200), -1)  # menu strip
    for i, s in enumerate(("File", "Edit", "View")):
        cv2.putText(
            t, s, (6 + i * 22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_8
        )
    for row in range(4):  # pseudo text lines
        y = 36 + row * 6
        x = 6
        rs = np.random.default_rng(100 + row)
        while x < w - 8:
            tw = int(rs.integers(2, 9))
            c = int(rs.integers(15, 70))
            cv2.rectangle(t, (x, y), (x + tw, y + 3), (c, c, c), -1)
            x += tw + int(rs.integers(2, 5))
    for i, col in enumerate(((220, 70, 70), (230, 190, 60), (70, 200, 90))):  # traffic lights
        cv2.circle(t, (w - 30 + i * 9, 10), 3, col, -1)
    return t


def _tile_noise(h: int = 64, w: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _tile_gradient(h: int = 64, w: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    g = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    b = np.full((h, w), 128, np.uint8)
    return np.stack([r, g, b], axis=2)


def _unit_checks(failures: list[str]) -> None:
    print("== unit checks ==")

    def check(name: str, cond: bool) -> None:
        print(f"  {name:<28} {'OK' if cond else 'FAIL'}")
        if not cond:
            failures.append(name)

    # paeth scalar (PNG spec behavior: ties prefer a, then b)
    check("paeth_predictor scalar", paeth_predictor(100, 100, 100) == 100)
    check("paeth_predictor picks b", paeth_predictor(10, 20, 10) == 20)
    check("paeth_predictor picks a", paeth_predictor(50, 200, 200) == 50)

    # packbits roundtrip: runs + literals + long runs
    rng = np.random.default_rng(7)
    cases = [
        np.zeros(500, np.uint8),
        (np.arange(300) % 256).astype(np.uint8),
        np.array([1, 1, 1, 2, 3, 3, 3, 3, 3], np.uint8),
        np.concatenate(
            [np.full(300, 9, np.uint8), rng.integers(0, 256, 17, dtype=np.uint8)]
        ),
        np.full(1, 42, np.uint8),
    ]
    ok = True
    for c in cases:
        dec = _packbits_decode(_packbits_encode(c), c.shape[0])
        ok &= bool(np.array_equal(dec, c))
    check("packbits roundtrip", ok)

    # pred_copy / pred_left
    ref = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    cur = (ref.astype(np.int16) + 5).astype(np.uint8)
    resid = pred_copy(ref, cur)
    check("pred_copy residual", bool(np.all(resid == 5)))
    check("pred_copy identity", bool(np.array_equal(pred_copy(ref), ref)))
    pl = pred_left(cur)
    check(
        "pred_left neighbors",
        bool(np.all(pl[:, 0] == 0) and np.all(pl[:, 1:] == cur[:, :-1])),
    )

    # intra residual -> recon bit-exact (incl. wrapping)
    tile = _tile_desktop(33, 29)  # odd size exercises edge rows/cols
    wrapped = np.mod(intra_residual(tile), 256).astype(np.uint8)
    check("intra_residual/recon", bool(np.array_equal(intra_recon(wrapped), tile)))


def _eval_content(label: str, tile: np.ndarray, failures: list[str]) -> str:
    h, w = tile.shape[:2]
    ncol = int(np.unique(tile.reshape(-1, 3), axis=0).shape[0])
    print(f"\n== content: {label} ({h}x{w}, {ncol} unique colors) ==")
    print(f"  {'coder':<16}{'bytes':>8}  roundtrip")

    sizes: dict[str, int] = {}
    oks: dict[str, bool] = {}

    png = png_encode(tile)
    sizes["png"] = len(png)
    oks["png"] = bool(np.array_equal(png_decode(png, tile.shape), tile))

    pal = palette_rle_encode(tile)
    sizes["palette_rle"] = len(pal)
    oks["palette_rle"] = bool(np.array_equal(palette_rle_decode(pal, h, w), tile))

    wrapped = np.mod(intra_residual(tile), 256).astype(np.uint8)
    paeth = zlib.compress(wrapped.tobytes(), 6)
    sizes["paeth_zlib"] = len(paeth)
    paeth_res = np.frombuffer(zlib.decompress(paeth), np.uint8).reshape(h, w, 3)
    oks["paeth_zlib"] = bool(np.array_equal(intra_recon(paeth_res), tile))

    for name in ("png", "palette_rle", "paeth_zlib"):
        print(f"  {name:<16}{sizes[name]:>8}  {'OK' if oks[name] else 'FAIL'}")
        if not oks[name]:
            failures.append(f"{label}/{name}")

    blob = encode_intra_tile(tile)
    rec = decode_intra_tile(blob, h, w)
    ok = bool(np.array_equal(rec, tile))
    method = blob[0]
    print(
        f"  {'encode_intra':<16}{len(blob):>8}  {'OK' if ok else 'FAIL'}"
        f"  (method={method} {METHOD_NAMES.get(method, '?')})"
    )
    if not ok:
        failures.append(f"{label}/encode_intra")

    expect = 1 + min(sizes.values())
    if len(blob) != expect:
        failures.append(f"{label}/intra_size")
        print(f"  !! encode_intra size {len(blob)} != 1+min {expect}")

    pal_mode = pal[0]
    print(
        f"  palette mode: {'palette' if pal_mode == PAL_MODE_PALETTE else 'png-fallback'}"
        f", raw RGB = {h * w * 3} B"
    )

    winner = min(sizes, key=sizes.get)
    print(f"  winner: {winner} ({sizes[winner]} B)")
    return winner


def main() -> int:
    failures: list[str] = []
    _unit_checks(failures)

    winners = {
        "desktop_flat_text": _eval_content("desktop_flat_text", _tile_desktop(), failures),
        "noise": _eval_content("noise", _tile_noise(), failures),
        "gradient": _eval_content("gradient", _tile_gradient(), failures),
    }

    print("\n== winners per content type ==")
    for k, v in winners.items():
        print(f"  {k:<20} {v}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
