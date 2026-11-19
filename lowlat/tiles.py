"""Tiling, dirty detection, motion search, prediction."""
from __future__ import annotations

import numpy as np

TILE = 32  # 32x32 blocks — good for screen content


def frame_tiles(frame: np.ndarray, tile: int = TILE):
    """Yield (ty, tx, y0, y1, x0, x1) tile grid for HxWx3 uint8."""
    h, w = frame.shape[:2]
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            yield y0 // tile, x0 // tile, y0, min(y0 + tile, h), x0, min(x0 + tile, w)


def sad(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).sum())


def dirty_mask(
    cur: np.ndarray,
    ref: np.ndarray,
    tile: int = TILE,
    thresh: int = 0,
) -> np.ndarray:
    """Boolean grid (nt, nx) True if tile differs from ref. Fully vectorized."""
    h, w = cur.shape[:2]
    nt, nx = (h + tile - 1) // tile, (w + tile - 1) // tile
    # pad to multiple of tile
    d = np.abs(cur.astype(np.int16) - ref.astype(np.int16)).sum(axis=2)  # (h,w)
    ph, pw = nt * tile, nx * tile
    if ph != h or pw != w:
        dp = np.zeros((ph, pw), dtype=d.dtype)
        dp[:h, :w] = d
        d = dp
    blocks = d.reshape(nt, tile, nx, tile).max(axis=(1, 3))  # max per tile
    if thresh == 0:
        return blocks > 0
    return blocks > thresh


def motion_search(
    cur_tile: np.ndarray,
    ref_full: np.ndarray,
    y0: int,
    x0: int,
    search: int = 8,
    step: int = 4,
) -> tuple[int, int, int]:
    """Coarse diamond ME. Returns (mv_y, mv_x, sad_best)."""
    h, w = cur_tile.shape[:2]
    H, W = ref_full.shape[:2]
    cur_i = cur_tile.astype(np.int16)
    best = (0, 0, sad(cur_tile, ref_full[y0 : y0 + h, x0 : x0 + w]))
    # coarse grid
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            yy, xx = y0 + dy, x0 + dx
            if yy < 0 or xx < 0 or yy + h > H or xx + w > W:
                continue
            s = int(
                np.abs(cur_i - ref_full[yy : yy + h, xx : xx + w].astype(np.int16)).sum()
            )
            if s < best[2]:
                best = (dy, dx, s)
    # refine ±step
    by, bx, bs = best
    r = max(1, step // 2)
    for dy in range(by - r, by + r + 1):
        for dx in range(bx - r, bx + r + 1):
            yy, xx = y0 + dy, x0 + dx
            if yy < 0 or xx < 0 or yy + h > H or xx + w > W:
                continue
            s = int(
                np.abs(cur_i - ref_full[yy : yy + h, xx : xx + w].astype(np.int16)).sum()
            )
            if s < bs:
                bs, by, bx = s, dy, dx
    return by, bx, bs


def predict_tile(
    ref: np.ndarray,
    y0: int,
    x0: int,
    h: int,
    w: int,
    mv_y: int = 0,
    mv_x: int = 0,
) -> np.ndarray:
    yy, xx = y0 + mv_y, x0 + mv_x
    return ref[yy : yy + h, xx : xx + w].copy()


def border_context(
    recon: np.ndarray,
    y0: int,
    x0: int,
    h: int,
    w: int,
    mv_y: int = 0,
    mv_x: int = 0,
) -> np.ndarray:
    """(h,w,3) float0..1 — left/top edge fill, fully vectorized."""
    left_col = recon[y0 : y0 + h, max(x0 - 1, 0)].astype(np.float32)  # (h,3)
    top_row = recon[max(y0 - 1, 0), x0 : x0 + w].astype(np.float32)  # (w,3)
    # horizontal fill from left, then blend top row
    out = np.broadcast_to(left_col[:, None, :], (h, w, 3)).copy()
    # vertical contribution from top: blend factor ramps with x? keep simple avg on row0+
    # out[y,x] = 0.5*left[y] + 0.5*top[x] for all (matches prior per-pixel blend intent)
    out = 0.5 * out + 0.5 * top_row[None, :, :]
    return out / 255.0
