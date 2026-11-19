"""Synthetic 2K screen-like content generator (desktop, text, cursor, scroll)."""
from __future__ import annotations

import numpy as np
import cv2


def make_desktop_bg(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth gradient wallpaper (no per-pixel dither — like a real screenshot)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 40 + 40 * (xx / w) + 10 * np.sin(yy / 80.0)
    g = 50 + 50 * (yy / h) + 10 * np.cos(xx / 90.0)
    b = 70 + 30 * ((xx + yy) / (w + h))
    img = np.stack([r, g, b], axis=2)
    # smooth low-frequency texture (compressible), not white noise
    tex = 2.0 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
    img += tex[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_window(img: np.ndarray, x: int, y: int, ww: int, wh: int, title: str, rng) -> None:
    # frame
    cv2.rectangle(img, (x, y), (x + ww, y + wh), (220, 220, 220), -1)
    cv2.rectangle(img, (x, y), (x + ww, y + wh), (30, 30, 30), 2)
    # title bar
    cv2.rectangle(img, (x + 2, y + 2), (x + ww - 2, y + 30), (50, 90, 160), -1)
    cv2.putText(
        img, title, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    # fake controls
    for i, col in enumerate([(200, 80, 80), (200, 180, 60), (80, 180, 90)]):
        cx = x + ww - 20 - i * 22
        cv2.circle(img, (cx, y + 16), 6, col, -1)


def draw_text_lines(img: np.ndarray, x: int, y: int, n: int, rng) -> None:
    for i in range(n):
        yy = y + i * 18
        if yy >= img.shape[0] - 4:
            break
        # pseudo text: random short dashes
        xx = x
        while xx < x + 400:
            tw = int(rng.integers(4, 28))
            col = int(rng.integers(20, 60))
            cv2.rectangle(img, (xx, yy), (xx + tw, yy + 10), (col, col, col), -1)
            xx += tw + int(rng.integers(3, 8))


def generate_sequence(
    n_frames: int = 90,
    h: int = 1440,
    w: int = 2560,
    fps: int = 30,
    seed: int = 0,
    motion: str = "normal",
) -> np.ndarray:
    """Return (N,H,W,3) uint8 sequence — screen-content-like.

    motion: normal | heavy | static
    """
    rng = np.random.default_rng(seed)
    base = make_desktop_bg(h, w, rng)

    # static windows on base
    frame0 = base.copy()
    draw_window(frame0, 80, 60, 900, 620, "Editor", rng)
    draw_text_lines(frame0, 100, 110, 28, rng)
    draw_window(frame0, 1100, 100, 700, 500, "Terminal", rng)
    for i in range(12):
        col = (30, 200, 80) if i % 3 else (200, 200, 200)
        cv2.putText(
            frame0,
            f"$ cmd --flag {i} output_{i}",
            (1120, 150 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            1,
            cv2.LINE_AA,
        )
    draw_window(frame0, 600, 750, 1200, 600, "Browser", rng)
    draw_text_lines(frame0, 620, 800, 24, rng)

    frames = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    frames[0] = frame0

    # content region for "video" inside browser (worst case entropy)
    vid_h, vid_w = 360, 640
    vid_y, vid_x = 900, 700

    cursor = np.array([700, 400])
    scroll_off = 0

    for t in range(1, n_frames):
        if motion == "static" and t > 1:
            # tiny cursor blink only
            f = frames[t - 1].copy()
            if t % 15 == 0:
                f = frames[0].copy()
                cv2.circle(f, (int(cursor[1]), int(cursor[0])), 8, (0, 0, 0), -1)
                cv2.circle(f, (int(cursor[1]), int(cursor[0])), 7, (255, 255, 255), -1)
            frames[t] = f
            continue

        f = frame0.copy()  # restart from clean desktop each time (windows persistent)

        # typing: stable base text + only append new chars occasionally (VNC-like)
        ntyped = min(20, 8 + (t // 3))
        # reuse same rng stream but only redraw the NEW line each frame
        for i in range(ntyped):
            yy = 110 + i * 18
            if yy >= h - 4:
                break
            xx = 100
            # deterministic pseudo-text from index (stable across frames)
            rs = np.random.default_rng(1000 + i)
            while xx < 500:
                tw = int(rs.integers(4, 28))
                col = int(rs.integers(20, 60))
                cv2.rectangle(f, (xx, yy), (xx + tw, yy + 10), (col, col, col), -1)
                xx += tw + int(rs.integers(3, 8))

        # terminal: only shift one line when scroll ticks
        scroll_step = t // 4
        for i in range(12):
            line_id = i + scroll_step
            yy = 150 + i * 28
            if 110 < yy < 580:
                rs = np.random.default_rng(5000 + line_id)
                msg = f"$ log {line_id:04d} {'x' * int(rs.integers(0, 7))}"
                cv2.putText(
                    f, msg, (1120, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 200, 80), 1, cv2.LINE_AA,
                )

        # cursor motion
        if motion == "heavy":
            cursor[0] = int(cursor[0] + rng.normal(0, 40)) % (h - 20) + 10
            cursor[1] = int(cursor[1] + rng.normal(0, 60)) % (w - 20) + 10
        else:
            cursor[0] = int(np.clip(cursor[0] + rng.normal(0, 8), 20, h - 20))
            cursor[1] = int(np.clip(cursor[1] + rng.normal(0, 12), 20, w - 20))
        cv2.circle(f, (int(cursor[1]), int(cursor[0])), 8, (0, 0, 0), -1)
        cv2.drawMarker(
            f, (int(cursor[1]), int(cursor[0])), (255, 255, 255), cv2.MARKER_CROSS, 16, 2
        )

        # window drag (browser moves)
        if motion == "heavy" and t % 30 < 15:
            dx = 4 * ((t % 30) - 7)
            draw_window(f, 600 + dx, 750, 1200, 600, "Browser", rng)
            draw_text_lines(f, 620 + dx, 800, 24, rng)

        # high-entropy region: animated "video" patch (only if motion heavy)
        if motion == "heavy" and vid_y + vid_h <= h and vid_x + vid_w <= w:
            noise = rng.integers(0, 255, (vid_h, vid_w, 3), dtype=np.uint8)
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
            f[vid_y : vid_y + vid_h, vid_x : vid_x + vid_w] = noise
        elif motion == "heavy":
            vh, vw = min(vid_h, h // 3), min(vid_w, w // 3)
            vy, vx = h // 2, w // 4
            noise = rng.integers(0, 255, (vh, vw, 3), dtype=np.uint8)
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
            f[vy : vy + vh, vx : vx + vw] = noise

        # occasional full redraw (palette shift) — rare
        if t % 45 == 0 and motion != "static":
            f = np.roll(f, 1, axis=1)

        frames[t] = f

    return frames
