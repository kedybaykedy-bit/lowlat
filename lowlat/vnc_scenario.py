"""Realistic VNC/desktop activity generator.

Models actual remote-desktop workloads:
  - large static regions (windows, wallpaper) → 0 bytes most frames
  - cursor movement (tiny sparse changes)
  - typing bursts (small text regions)
  - smooth scrolling (motion-compensatable)
  - occasional window interaction

This is the workload VNC is designed for, and where lossless 10KB/s is
information-theoretically achievable.
"""
from __future__ import annotations

import numpy as np
import cv2

from .content import make_desktop_bg, draw_window


def _stable_text(img, x, y, lines, base_seed, h, w):
    for i in range(lines):
        yy = y + i * 18
        if yy >= h - 4 or yy < 0:
            continue
        rs = np.random.default_rng(base_seed + i)
        xx = x
        while xx < x + 380 and xx < w - 4:
            tw = int(rs.integers(4, 26))
            col = int(rs.integers(20, 60))
            cv2.rectangle(img, (xx, yy), (xx + tw, yy + 10), (col, col, col), -1)
            xx += tw + int(rs.integers(3, 8))


def _draw_cursor(img, x, y):
    cv2.circle(img, (int(x), int(y)), 7, (0, 0, 0), -1)
    cv2.drawMarker(img, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)


def _base_desktop(h, w, seed):
    rng = np.random.default_rng(seed)
    f = make_desktop_bg(h, w, rng)
    # a couple of windows
    draw_window(f, int(w * 0.05), int(h * 0.08), int(w * 0.42), int(h * 0.55), "Editor", rng)
    _stable_text(f, int(w * 0.07), int(h * 0.16), min(30, int(h * 0.5) // 18), 1000, h, w)
    draw_window(f, int(w * 0.52), int(h * 0.10), int(w * 0.42), int(h * 0.5), "Terminal", rng)
    draw_window(f, int(w * 0.25), int(h * 0.68), int(w * 0.6), int(h * 0.28), "Browser", rng)
    return f


def generate_vnc(n_frames=90, h=480, w=854, seed=0, fps=30, activity="typical"):
    """activity:
       idle    — only cursor blinks/moves
       typical — cursor + occasional typing + rare scroll
       busy    — cursor + continuous typing + frequent scroll
    """
    rng = np.random.default_rng(seed)
    base = _base_desktop(h, w, seed)

    frames = np.empty((n_frames, h, w, 3), dtype=np.uint8)

    # Editor text state: number of typed lines, grows over time in bursts
    typed_lines = 6
    term_top_line = 0
    cursor = np.array([h * 0.5, w * 0.5], dtype=np.float32)
    browser_scroll = 0

    def render(cursor_xy, typed, term_off, bscroll, blink_on, with_cursor=True):
        f = base.copy()
        # editor typed text (stable lines)
        _stable_text(f, int(w * 0.07), int(h * 0.16), typed, 1000, h, w)
        # terminal visible lines (scrollable region)
        for i in range(12):
            line_id = term_off + i
            yy = int(h * 0.16) + i * 20
            if yy < h - 10:
                rs = np.random.default_rng(7000 + line_id)
                msg = f"$ {line_id:04d} {'x' * int(rs.integers(0, 6))}"
                cv2.putText(f, msg, (int(w * 0.54), yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 200, 80), 1, cv2.LINE_AA)
        # browser scrolled content
        _stable_text(f, int(w * 0.27), int(h * 0.72) - bscroll, 26, 3000, h, w)
        if with_cursor and blink_on:
            _draw_cursor(f, cursor_xy[1], cursor_xy[0])
        return f

    for t in range(n_frames):
        # cursor: real users move the mouse in short bursts, then it rests.
        move_cursor = False
        if activity == "idle":
            move_cursor = (t % 45 in (0, 1, 2))       # brief move every 1.5s
        elif activity == "typical":
            move_cursor = (t % 20 in (0, 1, 2, 3)) or (t % 37 == 5)
        else:  # busy
            move_cursor = (t % 6 in (0, 1, 2))
        if move_cursor:
            cursor[0] = float(np.clip(cursor[0] + rng.normal(0, 14), 20, h - 20))
            cursor[1] = float(np.clip(cursor[1] + rng.normal(0, 22), 20, w - 20))

        if activity == "idle":
            if t % 20 == 0:
                term_top_line += 1  # one line occasionally
        elif activity == "typical":
            # typing burst every ~40 frames: add a line
            if t > 0 and t % 40 == 0:
                typed_lines = min(typed_lines + 1, 30)
            # terminal scroll every ~30 frames by one line
            if t > 0 and t % 30 == 0:
                term_top_line += 1
            # browser scroll occasionally, small step
            if t > 0 and t % 75 == 0:
                browser_scroll += 12
        else:  # busy
            if t > 0 and t % 8 == 0:
                typed_lines = min(typed_lines + 1, 30)
            if t > 0 and t % 10 == 0:
                term_top_line += 1
            if t > 0 and t % 20 == 0:
                browser_scroll += 20

        blink = (t // 8) % 2 == 0
        frames[t] = render(cursor, typed_lines, term_top_line, browser_scroll, blink, with_cursor=True)

    return frames


def generate_vnc_layered(n_frames=90, h=480, w=854, seed=0, activity="typical"):
    """Returns (base_frames, cursors):
       base_frames[t] = desktop WITHOUT cursor (what the codec compresses)
       cursors[t] = None or (x, y) — tiny side channel (composited client-side)
       composite = draw_cursor(base_frames[t], x, y) reproduces the true frame bit-exactly.
    """
    rng = np.random.default_rng(seed)
    base = _base_desktop(h, w, seed)
    base_frames = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    cursors: list = []

    typed_lines = 6
    term_top_line = 0
    cursor = np.array([h * 0.5, w * 0.5], dtype=np.float32)
    browser_scroll = 0

    def render(typed, term_off, bscroll):
        f = base.copy()
        _stable_text(f, int(w * 0.07), int(h * 0.16), typed, 1000, h, w)
        for i in range(12):
            line_id = term_off + i
            yy = int(h * 0.16) + i * 20
            if yy < h - 10:
                rs = np.random.default_rng(7000 + line_id)
                msg = f"$ {line_id:04d} {'x' * int(rs.integers(0, 6))}"
                cv2.putText(f, msg, (int(w * 0.54), yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 200, 80), 1, cv2.LINE_AA)
        _stable_text(f, int(w * 0.27), int(h * 0.72) - bscroll, 26, 3000, h, w)
        return f

    for t in range(n_frames):
        move_cursor = False
        if activity == "idle":
            move_cursor = (t % 45 in (0, 1, 2))
        elif activity == "typical":
            move_cursor = (t % 20 in (0, 1, 2, 3)) or (t % 37 == 5)
        else:
            move_cursor = (t % 6 in (0, 1, 2))
        if move_cursor:
            cursor[0] = float(np.clip(cursor[0] + rng.normal(0, 14), 20, h - 20))
            cursor[1] = float(np.clip(cursor[1] + rng.normal(0, 22), 20, w - 20))

        if activity == "idle":
            if t % 20 == 0:
                term_top_line += 1
        elif activity == "typical":
            if t > 0 and t % 40 == 0:
                typed_lines = min(typed_lines + 1, 30)
            if t > 0 and t % 30 == 0:
                term_top_line += 1
            if t > 0 and t % 75 == 0:
                browser_scroll += 12
        else:
            if t > 0 and t % 8 == 0:
                typed_lines = min(typed_lines + 1, 30)
            if t > 0 and t % 10 == 0:
                term_top_line += 1
            if t > 0 and t % 20 == 0:
                browser_scroll += 20

        blink = (t // 8) % 2 == 0
        base_frames[t] = render(typed_lines, term_top_line, browser_scroll)
        cursors.append((int(cursor[1]), int(cursor[0])) if blink else None)

    return base_frames, cursors


def composite_cursor(frame, xy):
    """Draw the cursor sprite on a copy — deterministic, matches _draw_cursor."""
    if xy is None:
        return frame
    f = frame.copy()
    _draw_cursor(f, xy[0], xy[1])
    return f



# ====================== ARTICLE (smooth scrolling text) ======================
def _article_page(h, w, seed):
    """A dense text document on white paper (mostly text tiles)."""
    img = np.full((h, w, 3), 250, np.uint8)
    # margins / page block
    cv2.rectangle(img, (w // 8, 0), (w * 7 // 8, h), (252, 250, 245), -1)
    cv2.putText(img, "A Study of Lossless Remote Display", (w // 8 + 40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 30), 3, cv2.LINE_AA)
    y = 130
    ln = 0
    rng = np.random.default_rng(seed)
    # headings + paragraph "words"
    while y < h - 20:
        if ln % 18 == 0:
            cv2.putText(img, f"Section {ln//18 + 1}. Subheading text", (w // 8 + 40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 60), 2, cv2.LINE_AA)
            y += 30
        else:
            x = w // 8 + 40
            xmax = w * 7 // 8 - 40
            while x < xmax:
                wl = int(rng.integers(18, 70))
                if x + wl > xmax:
                    break
                cv2.putText(img, "Lorem", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (25, 25, 30), 2, cv2.LINE_AA)
                x += wl
            y += 26
        ln += 1
    return img


def generate_article(n_frames=90, h=1080, w=1920, seed=0, scroll_speed=4.0):
    """Scroll a tall article smoothly (sub-tile motion → ME handles it).
    Returns (frames, cursors) in the same shape as generate_vnc_layered.
    """
    tall = _article_page(h * 4, w, seed)
    frames = np.empty((n_frames, h, w, 3), np.uint8)
    cursors = []
    y = 0.0
    for t in range(n_frames):
        y = min(max(y + scroll_speed, 0), tall.shape[0] - h)
        yi = int(round(y))
        frames[t] = tall[yi:yi + h, :]
        cursors.append(None)
    return frames, cursors


# ====================== STRESS (video region + static text UI) ======================
def generate_stress(n_frames=90, h=1080, w=1920, seed=0, video_frac=0.30):
    """Text UI that must stay lossless + an animated high-entropy 'video' region
    (non-text) that deliberately exceeds a 10KB/s budget, forcing adaptive loss
    ONLY on the video tiles while text stays pixel-exact and smooth.
    """
    base = _base_desktop(h, w, seed)
    # a dense text panel
    _stable_text(base, int(w * 0.05), int(h * 0.12), min(34, int(h * 0.7) // 18), 4400, h, w)
    vh = int(h * np.sqrt(video_frac))
    vw = int(w * np.sqrt(video_frac))
    vy, vx = int(h * 0.30), int(w * 0.35)
    frames = np.empty((n_frames, h, w, 3), np.uint8)
    cursors = []
    rng = np.random.default_rng(seed)
    # smooth animated "video": low-res noise upscaled + shifting gradient
    yy, xx = np.mgrid[0:vh, 0:vw].astype(np.float32)
    for t in range(n_frames):
        f = base.copy()
        # cheap animated plasma (non-text, textured) — large & incompressible-ish
        ph = t * 0.25
        plasma = (
            128 + 60 * np.sin(xx / 23.0 + ph)
            + 60 * np.cos(yy / 31.0 - ph * 1.3)
            + 30 * np.sin((xx + yy) / 17.0 + ph * 0.7)
        )
        img = np.empty((vh, vw, 3), np.uint8)
        img[..., 0] = np.clip(plasma, 0, 255)
        img[..., 1] = np.clip(255 - plasma, 0, 255)
        img[..., 2] = np.clip(128 + 100 * np.sin(xx / 41.0 + ph), 0, 255)
        f[vy:vy + vh, vx:vx + vw] = img
        frames[t] = f
        cursors.append(None)
    return frames, cursors
