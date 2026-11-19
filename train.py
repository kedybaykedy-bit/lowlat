"""Train ResidualCDF — fast path: zero-MV + coarse ME sampling."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowlat.model import ResidualCDF
from lowlat.tiles import TILE, border_context, dirty_mask, predict_tile


class ResidualTiles(Dataset):
    def __init__(self, frames: np.ndarray, max_tiles: int = 8000, seed: int = 0):
        rng = np.random.default_rng(seed)
        h, w = frames.shape[1:3]
        nt, nx = (h + TILE - 1) // TILE, (w + TILE - 1) // TILE

        # Collect (frame_idx, ty, tx, mv_y, mv_x) samples with cheap ME
        samples = []
        recon = frames[0].copy()
        # frame 0: residual vs zeros (keyframe path)
        for ty in range(nt):
            for tx in range(nx):
                samples.append((0, ty, tx, 0, 0))
        recon = frames[0].copy()

        for fi in range(1, len(frames)):
            f = frames[fi]
            dm = dirty_mask(f, recon, TILE)
            for ty in range(nt):
                for tx in range(nx):
                    if not dm[ty, tx]:
                        continue
                    y0, x0 = ty * TILE, tx * TILE
                    y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                    hh, ww = y1 - y0, x1 - x0
                    cur = f[y0:y1, x0:x1]
                    # coarse ME only (step 4, range 8) — fast
                    best = (0, 0, 1 << 30)
                    for dy in range(-8, 9, 4):
                        for dx in range(-8, 9, 4):
                            yy, xx = y0 + dy, x0 + dx
                            if yy < 0 or xx < 0 or yy + hh > h or xx + ww > w:
                                continue
                            d = np.abs(
                                cur.astype(np.int16) - recon[yy : yy + hh, xx : xx + ww].astype(np.int16)
                            ).sum()
                            if d < best[2]:
                                best = (dy, dx, int(d))
                    samples.append((fi, ty, tx, best[0], best[1]))
            recon = f.copy()

        if len(samples) > max_tiles:
            pick = rng.choice(len(samples), size=max_tiles, replace=False)
            samples = [samples[i] for i in pick]

        # Build tensors with closed-loop recon walk (zero extra ME)
        self.items = []
        recon = np.zeros_like(frames[0])
        prev_res = np.zeros((h, w, 3), dtype=np.float32)
        # index samples by frame
        by_f: dict[int, list] = {}
        for s in samples:
            by_f.setdefault(s[0], []).append(s)

        for fi in range(len(frames)):
            f = frames[fi]
            new_recon = recon.copy()
            new_res = np.zeros_like(prev_res)
            flist = by_f.get(fi, [])
            # update recon fully each frame for causality of next frame samples
            for ty in range(nt):
                for tx in range(nx):
                    y0, x0 = ty * TILE, tx * TILE
                    y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                    hh, ww = y1 - y0, x1 - x0
                    # find mv for this tile if sampled
                    mv_y = mv_x = 0
                    for s in flist:
                        if s[1] == ty and s[2] == tx:
                            mv_y, mv_x = s[3], s[4]
                            break
                    if fi == 0:
                        pred = np.zeros((hh, ww, 3), dtype=np.uint8)
                        mv_y = mv_x = 0
                    else:
                        pred = predict_tile(recon, y0, x0, hh, ww, mv_y, mv_x)
                    resid = f[y0:y1, x0:x1].astype(np.int16) - pred.astype(np.int16)
                    new_res[y0:y1, x0:x1] = resid
                    # store sample if in flist or keyframe
                    wanted = fi == 0 or any(s[1] == ty and s[2] == tx for s in flist)
                    if wanted:
                        border = border_context(new_recon, y0, x0, hh, ww, mv_y, mv_x)
                        pr = prev_res[y0:y1, x0:x1]
                        me = np.abs(pred.astype(np.float32) - pr).mean(axis=2, keepdims=True) / 255.0
                        ctx = np.concatenate(
                            [
                                pred.astype(np.float32) / 255.0,
                                border,
                                np.clip(pr / 255.0, -1, 1),
                                me,
                            ],
                            axis=2,
                        ).astype(np.float32)
                        tgt = resid.astype(np.int32)
                        if hh != TILE or ww != TILE:
                            ctx_p = np.zeros((10, TILE, TILE), dtype=np.float32)
                            ctx_p[:, :hh, :ww] = ctx.transpose(2, 0, 1)
                            tgt_p = np.zeros((TILE, TILE, 3), dtype=np.int32)
                            tgt_p[:hh, :ww] = tgt
                            ctx, tgt = ctx_p, tgt_p
                        else:
                            ctx = np.ascontiguousarray(ctx.transpose(2, 0, 1))
                        self.items.append(
                            (np.ascontiguousarray(ctx), np.ascontiguousarray(tgt))
                        )
                    new_recon[y0:y1, x0:x1] = f[y0:y1, x0:x1]
            recon = new_recon
            prev_res = new_res

        print(f"dataset tiles: {len(self.items)}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ctx, tgt = self.items[i]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def main():
    from lowlat.content import generate_sequence

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    t0 = time.time()
    frames = generate_sequence(n_frames=24, h=720, w=1280, seed=1, motion="normal")
    frames_h = generate_sequence(n_frames=16, h=720, w=1280, seed=2, motion="heavy")
    frames = np.concatenate([frames, frames_h], axis=0)
    print(f"gen frames {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    ds = ResidualTiles(frames, max_tiles=5000)
    print(f"dataset {time.time()-t0:.1f}s", flush=True)
    c0, t0v = ds[0]
    print("ctx", tuple(c0.shape), "tgt", tuple(t0v.shape), flush=True)

    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    model = ResidualCDF(hidden=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    for epoch in range(20):
        model.train()
        tot = 0.0
        n = 0
        for ctx, tgt in dl:
            ctx, tgt = ctx.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            logits, loss = model(ctx, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * ctx.size(0)
            n += ctx.size(0)
        sched.step()
        avg = tot / max(n, 1)
        print(f"epoch {epoch} nll {avg:.4f} bits {avg/np.log(2):.4f}", flush=True)

    out = Path(__file__).resolve().parent / "models" / "rescdf.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
