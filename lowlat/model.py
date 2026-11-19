"""Neural residual CDF model (EnCodec-style discrete latents → but lossless via ANS)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# residual alphabet: -255..255 → 511 bins
RES_MIN = -255
N_BINS = 511


def residual_to_bin(r: torch.Tensor) -> torch.Tensor:
    return (r - RES_MIN).clamp(0, N_BINS - 1).long()


def bin_to_residual(b: torch.Tensor) -> torch.Tensor:
    return b + RES_MIN


class ResidualCDF(nn.Module):
    """Predict categorical distribution over residual bins (per RGB channel).

    Context channels (all available to both encoder & decoder):
      0-2: predicted tile (0..1)
      3-5: neighbor mean of *already reconstructed* left/top borders (replicated)
      6-8: previous-frame residual (same tile) — causal, from prior coded frame
      9:   motion energy proxy (|pred - prev_rec| mean)
    Output logits: (B, 3*N_BINS, H, W) — R,G,B independent heads.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        c_in = 10
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden, 3 * N_BINS, 1),
        )
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, ctx: torch.Tensor, target: torch.Tensor | None = None):
        """ctx: (B,10,H,W); target: (B,3,H,W) residual ints for NLL."""
        logits = self.net(ctx) * self.log_temp.exp().clamp(0.2, 5.0)
        if target is None:
            return logits
        b, _, h, w = logits.shape
        logits_c = logits.view(b, 3, N_BINS, h, w)
        if target.dim() == 4 and target.shape[-1] == 3:
            target = target.permute(0, 3, 1, 2)  # B,H,W,3 → B,3,H,W
        tgt_b = residual_to_bin(target)  # (B,3,H,W)
        nll = F.cross_entropy(
            logits_c.reshape(b * 3, N_BINS, h * w),
            tgt_b.reshape(b * 3, h * w),
            reduction="none",
        )
        return logits, nll.mean()


def build_context(
    pred: torch.Tensor,
    border_ctx: torch.Tensor,
    prev_res: torch.Tensor,
    motion: torch.Tensor,
) -> torch.Tensor:
    """Assemble (B,10,H,W) context. All float 0..1-ish."""
    return torch.cat(
        [
            pred,
            border_ctx,
            prev_res,
            motion,
        ],
        dim=1,
    )


class SmallHyper(nn.Module):
    """Optional: predict per-tile scale to sharpen CDF (rate gain)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return self.net(pred) + 1e-3
