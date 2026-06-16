# models/drt_baseline.py
"""DRT baseline model for the SRG-CAUN training pipeline.

This baseline removes the three DRT-Net paper innovations:
1. no rectangular transformer, only ordinary full-token cross attention;
2. no SAFA multiscale adaptive aggregation;
3. no contrastive branch or MoCo/CESR dependency.

The interface matches SRG-CAUN: model(lr_hsi, hr_msi) returns a HR-HSI tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VanillaCrossTransformerBlock(nn.Module):
    """Ordinary transformer block, not rectangular-window attention."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + attn_out
        query = query + self.ffn(self.norm_ffn(query))
        return query


class VanillaCrossTransformer(nn.Module):
    """Full-token cross attention used as a non-rectangular transformer baseline."""

    def __init__(self, channels: int, depth: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [VanillaCrossTransformerBlock(channels, heads, dropout=dropout) for _ in range(depth)]
        )
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, hsi_feat: torch.Tensor, msi_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = hsi_feat.shape
        query = hsi_feat.flatten(2).transpose(1, 2)
        context = msi_feat.flatten(2).transpose(1, 2)
        for block in self.blocks:
            query = block(query, context)
        out = query.transpose(1, 2).reshape(b, c, h, w)
        return self.out(out)


@dataclass
class DRTBaselineConfig:
    n_bands: int
    n_msi_bands: int
    scale_ratio: int = 4
    hidden_dim: int = 48
    depth: int = 2
    heads: int = 4
    dropout: float = 0.0
    srf_weights: Optional[object] = None


class DRTBaseline(nn.Module):
    """Plain DRT-style baseline for HSI-MSI fusion.

    The model keeps the dual-branch HSI/MSI interaction, but removes the three
    paper-specific additions: rectangular windows, SAFA, and contrastive learning.
    """

    def __init__(self, cfg: DRTBaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.scale_ratio = cfg.scale_ratio
        self.msi_lift = nn.Sequential(
            nn.Conv2d(cfg.n_msi_bands, cfg.n_bands, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.n_bands, cfg.n_bands, 3, padding=1),
        )
        self.hsi_stem = ConvStem(cfg.n_bands, cfg.hidden_dim)
        self.msi_stem = ConvStem(cfg.n_bands, cfg.hidden_dim)
        self.cross_32 = VanillaCrossTransformer(cfg.hidden_dim, depth=cfg.depth, heads=cfg.heads, dropout=cfg.dropout)
        self.cross_8 = VanillaCrossTransformer(cfg.hidden_dim, depth=cfg.depth, heads=cfg.heads, dropout=cfg.dropout)
        self.fuse = nn.Sequential(
            nn.Conv2d(cfg.hidden_dim * 4 + cfg.n_bands * 2, cfg.hidden_dim, 3, padding=1),
            nn.GELU(),
            ResidualBlock(cfg.hidden_dim),
            ResidualBlock(cfg.hidden_dim),
            nn.Conv2d(cfg.hidden_dim, cfg.n_bands, 3, padding=1),
        )
        self.latest_aux: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _safe_pool(x: torch.Tensor, target_size: int) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h <= target_size or w <= target_size:
            return F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d(x, output_size=(target_size, target_size))

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, return_aux: bool = False):
        target_size = hr_msi.shape[-2:]
        lr_up = F.interpolate(lr_hsi, size=target_size, mode="bicubic", align_corners=False)
        msi_hsi = self.msi_lift(hr_msi)

        hsi_feat = self.hsi_stem(lr_up)
        msi_feat = self.msi_stem(msi_hsi)

        hsi_32 = self._safe_pool(hsi_feat, 32)
        msi_32 = self._safe_pool(msi_feat, 32)
        cross_32 = self.cross_32(hsi_32, msi_32)
        cross_32_up = F.interpolate(cross_32, size=target_size, mode="bilinear", align_corners=False)

        hsi_8 = self._safe_pool(hsi_feat, 8)
        msi_8 = self._safe_pool(msi_feat, 8)
        cross_8 = self.cross_8(hsi_8, msi_8)
        cross_8_up = F.interpolate(cross_8, size=target_size, mode="bilinear", align_corners=False)

        delta = self.fuse(torch.cat([hsi_feat, msi_feat, cross_32_up, cross_8_up, lr_up, msi_hsi], dim=1))
        out = torch.clamp(lr_up + delta, 0.0, 1.0)

        aux = {
            "initial": lr_up.detach(),
            "msi_lift": msi_hsi.detach(),
            "cross_32": cross_32_up.detach(),
            "cross_8": cross_8_up.detach(),
            "stage_outputs": [],
            "shadow_risk": None,
        }
        self.latest_aux = aux
        if return_aux:
            return out, aux
        return out


def build_drt_baseline(
    n_bands: int,
    n_msi_bands: int,
    scale_ratio: int = 4,
    hidden_dim: int = 48,
    depth: int = 2,
    heads: int = 4,
    dropout: float = 0.0,
    srf_weights=None,
) -> DRTBaseline:
    cfg = DRTBaselineConfig(
        n_bands=n_bands,
        n_msi_bands=n_msi_bands,
        scale_ratio=scale_ratio,
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        dropout=dropout,
        srf_weights=srf_weights,
    )
    return DRTBaseline(cfg)
