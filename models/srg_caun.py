# models/srg_caun.py
"""
SRG-CAUN: Shadow-Reliability Guided Content-Adaptive Unfolding Network.

第一版实现目标：
1. 保留 CAUWT 风格的内容自适应展开思想；
2. 构建场景内部非阴影参考库，做跨尺度光谱方向参考匹配；
3. 构建阴影门控的 Contourlet-like 多方向频率先验；
4. 接口保持为 pred = model(lr_hsi, hr_msi)，方便复用现有分析和可视化脚本。

说明：
这里的 ContourletRefinementPrior 使用可训练方向卷积 + 多尺度池化近似实现，
避免第一版引入复杂 contourlet 第三方依赖。后续若需要严格 Contourlet 变换，
可以只替换该模块，主干接口不用变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class ResidualStack(nn.Module):
    def __init__(self, channels: int, depth: int):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SpectralProjector(nn.Module):
    """HSI -> MSI projection and MSI -> HSI spectral lifting."""

    def __init__(self, n_bands: int, n_msi_bands: int, srf_weights=None):
        super().__init__()
        self.n_bands = n_bands
        self.n_msi_bands = n_msi_bands

        if srf_weights is None:
            weight = torch.zeros(n_msi_bands, n_bands, dtype=torch.float32)
            indices = torch.linspace(0, n_bands - 1, n_msi_bands).round().long()
            for i, idx in enumerate(indices):
                weight[i, idx] = 1.0
        else:
            weight = torch.as_tensor(srf_weights, dtype=torch.float32)
            if weight.shape != (n_msi_bands, n_bands):
                raise ValueError(
                    f"srf_weights should be ({n_msi_bands}, {n_bands}), got {tuple(weight.shape)}"
                )

        self.register_buffer("srf", weight)

    def hsi_to_msi(self, hsi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bchw,mc->bmhw", hsi, self.srf)

    def msi_to_hsi(self, msi: torch.Tensor) -> torch.Tensor:
        denom = self.srf.sum(dim=0, keepdim=True).clamp_min(1e-6)
        lifting = self.srf / denom
        return torch.einsum("bmhw,mc->bchw", msi, lifting)


class InitialReconstruction(nn.Module):
    """初始重建模块。

    修改点：HR-MSI 不再先扩展到 HSI 通道数，直接与 LR-HSI 上采样结果拼接。
    这样避免初始阶段用 SRF 转置投影产生过强的伪光谱提示。
    """

    def __init__(self, n_bands: int, n_msi_bands: int, hidden_dim: int, scale_ratio: int, srf_weights=None):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.in_proj = nn.Conv2d(n_bands + n_msi_bands, hidden_dim, 3, padding=1)
        self.body = ResidualStack(hidden_dim, depth=2)
        self.out_proj = nn.Conv2d(hidden_dim, n_bands, 3, padding=1)

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = hr_msi.shape[-2:]
        lr_up = F.interpolate(lr_hsi, size=target_size, mode="bicubic", align_corners=False)
        feat = torch.cat([lr_up, hr_msi], dim=1)
        feat = self.body(self.in_proj(feat))
        pred = torch.clamp(lr_up + self.out_proj(feat), 0.0, 1.0)
        return pred, lr_up, hr_msi


class ShadowReliabilityEstimator(nn.Module):
    """输出连续可靠性图。数值越大表示越可信、越接近非阴影。"""

    def __init__(self, n_bands: int, n_msi_bands: int, hidden_dim: int):
        super().__init__()
        in_channels = n_bands + n_msi_bands + 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 3, 1),
        )

    @staticmethod
    def _gradient_mag(x: torch.Tensor) -> torch.Tensor:
        gx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        gy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return gx + gy

    def forward(self, z: torch.Tensor, hr_msi: torch.Tensor, lr_residual: torch.Tensor, msi_residual: torch.Tensor) -> Dict[str, torch.Tensor]:
        spectral_norm = torch.sqrt(torch.sum(z * z, dim=1, keepdim=True) + 1e-8)
        spectral_norm = spectral_norm / spectral_norm.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        msi_intensity = hr_msi.mean(dim=1, keepdim=True)
        msi_intensity = msi_intensity / msi_intensity.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        lr_res = torch.mean(torch.abs(lr_residual), dim=1, keepdim=True)
        msi_res = torch.mean(torch.abs(msi_residual), dim=1, keepdim=True)
        local_grad = self._gradient_mag(msi_intensity)
        x = torch.cat([z, hr_msi, spectral_norm, msi_intensity, lr_res + msi_res, local_grad], dim=1)
        out = torch.sigmoid(self.net(x))
        reliability = out[:, 0:1]
        return {
            "reliability": reliability,
            "shadow_risk": 1.0 - reliability,
            "boundary": out[:, 1:2],
            "low_reflectance_risk": out[:, 2:3],
        }


class ShadowAwareParameterPredictor(nn.Module):
    def __init__(self, n_bands: int, n_msi_bands: int, hidden_dim: int):
        super().__init__()
        in_channels = n_bands + n_msi_bands + 5
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 6, 1),
        )

    def forward(self, z: torch.Tensor, hr_msi: torch.Tensor, reliability: torch.Tensor, lr_residual_map: torch.Tensor, msi_residual_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        shadow = 1.0 - reliability
        x = torch.cat([z, hr_msi, reliability, shadow, lr_residual_map, msi_residual_map, torch.abs(lr_residual_map - msi_residual_map)], dim=1)
        raw = torch.sigmoid(self.net(x))
        step = 0.02 + 0.18 * raw[:, 0:1]
        w_lr = 0.05 + 0.95 * raw[:, 1:2]
        w_msi = 0.05 + 0.95 * raw[:, 2:3]
        w_ref = raw[:, 3:4] * shadow
        w_freq = raw[:, 4:5]
        w_prior = raw[:, 5:6]
        w_msi = w_msi * (0.35 + 0.65 * reliability)
        w_lr = w_lr * (0.65 + 0.35 * shadow)
        w_freq = w_freq * (0.30 + 0.70 * reliability)
        return {"step": step, "w_lr": w_lr, "w_msi": w_msi, "w_ref": w_ref, "w_freq": w_freq, "w_prior": w_prior}


class PhysicalConsistencyUpdate(nn.Module):
    def __init__(self, n_bands: int, n_msi_bands: int, scale_ratio: int, srf_weights=None):
        super().__init__()
        self.scale_ratio = scale_ratio
        self.projector = SpectralProjector(n_bands, n_msi_bands, srf_weights=srf_weights)

    def residuals(self, z: torch.Tensor, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_lr = F.interpolate(z, size=lr_hsi.shape[-2:], mode="bicubic", align_corners=False)
        lr_residual = pred_lr - lr_hsi
        pred_msi = self.projector.hsi_to_msi(z)
        msi_residual = pred_msi - hr_msi
        return lr_residual, msi_residual

    def forward(self, z: torch.Tensor, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        lr_residual, msi_residual = self.residuals(z, lr_hsi, hr_msi)
        lr_grad = F.interpolate(lr_residual, size=z.shape[-2:], mode="bicubic", align_corners=False)
        msi_grad = self.projector.msi_to_hsi(msi_residual)
        lr_map = torch.mean(torch.abs(lr_grad), dim=1, keepdim=True)
        msi_map = torch.mean(torch.abs(msi_grad), dim=1, keepdim=True)
        grad = params["w_lr"] * lr_grad + params["w_msi"] * msi_grad
        z_next = torch.clamp(z - params["step"] * grad, 0.0, 1.0)
        return z_next, {"lr_residual": lr_residual, "msi_residual": msi_residual, "lr_residual_map": lr_map, "msi_residual_map": msi_map}


class NonShadowReferenceBank(nn.Module):
    """从当前场景内部的高可靠非阴影区域构建参考库。"""

    def __init__(self, n_bands: int, n_msi_bands: int, hidden_dim: int, topk: int = 4):
        super().__init__()
        self.topk = topk
        self.query_encoder = nn.Sequential(nn.Conv2d(n_bands + n_msi_bands + 1, hidden_dim, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 1))
        self.key_encoder = nn.Sequential(nn.Conv2d(n_bands + n_msi_bands + 1, hidden_dim, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 1))
        self.value_encoder = nn.Sequential(nn.Conv2d(n_bands * 2, hidden_dim, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, n_bands, 1))
        self.out_proj = nn.Sequential(nn.Conv2d(n_bands * 2 + 1, hidden_dim, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, n_bands, 3, padding=1))

    @staticmethod
    def _normalize_spectrum(x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-8)
        return x / norm.clamp_min(1e-6)

    @staticmethod
    def _spectral_diff(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, 1:, :, :] - x[:, :-1, :, :]
        return F.pad(diff, (0, 0, 0, 0, 0, 1))

    def _match_one_scale(self, z: torch.Tensor, hr_msi: torch.Tensor, reliability: torch.Tensor, scale: int) -> torch.Tensor:
        if scale > 1:
            z_s = F.avg_pool2d(z, kernel_size=scale, stride=scale)
            msi_s = F.avg_pool2d(hr_msi, kernel_size=scale, stride=scale)
            rel_s = F.avg_pool2d(reliability, kernel_size=scale, stride=scale)
        else:
            z_s, msi_s, rel_s = z, hr_msi, reliability

        b, c, h, w = z_s.shape
        query_in = torch.cat([z_s, msi_s, 1.0 - rel_s], dim=1)
        key_in = torch.cat([z_s, msi_s, rel_s], dim=1)
        q = F.normalize(self.query_encoder(query_in).flatten(2).transpose(1, 2), dim=-1)
        k = F.normalize(self.key_encoder(key_in).flatten(2).transpose(1, 2), dim=-1)
        spec_dir = self._normalize_spectrum(z_s)
        spec_diff = self._spectral_diff(spec_dir)
        value_map = self.value_encoder(torch.cat([spec_dir, spec_diff], dim=1))
        v = value_map.flatten(2).transpose(1, 2)
        ref_mask = (rel_s.flatten(2) > 0.55).float()
        shadow_query_mask = (1.0 - rel_s).flatten(2).transpose(1, 2)
        sim = torch.bmm(q, k.transpose(1, 2)).masked_fill(ref_mask <= 0.0, -1e4)
        k_eff = min(self.topk, sim.shape[-1])
        topv, topi = torch.topk(sim, k=k_eff, dim=-1)
        attn = torch.softmax(topv, dim=-1)
        gather_index = topi.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
        v_expand = v.unsqueeze(1).expand(-1, topi.shape[1], -1, -1)
        selected = torch.gather(v_expand, 2, gather_index)
        matched = torch.sum(selected * attn.unsqueeze(-1), dim=2) * shadow_query_mask
        matched = matched.transpose(1, 2).reshape(b, c, h, w)
        if scale > 1:
            matched = F.interpolate(matched, size=z.shape[-2:], mode="bilinear", align_corners=False)
        return matched

    def forward(self, z: torch.Tensor, hr_msi: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        matched = 0.0
        for scale, weight in zip([1, 2, 4], [0.50, 0.30, 0.20]):
            if z.shape[-1] >= scale and z.shape[-2] >= scale:
                matched = matched + weight * self._match_one_scale(z, hr_msi, reliability, scale)
        z_dir = self._normalize_spectrum(z)
        ref_residual = matched - z_dir
        out = self.out_proj(torch.cat([z_dir, ref_residual, 1.0 - reliability], dim=1))
        return out * (1.0 - reliability)


class DirectionalHighPass(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernels = torch.tensor(
            [
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                [[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]],
                [[[2, 1, 0], [1, 0, -1], [0, -1, -2]]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("kernels", kernels)
        self.channels = channels
        self.mix = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(4):
            kernel = self.kernels[i].repeat(self.channels, 1, 1, 1)
            outs.append(F.conv2d(x, kernel, padding=1, groups=self.channels))
        return self.mix(torch.cat(outs, dim=1))


class ContourletRefinementPrior(nn.Module):
    def __init__(self, n_bands: int, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Conv2d(n_bands * 2 + 2, hidden_dim, 3, padding=1)
        self.low_branch = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(), ResidualBlock(hidden_dim))
        self.high_branch = DirectionalHighPass(hidden_dim)
        self.gate = nn.Sequential(nn.Conv2d(hidden_dim * 2 + 2, hidden_dim, 3, padding=1), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.Sigmoid())
        self.out_proj = nn.Conv2d(hidden_dim, n_bands, 3, padding=1)

    def forward(self, z: torch.Tensor, ref_residual: torch.Tensor, reliability: torch.Tensor, consistency_residual: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, ref_residual, reliability, consistency_residual], dim=1)
        feat = self.in_proj(x)
        low = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        low = self.low_branch(low)
        high = self.high_branch(feat - low)
        gate = self.gate(torch.cat([low, high, reliability, consistency_residual], dim=1)) * (0.25 + 0.75 * reliability)
        return self.out_proj(low + gate * high)


class SRGCAUNStage(nn.Module):
    def __init__(self, n_bands: int, n_msi_bands: int, hidden_dim: int, scale_ratio: int, srf_weights=None, topk: int = 4):
        super().__init__()
        self.param_predictor = ShadowAwareParameterPredictor(n_bands, n_msi_bands, hidden_dim)
        self.physics = PhysicalConsistencyUpdate(n_bands, n_msi_bands, scale_ratio, srf_weights=srf_weights)
        self.reference = NonShadowReferenceBank(n_bands, n_msi_bands, hidden_dim, topk=topk)
        self.contourlet = ContourletRefinementPrior(n_bands, hidden_dim)
        self.refine = nn.Sequential(nn.Conv2d(n_bands * 3 + 1, hidden_dim, 3, padding=1), nn.GELU(), ResidualBlock(hidden_dim), nn.Conv2d(hidden_dim, n_bands, 3, padding=1))

    def forward(self, z: torch.Tensor, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, reliability: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        lr_residual, msi_residual = self.physics.residuals(z, lr_hsi, hr_msi)
        lr_map = F.interpolate(torch.mean(torch.abs(lr_residual), dim=1, keepdim=True), size=z.shape[-2:], mode="bilinear", align_corners=False)
        msi_map = torch.mean(torch.abs(self.physics.projector.msi_to_hsi(msi_residual)), dim=1, keepdim=True)
        params = self.param_predictor(z, hr_msi, reliability, lr_map, msi_map)
        z_data, residual_info = self.physics(z, lr_hsi, hr_msi, params)
        ref_residual = self.reference(z_data, hr_msi, reliability)
        consistency = torch.clamp(residual_info["lr_residual_map"] + residual_info["msi_residual_map"], 0.0, 1.0)
        freq_residual = self.contourlet(z_data, ref_residual, reliability, consistency)
        prior_delta = params["w_ref"] * ref_residual + params["w_freq"] * freq_residual
        fused_delta = self.refine(torch.cat([z_data, ref_residual, freq_residual, reliability], dim=1))
        z_next = torch.clamp(z_data + params["w_prior"] * fused_delta + prior_delta, 0.0, 1.0)
        return z_next, {
            "reliability": reliability,
            "lr_residual_map": residual_info["lr_residual_map"].detach(),
            "msi_residual_map": residual_info["msi_residual_map"].detach(),
            "ref_residual": ref_residual.detach(),
            "freq_residual": freq_residual.detach(),
        }


@dataclass
class SRGCAUNConfig:
    n_bands: int
    n_msi_bands: int
    scale_ratio: int = 4
    hidden_dim: int = 48
    num_stages: int = 3
    ref_topk: int = 4
    srf_weights: Optional[object] = None


class SRGCAUN(nn.Module):
    def __init__(self, cfg: SRGCAUNConfig):
        super().__init__()
        self.cfg = cfg
        self.initial = InitialReconstruction(cfg.n_bands, cfg.n_msi_bands, cfg.hidden_dim, cfg.scale_ratio, srf_weights=cfg.srf_weights)
        self.projector = SpectralProjector(cfg.n_bands, cfg.n_msi_bands, srf_weights=cfg.srf_weights)
        self.reliability_estimator = ShadowReliabilityEstimator(cfg.n_bands, cfg.n_msi_bands, cfg.hidden_dim)
        self.stages = nn.ModuleList([
            SRGCAUNStage(cfg.n_bands, cfg.n_msi_bands, cfg.hidden_dim, cfg.scale_ratio, srf_weights=cfg.srf_weights, topk=cfg.ref_topk)
            for _ in range(cfg.num_stages)
        ])
        self.final_head = nn.Sequential(nn.Conv2d(cfg.n_bands, cfg.hidden_dim, 3, padding=1), nn.GELU(), ResidualBlock(cfg.hidden_dim), nn.Conv2d(cfg.hidden_dim, cfg.n_bands, 3, padding=1))
        self.latest_aux: Dict[str, torch.Tensor] = {}

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, return_aux: bool = False):
        z, lr_up, msi_raw = self.initial(lr_hsi, hr_msi)
        pred_lr_up = F.interpolate(
            F.interpolate(z, size=lr_hsi.shape[-2:], mode="bicubic", align_corners=False),
            size=z.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        lr_residual_proxy = z - pred_lr_up
        msi_residual = self.projector.hsi_to_msi(z) - hr_msi
        msi_residual_lift = self.projector.msi_to_hsi(msi_residual)
        reliability_info = self.reliability_estimator(z, hr_msi, lr_residual_proxy, msi_residual_lift)
        reliability = reliability_info["reliability"]
        stage_infos: List[Dict[str, torch.Tensor]] = []
        for stage in self.stages:
            z, info = stage(z, lr_hsi, hr_msi, reliability)
            msi_residual = self.projector.hsi_to_msi(z) - hr_msi
            msi_residual_lift = self.projector.msi_to_hsi(msi_residual)
            reliability_info = self.reliability_estimator(z, hr_msi, lr_residual_proxy, msi_residual_lift)
            reliability = reliability_info["reliability"]
            stage_infos.append(info)
        out = torch.clamp(z + self.final_head(z), 0.0, 1.0)
        aux = {
            "initial": z.detach(),
            "lr_up": lr_up.detach(),
            "msi_raw": msi_raw.detach(),
            "reliability": reliability.detach(),
            "shadow_risk": (1.0 - reliability).detach(),
            "stage_infos": stage_infos,
        }
        self.latest_aux = aux
        if return_aux:
            return out, aux
        return out


def build_srg_caun(
    n_bands: int,
    n_msi_bands: int,
    scale_ratio: int = 4,
    hidden_dim: int = 48,
    num_stages: int = 3,
    ref_topk: int = 4,
    srf_weights=None,
) -> SRGCAUN:
    cfg = SRGCAUNConfig(
        n_bands=n_bands,
        n_msi_bands=n_msi_bands,
        scale_ratio=scale_ratio,
        hidden_dim=hidden_dim,
        num_stages=num_stages,
        ref_topk=ref_topk,
        srf_weights=srf_weights,
    )
    return SRGCAUN(cfg)
