# models/srg_caun_hier_match.py
"""Independent SRG-CAUN variant with hierarchical coarse-to-fine reference matching.

This file intentionally leaves models/srg_caun.py unchanged.  The main change is
inside the reference bank:
- scale=4 uses global non-shadow attention;
- scale=2 uses 11x11 local matching plus scale=4 coarse positions refined in a
  small scale=2 window;
- scale=1 uses 11x11 local matching plus scale=2 and scale=4 coarse positions
  refined in small scale=1 windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .srg_caun import (
    ContourletRefinementPrior,
    InitialReconstruction,
    PhysicalConsistencyUpdate,
    ResidualBlock,
    ShadowAwareParameterPredictor,
    ShadowReliabilityEstimator,
    SpectralProjector,
)


class HierarchicalCoarseToFineReferenceBank(nn.Module):
    """Shadow-aware non-shadow reference bank with coarse-to-fine matching."""

    def __init__(
        self,
        n_bands: int,
        n_msi_bands: int,
        hidden_dim: int,
        topk: int = 4,
        local_window: int = 11,
        fine_window: int = 5,
        ref_threshold: float = 0.60,
    ):
        super().__init__()
        if local_window % 2 == 0 or fine_window % 2 == 0:
            raise ValueError("local_window and fine_window should be odd numbers.")
        self.topk = topk
        self.local_window = local_window
        self.fine_window = fine_window
        self.ref_threshold = ref_threshold
        self.query_encoder = nn.Sequential(
            nn.Conv2d(n_bands + n_msi_bands + 1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.key_encoder = nn.Sequential(
            nn.Conv2d(n_bands + n_msi_bands + 1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.value_encoder = nn.Sequential(
            nn.Conv2d(n_bands * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_bands, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(n_bands * 2 + 1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_bands, 3, padding=1),
        )

    @staticmethod
    def _normalize_spectrum(x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-8)
        return x / norm.clamp_min(1e-6)

    @staticmethod
    def _spectral_diff(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, 1:, :, :] - x[:, :-1, :, :]
        return F.pad(diff, (0, 0, 0, 0, 0, 1))

    def _encode_scale(self, z: torch.Tensor, hr_msi: torch.Tensor, reliability: torch.Tensor, scale: int) -> Dict[str, torch.Tensor]:
        if scale > 1:
            z_s = F.avg_pool2d(z, kernel_size=scale, stride=scale)
            msi_s = F.avg_pool2d(hr_msi, kernel_size=scale, stride=scale)
            rel_s = F.avg_pool2d(reliability, kernel_size=scale, stride=scale)
        else:
            z_s, msi_s, rel_s = z, hr_msi, reliability

        query_in = torch.cat([z_s, msi_s, 1.0 - rel_s], dim=1)
        key_in = torch.cat([z_s, msi_s, rel_s], dim=1)
        q_map = F.normalize(self.query_encoder(query_in), dim=1)
        k_map = F.normalize(self.key_encoder(key_in), dim=1)
        spec_dir = self._normalize_spectrum(z_s)
        spec_diff = self._spectral_diff(spec_dir)
        value_map = self.value_encoder(torch.cat([spec_dir, spec_diff], dim=1))
        return {"q": q_map, "k": k_map, "v": value_map, "rel": rel_s, "z_dir": spec_dir}

    @staticmethod
    def _window_abs_indices(h: int, w: int, window: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = window // 2
        y = torch.arange(h, device=device).repeat_interleave(w)
        x = torch.arange(w, device=device).repeat(h)
        offsets = torch.arange(-pad, pad + 1, device=device)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        yy = yy.reshape(-1, 1)
        xx = xx.reshape(-1, 1)
        cand_y = y.unsqueeze(0) + yy
        cand_x = x.unsqueeze(0) + xx
        valid = (cand_y >= 0) & (cand_y < h) & (cand_x >= 0) & (cand_x < w)
        cand_y = cand_y.clamp(0, h - 1)
        cand_x = cand_x.clamp(0, w - 1)
        return cand_y * w + cand_x, valid

    @staticmethod
    def _query_to_coarse_index(h: int, w: int, coarse_h: int, coarse_w: int, factor: int, device: torch.device) -> torch.Tensor:
        y = torch.arange(h, device=device).repeat_interleave(w)
        x = torch.arange(w, device=device).repeat(h)
        cy = (y // factor).clamp(0, coarse_h - 1)
        cx = (x // factor).clamp(0, coarse_w - 1)
        return cy * coarse_w + cx

    def _local_candidates(self, enc: Dict[str, torch.Tensor], window: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q_map, k_map, rel = enc["q"], enc["k"], enc["rel"]
        b, d, h, w = q_map.shape
        n = h * w
        pad = window // 2
        win_area = window * window

        k_unfold = F.unfold(k_map, kernel_size=window, padding=pad)
        k_unfold = k_unfold.view(b, d, win_area, n).permute(0, 3, 2, 1)
        q_flat = q_map.flatten(2).transpose(1, 2).unsqueeze(2)
        sim = torch.sum(q_flat * k_unfold, dim=-1)

        ref_mask = (rel > self.ref_threshold).float()
        mask = F.unfold(ref_mask, kernel_size=window, padding=pad).view(b, win_area, n).transpose(1, 2) > 0
        _, valid_table = self._window_abs_indices(h, w, window, q_map.device)
        mask = mask & valid_table.transpose(0, 1).unsqueeze(0)
        sim = sim.masked_fill(~mask, -1e4)

        idx_table, _ = self._window_abs_indices(h, w, window, q_map.device)
        cand_idx = idx_table.transpose(0, 1).unsqueeze(0).expand(b, -1, -1)
        return cand_idx, sim

    def _global_candidates(self, enc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        q_map, k_map, rel = enc["q"], enc["k"], enc["rel"]
        b, _, h, w = q_map.shape
        n = h * w
        q = q_map.flatten(2).transpose(1, 2)
        k = k_map.flatten(2).transpose(1, 2)
        sim = torch.bmm(q, k.transpose(1, 2))
        ref_mask = (rel.flatten(2) > self.ref_threshold)
        sim = sim.masked_fill(~ref_mask, -1e4)
        cand_idx = torch.arange(n, device=q_map.device).view(1, 1, n).expand(b, n, n)
        return cand_idx, sim

    def _coarse_refine_candidates(
        self,
        query_enc: Dict[str, torch.Tensor],
        target_enc: Dict[str, torch.Tensor],
        coarse_topi: torch.Tensor,
        coarse_hw: Tuple[int, int],
        factor: int,
        fine_window: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_map = query_enc["q"]
        k_map, rel = target_enc["k"], target_enc["rel"]
        b, d, hq, wq = q_map.shape
        _, _, ht, wt = k_map.shape
        coarse_h, coarse_w = coarse_hw
        n_query = hq * wq
        k_coarse = coarse_topi.shape[-1]
        pad = fine_window // 2
        win_area = fine_window * fine_window

        coarse_y = coarse_topi // coarse_w
        coarse_x = coarse_topi % coarse_w
        center_y = coarse_y * factor + factor // 2
        center_x = coarse_x * factor + factor // 2

        offsets = torch.arange(-pad, pad + 1, device=q_map.device)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        yy = yy.reshape(1, 1, 1, win_area)
        xx = xx.reshape(1, 1, 1, win_area)
        cand_y = center_y.unsqueeze(-1) + yy
        cand_x = center_x.unsqueeze(-1) + xx
        valid = (cand_y >= 0) & (cand_y < ht) & (cand_x >= 0) & (cand_x < wt)
        cand_y = cand_y.clamp(0, ht - 1)
        cand_x = cand_x.clamp(0, wt - 1)
        cand_idx = (cand_y * wt + cand_x).reshape(b, n_query, k_coarse * win_area)
        valid = valid.reshape(b, n_query, k_coarse * win_area)

        q_flat = q_map.flatten(2).transpose(1, 2)
        k_flat = k_map.flatten(2).transpose(1, 2)
        rel_flat = rel.flatten(2).transpose(1, 2)
        flat_idx = cand_idx.reshape(b, -1)
        gathered_k = torch.gather(k_flat, 1, flat_idx.unsqueeze(-1).expand(-1, -1, d))
        gathered_k = gathered_k.view(b, n_query, k_coarse * win_area, d)
        gathered_rel = torch.gather(rel_flat, 1, flat_idx.unsqueeze(-1)).view(b, n_query, k_coarse * win_area)
        sim = torch.sum(q_flat.unsqueeze(2) * gathered_k, dim=-1)
        sim = sim.masked_fill(~valid | (gathered_rel <= self.ref_threshold), -1e4)
        return cand_idx, sim

    def _select_match(
        self,
        enc: Dict[str, torch.Tensor],
        cand_idx: torch.Tensor,
        cand_sim: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_map, rel = enc["v"], enc["rel"]
        b, c, h, w = v_map.shape
        n = h * w
        k_eff = min(self.topk, cand_sim.shape[-1])
        topv, top_pos = torch.topk(cand_sim, k=k_eff, dim=-1)
        top_idx = torch.gather(cand_idx, dim=-1, index=top_pos)

        valid = topv > -9999.0
        attn_logits = topv.masked_fill(~valid, -1e4)
        attn = torch.softmax(attn_logits, dim=-1) * valid.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        v_flat = v_map.flatten(2).transpose(1, 2)
        flat_idx = top_idx.reshape(b, -1)
        selected = torch.gather(v_flat, 1, flat_idx.unsqueeze(-1).expand(-1, -1, c))
        selected = selected.view(b, n, k_eff, c)
        matched = torch.sum(selected * attn.unsqueeze(-1), dim=2)
        shadow_query_mask = (1.0 - rel).flatten(2).transpose(1, 2)
        matched = matched * shadow_query_mask
        return matched.transpose(1, 2).reshape(b, c, h, w), top_idx

    def forward(self, z: torch.Tensor, hr_msi: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        enc1 = self._encode_scale(z, hr_msi, reliability, scale=1)
        enc2 = self._encode_scale(z, hr_msi, reliability, scale=2) if min(z.shape[-2:]) >= 2 else None
        enc4 = self._encode_scale(z, hr_msi, reliability, scale=4) if min(z.shape[-2:]) >= 4 else None

        matched_maps: List[Tuple[torch.Tensor, float]] = []
        topi4 = None
        topi2 = None

        if enc4 is not None:
            cand4, sim4 = self._global_candidates(enc4)
            matched4, topi4 = self._select_match(enc4, cand4, sim4)
            matched_maps.append((F.interpolate(matched4, size=z.shape[-2:], mode="bilinear", align_corners=False), 0.20))

        if enc2 is not None:
            cands2: List[torch.Tensor] = []
            sims2: List[torch.Tensor] = []
            cand2_local, sim2_local = self._local_candidates(enc2, self.local_window)
            cands2.append(cand2_local)
            sims2.append(sim2_local)
            if topi4 is not None:
                h2, w2 = enc2["q"].shape[-2:]
                h4, w4 = enc4["q"].shape[-2:]
                q4_idx = self._query_to_coarse_index(h2, w2, h4, w4, factor=2, device=z.device)
                topi4_for_2 = torch.gather(topi4, 1, q4_idx.view(1, -1, 1).expand(topi4.shape[0], -1, topi4.shape[-1]))
                cand2_cf, sim2_cf = self._coarse_refine_candidates(enc2, enc2, topi4_for_2, (h4, w4), factor=2, fine_window=self.fine_window)
                cands2.append(cand2_cf)
                sims2.append(sim2_cf)
            matched2, topi2 = self._select_match(enc2, torch.cat(cands2, dim=-1), torch.cat(sims2, dim=-1))
            matched_maps.append((F.interpolate(matched2, size=z.shape[-2:], mode="bilinear", align_corners=False), 0.30))

        cands1: List[torch.Tensor] = []
        sims1: List[torch.Tensor] = []
        cand1_local, sim1_local = self._local_candidates(enc1, self.local_window)
        cands1.append(cand1_local)
        sims1.append(sim1_local)

        h1, w1 = enc1["q"].shape[-2:]
        if topi2 is not None and enc2 is not None:
            h2, w2 = enc2["q"].shape[-2:]
            q2_idx = self._query_to_coarse_index(h1, w1, h2, w2, factor=2, device=z.device)
            topi2_for_1 = torch.gather(topi2, 1, q2_idx.view(1, -1, 1).expand(topi2.shape[0], -1, topi2.shape[-1]))
            cand1_from2, sim1_from2 = self._coarse_refine_candidates(enc1, enc1, topi2_for_1, (h2, w2), factor=2, fine_window=self.fine_window)
            cands1.append(cand1_from2)
            sims1.append(sim1_from2)
        if topi4 is not None and enc4 is not None:
            h4, w4 = enc4["q"].shape[-2:]
            q4_idx = self._query_to_coarse_index(h1, w1, h4, w4, factor=4, device=z.device)
            topi4_for_1 = torch.gather(topi4, 1, q4_idx.view(1, -1, 1).expand(topi4.shape[0], -1, topi4.shape[-1]))
            cand1_from4, sim1_from4 = self._coarse_refine_candidates(enc1, enc1, topi4_for_1, (h4, w4), factor=4, fine_window=self.fine_window)
            cands1.append(cand1_from4)
            sims1.append(sim1_from4)
        matched1, _ = self._select_match(enc1, torch.cat(cands1, dim=-1), torch.cat(sims1, dim=-1))
        matched_maps.append((matched1, 0.50))

        weight_sum = sum(weight for _, weight in matched_maps)
        matched = sum(match * weight for match, weight in matched_maps) / max(weight_sum, 1e-6)
        z_dir = self._normalize_spectrum(z)
        ref_residual = matched - z_dir
        out = self.out_proj(torch.cat([z_dir, ref_residual, 1.0 - reliability], dim=1))
        return out * (1.0 - reliability)


class SRGCAUNHierMatchStage(nn.Module):
    def __init__(
        self,
        n_bands: int,
        n_msi_bands: int,
        hidden_dim: int,
        scale_ratio: int,
        srf_weights=None,
        topk: int = 4,
        local_window: int = 11,
        fine_window: int = 5,
    ):
        super().__init__()
        self.param_predictor = ShadowAwareParameterPredictor(n_bands, n_msi_bands, hidden_dim)
        self.physics = PhysicalConsistencyUpdate(n_bands, n_msi_bands, scale_ratio, srf_weights=srf_weights)
        self.reference = HierarchicalCoarseToFineReferenceBank(
            n_bands,
            n_msi_bands,
            hidden_dim,
            topk=topk,
            local_window=local_window,
            fine_window=fine_window,
        )
        self.contourlet = ContourletRefinementPrior(n_bands, hidden_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(n_bands * 3 + 1, hidden_dim, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, n_bands, 3, padding=1),
        )

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
class SRGCAUNHierMatchConfig:
    n_bands: int
    n_msi_bands: int
    scale_ratio: int = 4
    hidden_dim: int = 48
    num_stages: int = 3
    ref_topk: int = 4
    ref_window: int = 11
    ref_fine_window: int = 5
    srf_weights: Optional[object] = None


class SRGCAUNHierMatch(nn.Module):
    def __init__(self, cfg: SRGCAUNHierMatchConfig):
        super().__init__()
        self.cfg = cfg
        self.initial = InitialReconstruction(cfg.n_bands, cfg.n_msi_bands, cfg.hidden_dim, cfg.scale_ratio, srf_weights=cfg.srf_weights)
        self.projector = SpectralProjector(cfg.n_bands, cfg.n_msi_bands, srf_weights=cfg.srf_weights)
        self.reliability_estimator = ShadowReliabilityEstimator(cfg.n_bands, cfg.n_msi_bands, cfg.hidden_dim)
        self.stages = nn.ModuleList([
            SRGCAUNHierMatchStage(
                cfg.n_bands,
                cfg.n_msi_bands,
                cfg.hidden_dim,
                cfg.scale_ratio,
                srf_weights=cfg.srf_weights,
                topk=cfg.ref_topk,
                local_window=cfg.ref_window,
                fine_window=cfg.ref_fine_window,
            )
            for _ in range(cfg.num_stages)
        ])
        self.final_head = nn.Sequential(
            nn.Conv2d(cfg.n_bands, cfg.hidden_dim, 3, padding=1),
            nn.GELU(),
            ResidualBlock(cfg.hidden_dim),
            nn.Conv2d(cfg.hidden_dim, cfg.n_bands, 3, padding=1),
        )
        self.latest_aux: Dict[str, torch.Tensor] = {}

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, return_aux: bool = False):
        z, lr_up, msi_raw = self.initial(lr_hsi, hr_msi)
        initial_z = z
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
        stage_outputs: List[torch.Tensor] = []
        for stage in self.stages:
            z, info = stage(z, lr_hsi, hr_msi, reliability)
            stage_outputs.append(z)
            msi_residual = self.projector.hsi_to_msi(z) - hr_msi
            msi_residual_lift = self.projector.msi_to_hsi(msi_residual)
            reliability_info = self.reliability_estimator(z, hr_msi, lr_residual_proxy, msi_residual_lift)
            reliability = reliability_info["reliability"]
            stage_infos.append(info)
        out = torch.clamp(z + self.final_head(z), 0.0, 1.0)
        aux = {
            "initial": initial_z.detach(),
            "lr_up": lr_up.detach(),
            "msi_raw": msi_raw.detach(),
            "reliability": reliability.detach(),
            "shadow_risk": (1.0 - reliability).detach(),
            "learned_reliability": reliability_info.get("learned_reliability", reliability).detach(),
            "prior_reliability": reliability_info.get("prior_reliability", reliability).detach(),
            "prior_shadow_risk": reliability_info.get("prior_shadow_risk", 1.0 - reliability).detach(),
            "stage_infos": stage_infos,
            "stage_outputs": stage_outputs,
        }
        latest_aux = dict(aux)
        latest_aux["stage_outputs"] = [stage_out.detach() for stage_out in stage_outputs]
        self.latest_aux = latest_aux
        if return_aux:
            return out, aux
        return out


def build_srg_caun_hier_match(
    n_bands: int,
    n_msi_bands: int,
    scale_ratio: int = 4,
    hidden_dim: int = 48,
    num_stages: int = 3,
    ref_topk: int = 4,
    ref_window: int = 11,
    ref_fine_window: int = 5,
    srf_weights=None,
) -> SRGCAUNHierMatch:
    cfg = SRGCAUNHierMatchConfig(
        n_bands=n_bands,
        n_msi_bands=n_msi_bands,
        scale_ratio=scale_ratio,
        hidden_dim=hidden_dim,
        num_stages=num_stages,
        ref_topk=ref_topk,
        ref_window=ref_window,
        ref_fine_window=ref_fine_window,
        srf_weights=srf_weights,
    )
    return SRGCAUNHierMatch(cfg)
