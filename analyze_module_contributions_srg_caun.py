# analyze_module_contributions_srg_caun.py
"""Diagnose PSNR/SAM changes after each SRG-CAUN module.

该脚本不训练模型，只对一个 checkpoint 执行逐模块前向诊断，记录：
1. LR-HSI bicubic 上采样结果；
2. 初始重建结果；
3. 每个展开阶段中的物理一致性更新结果；
4. 单独加入参考残差后的结果；
5. 单独加入频率残差后的结果；
6. 参考残差与频率残差共同加入后的结果；
7. 单独加入融合修正分支后的结果；
8. 当前阶段完整输出结果；
9. final head 输出结果。

同时在 all / gt_norm_shadow / gt_norm_nonshadow /
model_risk_shadow / model_risk_nonshadow / mask_overlap 等区域统计 PSNR 和 SAM，
用于判断到底哪些模块在改善整体区域、阴影区域或非阴影区域。

说明：
该脚本是“单次前向路径诊断”，不是重新训练后的严格消融。它用于定位模块在当前
checkpoint 上对中间结果的直接影响。
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from config import parse_args, print_config
from data_loader import build_loaders
from metrics import calc_metrics
from models import build_srg_caun
from utils import get_device, load_checkpoint, move_to_device, set_seed


def parse_diag_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--ref_topk", type=int, default=4)
    parser.add_argument("--shadow_percentile", type=float, default=20.0)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--save_csv", type=str, default="")
    args, remaining = parser.parse_known_args()

    cfg = parse_args(remaining)
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def checkpoint_name(path: str) -> str:
    name = os.path.basename(path) if path else "no_checkpoint"
    for suffix in [".pth", ".pt", ".ckpt"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("/", "_").replace("\\", "_")


def make_gt_norm_shadow_mask(gt: torch.Tensor, percentile: float) -> torch.Tensor:
    """gt: B,C,H,W -> B,1,H,W bool."""
    norm = torch.sqrt(torch.sum(gt * gt, dim=1, keepdim=True) + 1e-8)
    q = torch.quantile(norm.flatten(), percentile / 100.0)
    return norm <= q


def make_model_risk_percentile_mask(shadow_risk: torch.Tensor, percentile: float) -> torch.Tensor:
    """shadow_risk: B,1,H,W -> B,1,H,W bool."""
    q = torch.quantile(shadow_risk.flatten(), 1.0 - percentile / 100.0)
    return shadow_risk >= q


def masked_tensor(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """x: B,C,H,W, mask: B,1,H,W -> 1,C,N,1."""
    if mask.sum().item() == 0:
        return None
    mask_hw = mask.squeeze(1).bool()
    pixels = x.detach().permute(0, 2, 3, 1)[mask_hw]  # N,C
    if pixels.numel() == 0:
        return None
    return pixels.transpose(0, 1).unsqueeze(0).unsqueeze(-1).contiguous()


def calc_region_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, scale_ratio: int) -> Dict[str, float] | None:
    pred_m = masked_tensor(pred, mask)
    gt_m = masked_tensor(gt, mask)
    if pred_m is None or gt_m is None:
        return None
    metrics = calc_metrics(pred_m, gt_m, scale_ratio)
    metrics["pixel_count"] = int(mask.sum().item())
    return metrics


def region_masks(gt: torch.Tensor, shadow_risk: torch.Tensor, percentile: float) -> Dict[str, torch.Tensor]:
    gt_shadow = make_gt_norm_shadow_mask(gt, percentile)
    risk_shadow = make_model_risk_percentile_mask(shadow_risk, percentile)
    overlap = torch.logical_and(gt_shadow, risk_shadow)
    union = torch.logical_or(gt_shadow, risk_shadow)
    return {
        "all": torch.ones_like(gt_shadow, dtype=torch.bool),
        "gt_norm_shadow": gt_shadow,
        "gt_norm_nonshadow": ~gt_shadow,
        "model_risk_shadow": risk_shadow,
        "model_risk_nonshadow": ~risk_shadow,
        "mask_overlap": overlap,
        "gt_only": torch.logical_and(gt_shadow, ~risk_shadow),
        "risk_only": torch.logical_and(risk_shadow, ~gt_shadow),
        "mask_union": union,
    }


def append_record(records: List[Tuple[str, torch.Tensor]], name: str, tensor: torch.Tensor):
    records.append((name, torch.clamp(tensor.detach(), 0.0, 1.0)))


@torch.no_grad()
def instrumented_forward(model, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> Tuple[List[Tuple[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    records: List[Tuple[str, torch.Tensor]] = []

    z, lr_up, msi_raw = model.initial(lr_hsi, hr_msi)
    append_record(records, "00_lr_up", lr_up)
    append_record(records, "01_initial", z)

    pred_lr_up = F.interpolate(
        F.interpolate(z, size=lr_hsi.shape[-2:], mode="bicubic", align_corners=False),
        size=z.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    lr_residual_proxy = z - pred_lr_up
    msi_residual = model.projector.hsi_to_msi(z) - hr_msi
    msi_residual_lift = model.projector.msi_to_hsi(msi_residual)
    reliability_info = model.reliability_estimator(z, hr_msi, lr_residual_proxy, msi_residual_lift)
    reliability = reliability_info["reliability"]

    for stage_idx, stage in enumerate(model.stages, start=1):
        lr_residual, msi_residual = stage.physics.residuals(z, lr_hsi, hr_msi)
        lr_map = F.interpolate(
            torch.mean(torch.abs(lr_residual), dim=1, keepdim=True),
            size=z.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        msi_map = torch.mean(torch.abs(stage.physics.projector.msi_to_hsi(msi_residual)), dim=1, keepdim=True)

        params = stage.param_predictor(z, hr_msi, reliability, lr_map, msi_map)
        z_data, residual_info = stage.physics(z, lr_hsi, hr_msi, params)
        append_record(records, f"s{stage_idx:02d}_physics", z_data)

        ref_residual = stage.reference(z_data, hr_msi, reliability)
        z_ref_only = torch.clamp(z_data + params["w_ref"] * ref_residual, 0.0, 1.0)
        append_record(records, f"s{stage_idx:02d}_ref_only", z_ref_only)

        consistency = torch.clamp(residual_info["lr_residual_map"] + residual_info["msi_residual_map"], 0.0, 1.0)
        freq_residual = stage.contourlet(z_data, ref_residual, reliability, consistency)
        z_freq_only = torch.clamp(z_data + params["w_freq"] * freq_residual, 0.0, 1.0)
        append_record(records, f"s{stage_idx:02d}_freq_only", z_freq_only)

        prior_delta = params["w_ref"] * ref_residual + params["w_freq"] * freq_residual
        z_ref_freq = torch.clamp(z_data + prior_delta, 0.0, 1.0)
        append_record(records, f"s{stage_idx:02d}_ref_freq", z_ref_freq)

        fused_delta = stage.refine(torch.cat([z_data, ref_residual, freq_residual, reliability], dim=1))
        z_fused_only = torch.clamp(z_data + params["w_prior"] * fused_delta, 0.0, 1.0)
        append_record(records, f"s{stage_idx:02d}_fused_only", z_fused_only)

        z = torch.clamp(z_data + params["w_prior"] * fused_delta + prior_delta, 0.0, 1.0)
        append_record(records, f"s{stage_idx:02d}_stage_output", z)

        msi_residual = model.projector.hsi_to_msi(z) - hr_msi
        msi_residual_lift = model.projector.msi_to_hsi(msi_residual)
        reliability_info = model.reliability_estimator(z, hr_msi, lr_residual_proxy, msi_residual_lift)
        reliability = reliability_info["reliability"]

    final_out = torch.clamp(z + model.final_head(z), 0.0, 1.0)
    append_record(records, "final_head", final_out)

    aux = {
        "reliability": reliability.detach(),
        "shadow_risk": (1.0 - reliability).detach(),
        "learned_reliability": reliability_info.get("learned_reliability", reliability).detach(),
        "prior_reliability": reliability_info.get("prior_reliability", reliability).detach(),
        "prior_shadow_risk": reliability_info.get("prior_shadow_risk", 1.0 - reliability).detach(),
    }
    return records, aux


def average_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = ["pixel_count", "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC"]
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[key] = sum(values) / len(values)
    return out


@torch.no_grad()
def main():
    cfg = parse_diag_args()
    print_config(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    _, test_loader, info = build_loaders(cfg)
    model = build_srg_caun(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_select_bands"],
        scale_ratio=cfg.scale_ratio,
        hidden_dim=cfg.hidden_dim,
        num_stages=cfg.num_stages,
        ref_topk=cfg.ref_topk,
        srf_weights=info.get("srf_weights", None),
    ).to(device)

    if not cfg.resume:
        raise ValueError("Please provide --resume checkpoint path.")

    load_checkpoint(
        model,
        cfg.resume,
        optimizer=None,
        strict=False,
        map_location=str(device),
        load_optimizer=False,
    )
    model.eval()

    detail_rows = []
    saved_samples = 0

    for sample_idx, batch in enumerate(test_loader):
        if saved_samples >= cfg.max_samples:
            break
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        records, aux = instrumented_forward(model, lr_hsi, hr_msi)
        masks = region_masks(gt, aux["shadow_risk"], cfg.shadow_percentile)

        for checkpoint_idx, (name, pred) in enumerate(records):
            for region_name, mask in masks.items():
                metrics = calc_region_metrics(pred, gt, mask, cfg.scale_ratio)
                if metrics is None:
                    continue
                row = {
                    "sample_idx": sample_idx,
                    "checkpoint_idx": checkpoint_idx,
                    "checkpoint_name": name,
                    "region": region_name,
                    **metrics,
                }
                detail_rows.append(row)

        saved_samples += 1

    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[(row["checkpoint_idx"], row["checkpoint_name"], row["region"])].append(row)

    summary_rows = []
    for (checkpoint_idx, name, region), rows in sorted(grouped.items(), key=lambda x: (x[0][2], x[0][0])):
        avg = average_rows(rows)
        summary_rows.append({
            "checkpoint_idx": checkpoint_idx,
            "checkpoint_name": name,
            "region": region,
            **avg,
        })

    # 计算相邻模块变化量，便于快速判断哪个模块使 PSNR/SAM 变好或变差。
    prev_by_region = {}
    for row in sorted(summary_rows, key=lambda r: (r["region"], r["checkpoint_idx"])):
        key = row["region"]
        prev = prev_by_region.get(key)
        if prev is None:
            row["delta_PSNR_from_prev"] = ""
            row["delta_SAM_from_prev"] = ""
        else:
            row["delta_PSNR_from_prev"] = row["PSNR"] - prev["PSNR"]
            row["delta_SAM_from_prev"] = row["SAM"] - prev["SAM"]
        prev_by_region[key] = row

    if cfg.save_csv:
        detail_path = cfg.save_csv
    else:
        out_dir = os.path.join(cfg.output_root, "module_diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        detail_path = os.path.join(
            out_dir,
            f"{cfg.dataset}_{checkpoint_name(cfg.resume)}_module_contributions.csv",
        )
    os.makedirs(os.path.dirname(detail_path) or ".", exist_ok=True)
    summary_path = detail_path.replace(".csv", "_summary.csv")

    detail_fields = [
        "sample_idx", "checkpoint_idx", "checkpoint_name", "region", "pixel_count",
        "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC",
    ]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({key: row.get(key, "") for key in detail_fields})

    summary_fields = [
        "checkpoint_idx", "checkpoint_name", "region", "pixel_count",
        "PSNR", "delta_PSNR_from_prev", "RMSE", "SAM", "delta_SAM_from_prev", "ERGAS", "SSIM", "CC",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in summary_fields})

    print("=" * 100)
    print("Module Contribution Diagnostics")
    print("=" * 100)
    print(f"Saved detail CSV : {detail_path}")
    print(f"Saved summary CSV: {summary_path}")
    print("-" * 100)
    print("Key regions: all / gt_norm_shadow / model_risk_shadow / mask_overlap")
    print("delta_SAM_from_prev < 0 表示当前模块使 SAM 下降，delta_PSNR_from_prev > 0 表示当前模块使 PSNR 上升。")
    print("-" * 100)

    key_regions = {"all", "gt_norm_shadow", "model_risk_shadow", "mask_overlap"}
    for region in ["all", "gt_norm_shadow", "model_risk_shadow", "mask_overlap"]:
        print(f"\n[{region}]")
        rows = [r for r in summary_rows if r["region"] == region]
        for row in sorted(rows, key=lambda r: r["checkpoint_idx"]):
            dpsnr = row.get("delta_PSNR_from_prev", "")
            dsam = row.get("delta_SAM_from_prev", "")
            dpsnr_s = "" if dpsnr == "" else f" {dpsnr:+.4f}"
            dsam_s = "" if dsam == "" else f" {dsam:+.4f}"
            print(
                f"{row['checkpoint_idx']:02d} {row['checkpoint_name']:<24} "
                f"PSNR={row['PSNR']:.4f}{dpsnr_s:>10}  "
                f"SAM={row['SAM']:.4f}{dsam_s:>10}"
            )

    print("=" * 100)


if __name__ == "__main__":
    main()
