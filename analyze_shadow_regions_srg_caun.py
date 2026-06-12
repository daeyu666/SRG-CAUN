# analyze_shadow_regions_srg_caun.py
"""Analyze shadow / non-shadow reconstruction quality for SRG-CAUN.

该脚本用于判断模型是否真正降低了阴影区域 SAM。
它支持两种阴影划分方式：
1. gt_norm：根据 GT-HSI 光谱模长的低分位生成阴影/低反射率掩膜；
2. model_risk：根据模型输出的 shadow_risk 生成阴影风险掩膜。

建议优先看 gt_norm，因为它不依赖模型自身预测的可靠性图，更适合做客观对比。
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from config import parse_args, print_config
from data_loader import build_loaders
from metrics import calc_metrics
from models import build_srg_caun
from utils import get_device, load_checkpoint, move_to_device, set_seed


def parse_shadow_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--ref_topk", type=int, default=4)
    parser.add_argument("--mask_mode", type=str, default="gt_norm", choices=["gt_norm", "model_risk"])
    parser.add_argument("--shadow_percentile", type=float, default=20.0)
    parser.add_argument("--risk_threshold", type=float, default=0.50)
    parser.add_argument("--save_csv", type=str, default="")
    args, remaining = parser.parse_known_args()

    cfg = parse_args(remaining)
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def masked_tensor(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """B,C,H,W + B,1,H,W -> B,C,N，供指标函数按空间点计算。"""
    b, c, h, w = x.shape
    mask = mask.bool().expand(-1, c, -1, -1)
    values = x[mask].view(b, c, -1)
    return values.unsqueeze(-1)


def calc_masked_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, scale_ratio: int) -> Dict[str, float]:
    count = int(mask.sum().item())
    if count <= 0:
        return {
            "pixel_count": 0,
            "PSNR": float("nan"),
            "RMSE": float("nan"),
            "SAM": float("nan"),
            "ERGAS": float("nan"),
            "SSIM": float("nan"),
            "CC": float("nan"),
        }
    pred_m = masked_tensor(pred, mask)
    gt_m = masked_tensor(gt, mask)
    metrics = calc_metrics(pred_m, gt_m, scale_ratio)
    metrics["pixel_count"] = count
    return metrics


def make_gt_norm_shadow_mask(gt: torch.Tensor, percentile: float) -> torch.Tensor:
    """根据 GT 光谱模长低分位划分阴影/低反射率区域。"""
    norm = torch.sqrt(torch.sum(gt * gt, dim=1, keepdim=True) + 1e-8)
    flat = norm.flatten()
    q = torch.quantile(flat, percentile / 100.0)
    return norm <= q


def make_model_risk_shadow_mask(shadow_risk: torch.Tensor, threshold: float) -> torch.Tensor:
    return shadow_risk >= threshold


def average_metric_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for key in keys:
        values = [r[key] for r in rows if isinstance(r[key], (int, float)) and r[key] == r[key]]
        if not values:
            out[key] = float("nan")
        elif key == "pixel_count":
            out[key] = float(sum(values))
        else:
            out[key] = float(sum(values) / len(values))
    return out


@torch.no_grad()
def main():
    cfg = parse_shadow_args()
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

    all_rows = []
    region_buckets = {"all": [], "shadow": [], "non_shadow": []}

    for sample_idx, batch in enumerate(test_loader):
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        pred, aux = model(lr_hsi, hr_msi, return_aux=True)
        pred = torch.clamp(pred, 0.0, 1.0)

        if cfg.mask_mode == "gt_norm":
            shadow_mask = make_gt_norm_shadow_mask(gt, cfg.shadow_percentile)
        else:
            shadow_mask = make_model_risk_shadow_mask(aux["shadow_risk"], cfg.risk_threshold)

        non_shadow_mask = ~shadow_mask
        all_mask = torch.ones_like(shadow_mask, dtype=torch.bool)

        masks = {
            "all": all_mask,
            "shadow": shadow_mask,
            "non_shadow": non_shadow_mask,
        }

        for region_name, mask in masks.items():
            metrics = calc_masked_metrics(pred, gt, mask, cfg.scale_ratio)
            metrics.update({
                "sample_idx": sample_idx,
                "region": region_name,
                "mask_mode": cfg.mask_mode,
                "shadow_percentile": cfg.shadow_percentile,
                "risk_threshold": cfg.risk_threshold,
            })
            all_rows.append(metrics)
            region_buckets[region_name].append(metrics)

    print("=" * 80)
    print("Shadow Region Analysis")
    print("=" * 80)
    print(f"Dataset: {cfg.dataset}")
    print(f"Checkpoint: {cfg.resume}")
    print(f"Mask mode: {cfg.mask_mode}")
    if cfg.mask_mode == "gt_norm":
        print(f"Shadow mask: lowest {cfg.shadow_percentile:.1f}% GT spectral norm pixels")
    else:
        print(f"Shadow mask: model shadow_risk >= {cfg.risk_threshold:.3f}")
    print("-" * 80)

    summary_rows = []
    for region_name in ["all", "shadow", "non_shadow"]:
        avg = average_metric_dicts(region_buckets[region_name])
        summary_rows.append({"region": region_name, **avg})
        print(f"Region: {region_name}")
        print(f"  Pixels: {avg['pixel_count']:.0f}")
        print(f"  PSNR  : {avg['PSNR']:.4f}")
        print(f"  RMSE  : {avg['RMSE']:.6f}")
        print(f"  SAM   : {avg['SAM']:.4f}")
        print(f"  ERGAS : {avg['ERGAS']:.4f}")
        print(f"  SSIM  : {avg['SSIM']:.4f}")
        print(f"  CC    : {avg['CC']:.4f}")
        print("-" * 80)

    if cfg.save_csv:
        save_path = cfg.save_csv
    else:
        save_dir = os.path.join(cfg.output_root, "metrics")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = os.path.splitext(os.path.basename(cfg.resume))[0]
        save_path = os.path.join(save_dir, f"{cfg.dataset}_{ckpt_name}_shadow_regions.csv")

    fieldnames = [
        "sample_idx", "region", "mask_mode", "shadow_percentile", "risk_threshold",
        "pixel_count", "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary_path = save_path.replace(".csv", "_summary.csv")
    summary_fieldnames = ["region", "pixel_count", "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in summary_fieldnames})

    print(f"Saved detail CSV: {save_path}")
    print(f"Saved summary CSV: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
