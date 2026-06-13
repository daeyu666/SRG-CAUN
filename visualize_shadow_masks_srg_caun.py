# visualize_shadow_masks_srg_caun.py
"""Visualize two shadow mask generation methods for SRG-CAUN.

输出三张图并排对比：
1. 原图伪彩色图；
2. gt_norm 方式生成的最暗百分比阴影掩膜；
3. model_risk_percentile 方式生成的模型风险最高百分比阴影掩膜。

默认保存到 outputs/shadow_vis。
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

# 必须在导入 pyplot 之前设置无界面后端，避免服务器/SSH 环境触发 Qt xcb 报错。
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

from config import parse_args, print_config
from data_loader import build_loaders
from models import build_srg_caun
from utils import get_device, load_checkpoint, move_to_device, set_seed


def parse_vis_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--ref_topk", type=int, default=4)
    parser.add_argument("--shadow_percentile", type=float, default=20.0)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--rgb_bands", type=int, nargs=3, default=None,
                        help="Optional RGB band indices, for example --rgb_bands 60 30 10")
    args, remaining = parser.parse_known_args()

    cfg = parse_args(remaining)
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def normalize_2d(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin()
    x_max = x.amax()
    return (x - x_min) / (x_max - x_min).clamp_min(1e-8)


def make_rgb(gt: torch.Tensor, rgb_bands=None) -> torch.Tensor:
    """gt: C,H,W -> H,W,3."""
    c, h, w = gt.shape
    if rgb_bands is None:
        # 近似伪彩色：高波段作 R，中波段作 G，低波段作 B。
        r = int(round(0.65 * (c - 1)))
        g = int(round(0.35 * (c - 1)))
        b = int(round(0.10 * (c - 1)))
        bands = [r, g, b]
    else:
        bands = [max(0, min(c - 1, int(i))) for i in rgb_bands]

    rgb = torch.stack([normalize_2d(gt[i]) for i in bands], dim=-1)
    return torch.clamp(rgb, 0.0, 1.0)


def make_gt_norm_shadow_mask(gt: torch.Tensor, percentile: float) -> torch.Tensor:
    """gt: C,H,W -> H,W bool. 根据 GT 光谱模长最低 percentile% 生成阴影区域。"""
    norm = torch.sqrt(torch.sum(gt * gt, dim=0) + 1e-8)
    q = torch.quantile(norm.flatten(), percentile / 100.0)
    return norm <= q


def make_model_risk_percentile_mask(shadow_risk: torch.Tensor, percentile: float) -> torch.Tensor:
    """shadow_risk: 1,H,W 或 H,W -> H,W bool. 取 shadow_risk 最高 percentile% 区域。"""
    if shadow_risk.ndim == 3:
        shadow_risk = shadow_risk.squeeze(0)
    q = torch.quantile(shadow_risk.flatten(), 1.0 - percentile / 100.0)
    return shadow_risk >= q


def mask_to_rgb(mask: torch.Tensor) -> torch.Tensor:
    """H,W bool -> H,W,3, 白色为阴影，黑色为非阴影。"""
    m = mask.float()
    return torch.stack([m, m, m], dim=-1)


def overlay_mask(rgb: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    """红色半透明叠加，便于看阴影落在哪些地物上。"""
    out = rgb.clone()
    red = torch.zeros_like(out)
    red[..., 0] = 1.0
    out[mask] = (1.0 - alpha) * out[mask] + alpha * red[mask]
    return torch.clamp(out, 0.0, 1.0)


def calc_overlap(gt_mask: torch.Tensor, risk_mask: torch.Tensor) -> Tuple[int, int, int, float, float]:
    inter = torch.logical_and(gt_mask, risk_mask).sum().item()
    union = torch.logical_or(gt_mask, risk_mask).sum().item()
    gt_count = gt_mask.sum().item()
    risk_count = risk_mask.sum().item()
    iou = inter / max(union, 1)
    dice = 2.0 * inter / max(gt_count + risk_count, 1)
    return int(inter), int(union), int(gt_count), float(iou), float(dice)


@torch.no_grad()
def main():
    cfg = parse_vis_args()
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

    save_dir = cfg.save_dir or os.path.join(cfg.output_root, "shadow_vis")
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    for sample_idx, batch in enumerate(test_loader):
        if saved >= cfg.max_samples:
            break

        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        _, aux = model(lr_hsi, hr_msi, return_aux=True)

        # 当前脚本建议 batch_size=1。若 batch>1，逐个样本保存。
        bsz = gt.shape[0]
        for bi in range(bsz):
            if saved >= cfg.max_samples:
                break

            gt_i = gt[bi].detach().cpu()
            risk_i = aux["shadow_risk"][bi].detach().cpu()

            rgb = make_rgb(gt_i, cfg.rgb_bands)
            gt_mask = make_gt_norm_shadow_mask(gt_i, cfg.shadow_percentile)
            risk_mask = make_model_risk_percentile_mask(risk_i, cfg.shadow_percentile)

            inter, union, gt_count, iou, dice = calc_overlap(gt_mask, risk_mask)
            risk_count = int(risk_mask.sum().item())

            gt_overlay = overlay_mask(rgb, gt_mask)
            risk_overlay = overlay_mask(rgb, risk_mask)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
            axes[0].imshow(rgb.numpy())
            axes[0].set_title("Original pseudo-RGB")

            axes[1].imshow(gt_overlay.numpy())
            axes[1].set_title(f"GT-norm lowest {cfg.shadow_percentile:.0f}%\nshadow pixels={gt_count}")

            axes[2].imshow(risk_overlay.numpy())
            axes[2].set_title(f"Model-risk top {cfg.shadow_percentile:.0f}%\nshadow pixels={risk_count}")

            for ax in axes:
                ax.axis("off")

            fig.suptitle(f"sample={sample_idx}, IoU={iou:.4f}, Dice={dice:.4f}, intersection={inter}, union={union}")
            fig.tight_layout()

            save_path = os.path.join(save_dir, f"{cfg.dataset}_sample{sample_idx:03d}_item{bi}_shadow_compare.png")
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)

            print(f"Saved: {save_path}")
            print(f"  gt_pixels={gt_count}, model_pixels={risk_count}, intersection={inter}, union={union}, IoU={iou:.4f}, Dice={dice:.4f}")
            saved += 1

    print(f"Done. Saved {saved} visualizations to: {save_dir}")


if __name__ == "__main__":
    main()
