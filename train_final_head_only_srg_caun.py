# train_final_head_only_srg_caun.py
"""Train a final-head-only SRG-CAUN baseline.

目的：验证当前 SRG-CAUN 的最终指标到底有多少来自最后的 final_head。

模型结构：
    LR-HSI -> bicubic upsample -> residual final_head -> HR-HSI

也就是说，该脚本不使用 InitialReconstruction、物理一致性展开、非阴影参考库、
Contourlet 频率先验和多阶段 refine，只保留与 SRG-CAUN 中 final_head 等价的残差 CNN 头。

输出：
    checkpoints/final_head_only/*
    logs/*_final_head_only.log
    logs/*_final_head_only.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import parse_args, print_config
from data_loader import build_loaders
from losses import BaseHSRLoss
from metrics import MetricAverager, calc_metrics
from utils import (
    CSVLogger,
    count_parameters,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


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


class FinalHeadOnlyModel(nn.Module):
    """Only the residual final_head, using bicubic LR-HSI upsampling as input."""

    def __init__(self, n_bands: int, hidden_dim: int = 48):
        super().__init__()
        self.final_head = nn.Sequential(
            nn.Conv2d(n_bands, hidden_dim, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, n_bands, 3, padding=1),
        )

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, return_aux: bool = False):
        lr_up = F.interpolate(
            lr_hsi,
            size=hr_msi.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        pred = torch.clamp(lr_up + self.final_head(lr_up), 0.0, 1.0)
        if return_aux:
            return pred, {"lr_up": lr_up.detach()}
        return pred


def parse_final_head_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--score_sam_weight", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args, remaining = parser.parse_known_args()

    cfg = parse_args(remaining)
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def build_criterion(cfg, info):
    return BaseHSRLoss(
        scale_ratio=cfg.scale_ratio,
        n_select_bands=info["n_select_bands"],
        lambda_l1=cfg.lambda_l1,
        lambda_sam=cfg.lambda_sam,
        lambda_dc=cfg.lambda_dc,
        lambda_sgrad=cfg.lambda_sgrad,
        lambda_sdir=cfg.lambda_sdir,
        lambda_ns_l1=cfg.lambda_ns_l1,
        lambda_srf_region=cfg.lambda_srf_region,
        srf_weights=info.get("srf_weights", None),
    )


def calc_checkpoint_score(eval_metrics: Dict[str, float], cfg) -> float:
    return float(eval_metrics.get("PSNR", -1.0)) - float(cfg.score_sam_weight) * float(eval_metrics.get("SAM", 1e6))


@torch.no_grad()
def evaluate_input_lr_up(loader, cfg, device):
    meter = MetricAverager()
    for batch in loader:
        batch = move_to_device(batch, device)
        lr_up = F.interpolate(
            batch["lr_hsi"],
            size=batch["gt"].shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        meter.update(calc_metrics(torch.clamp(lr_up, 0.0, 1.0), batch["gt"], cfg.scale_ratio))
    return meter.average()


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    meter = MetricAverager()
    for batch in loader:
        batch = move_to_device(batch, device)
        pred = model(batch["lr_hsi"], batch["hr_msi"])
        pred = torch.clamp(pred, 0.0, 1.0)
        meter.update(calc_metrics(pred, batch["gt"], cfg.scale_ratio))
    return meter.average()


def train_one_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    meter = MetricAverager()
    total_loss = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        optimizer.zero_grad(set_to_none=True)
        pred = model(lr_hsi, hr_msi)
        loss, loss_dict = criterion(pred, gt, lr_hsi, hr_msi)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        meter.update({k: float(v.item()) for k, v in loss_dict.items()})

    avg = meter.average()
    avg["train_loss_total"] = total_loss / max(len(loader), 1)
    return avg


def main():
    cfg = parse_final_head_args()
    print_config(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_loader, test_loader, info = build_loaders(cfg)
    model = FinalHeadOnlyModel(
        n_bands=info["n_bands"],
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = build_criterion(cfg, info).to(device)

    save_name = cfg.save_name or f"{cfg.dataset}_final_head_only.pth"
    ckpt_dir = os.path.join(cfg.checkpoint_root, "final_head_only")
    ckpt_path = os.path.join(ckpt_dir, save_name)
    best_path = os.path.join(ckpt_dir, save_name.replace(".pth", "_best.pth"))
    log_path = os.path.join(cfg.log_root, f"{cfg.dataset}_final_head_only.log")
    csv_path = os.path.join(cfg.log_root, f"{cfg.dataset}_final_head_only.csv")

    write_log(log_path, f"Model parameters: {count_parameters(model):.3f} M")
    write_log(log_path, f"Dataset info: {info}")
    write_log(log_path, "Model: lr_hsi bicubic upsample + final_head residual CNN only")
    write_log(log_path, f"Checkpoint score: PSNR - {cfg.score_sam_weight:.3f} * SAM")

    baseline_metrics = evaluate_input_lr_up(test_loader, cfg, device)
    write_log(
        log_path,
        "LR-up baseline | "
        f"PSNR={baseline_metrics['PSNR']:.4f} RMSE={baseline_metrics['RMSE']:.6f} "
        f"SAM={baseline_metrics['SAM']:.4f} ERGAS={baseline_metrics['ERGAS']:.4f} "
        f"SSIM={baseline_metrics['SSIM']:.4f} CC={baseline_metrics['CC']:.4f}",
    )

    start_epoch = 1
    best_score = -1e9
    best_psnr = -1.0
    best_sam = 1e9
    if cfg.resume:
        start_epoch, best_score = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=False,
            map_location=str(device),
            load_optimizer=True,
        )
        start_epoch += 1

    if cfg.stage == "test":
        eval_metrics = evaluate(model, test_loader, cfg, device)
        write_log(
            log_path,
            "Test only | "
            f"PSNR={eval_metrics['PSNR']:.4f} RMSE={eval_metrics['RMSE']:.6f} "
            f"SAM={eval_metrics['SAM']:.4f} ERGAS={eval_metrics['ERGAS']:.4f} "
            f"SSIM={eval_metrics['SSIM']:.4f} CC={eval_metrics['CC']:.4f}",
        )
        return

    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "train_loss_total", "loss", "l1", "sam", "dc", "sgrad", "sdir", "ns_l1", "srf_region",
            "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC",
            "checkpoint_score", "best_score", "best_psnr", "best_sam",
        ],
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, cfg, device)

        eval_metrics = {}
        checkpoint_score = ""
        if epoch % cfg.eval_interval == 0:
            eval_metrics = evaluate(model, test_loader, cfg, device)
            checkpoint_score = calc_checkpoint_score(eval_metrics, cfg)
            psnr = eval_metrics.get("PSNR", -1.0)
            sam = eval_metrics.get("SAM", 1e9)
            if checkpoint_score > best_score:
                best_score = checkpoint_score
                best_psnr = psnr
                best_sam = sam
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_score,
                    best_path,
                    extra={
                        "cfg": cfg.__dict__,
                        "info": info,
                        "best_score": best_score,
                        "best_psnr": best_psnr,
                        "best_sam": best_sam,
                        "score_formula": f"PSNR - {cfg.score_sam_weight} * SAM",
                        "model_type": "final_head_only",
                    },
                )

        if epoch % cfg.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_score,
                ckpt_path,
                extra={
                    "cfg": cfg.__dict__,
                    "info": info,
                    "best_score": best_score,
                    "best_psnr": best_psnr,
                    "best_sam": best_sam,
                    "score_formula": f"PSNR - {cfg.score_sam_weight} * SAM",
                    "model_type": "final_head_only",
                },
            )

        msg = f"Epoch {epoch:04d}/{cfg.epochs} | train_loss={train_metrics['train_loss_total']:.6f}"
        if eval_metrics:
            msg += (
                f" | PSNR={eval_metrics['PSNR']:.4f} RMSE={eval_metrics['RMSE']:.6f} "
                f"SAM={eval_metrics['SAM']:.4f} ERGAS={eval_metrics['ERGAS']:.4f} "
                f"SSIM={eval_metrics['SSIM']:.4f} CC={eval_metrics['CC']:.4f} "
                f"score={checkpoint_score:.4f} best_score={best_score:.4f} "
                f"best_psnr={best_psnr:.4f} best_sam={best_sam:.4f}"
            )
        write_log(log_path, msg)

        row = {
            "epoch": epoch,
            **train_metrics,
            **eval_metrics,
            "checkpoint_score": checkpoint_score,
            "best_score": best_score,
            "best_psnr": best_psnr,
            "best_sam": best_sam,
        }
        csv_logger.write(row)

    save_checkpoint(
        model,
        optimizer,
        cfg.epochs,
        best_score,
        ckpt_path,
        extra={
            "cfg": cfg.__dict__,
            "info": info,
            "best_score": best_score,
            "best_psnr": best_psnr,
            "best_sam": best_sam,
            "score_formula": f"PSNR - {cfg.score_sam_weight} * SAM",
            "model_type": "final_head_only",
        },
    )
    write_log(
        log_path,
        f"Training finished. Best score={best_score:.4f}, best PSNR={best_psnr:.4f}, "
        f"best SAM={best_sam:.4f}. Best checkpoint: {best_path}",
    )


if __name__ == "__main__":
    main()
