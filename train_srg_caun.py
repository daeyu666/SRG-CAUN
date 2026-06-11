# train_srg_caun.py
"""Train / test entry for SRG-CAUN."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import parse_args, print_config
from data_loader import build_loaders
from losses import BaseHSRLoss, SAMLoss
from metrics import MetricAverager, calc_metrics
from models import build_srg_caun
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


class ShadowWeightedSAMLoss(nn.Module):
    """阴影/低可靠区域加权 SAM。

    模型输出 aux['shadow_risk'] 后，该损失提高阴影风险区域的 SAM 权重。
    没有 aux 时退化为普通 SAM。
    """

    def __init__(self, weight_scale: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.weight_scale = weight_scale
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, shadow_risk: torch.Tensor = None) -> torch.Tensor:
        dot = torch.sum(pred * target, dim=1)
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + self.eps)
        target_norm = torch.sqrt(torch.sum(target * target, dim=1) + self.eps)
        cos = dot / (pred_norm * target_norm + self.eps)
        cos = torch.clamp(cos, -1.0 + self.eps, 1.0 - self.eps)
        angle = torch.acos(cos)

        if shadow_risk is None:
            return angle.mean()

        weight = 1.0 + self.weight_scale * shadow_risk.squeeze(1).detach()
        return torch.mean(angle * weight)


class ReferenceDirectionLoss(nn.Module):
    """约束参考残差不要强行改变光谱幅值，只作为方向修正。"""

    def __init__(self):
        super().__init__()

    def forward(self, model) -> torch.Tensor:
        aux = getattr(model, "latest_aux", {})
        stage_infos = aux.get("stage_infos", [])
        if not stage_infos:
            return next(model.parameters()).new_tensor(0.0)
        losses = []
        for info in stage_infos:
            ref = info.get("ref_residual", None)
            reliability = info.get("reliability", None)
            if ref is None or reliability is None:
                continue
            shadow = 1.0 - reliability.detach()
            # 控制参考残差幅度，避免非阴影参考值被直接复制到阴影区域。
            losses.append(torch.mean(torch.abs(ref) * shadow))
        if not losses:
            return next(model.parameters()).new_tensor(0.0)
        return sum(losses) / len(losses)


def parse_srg_args():
    cfg = parse_args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--ref_topk", type=int, default=4)
    parser.add_argument("--lambda_shadow_sam", type=float, default=0.10)
    parser.add_argument("--lambda_ref_dir", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args, _ = parser.parse_known_args()
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def build_criterion(cfg, info):
    srf_weights = info.get("srf_weights", None)
    base = BaseHSRLoss(
        scale_ratio=cfg.scale_ratio,
        n_select_bands=info["n_select_bands"],
        lambda_l1=cfg.lambda_l1,
        lambda_sam=cfg.lambda_sam,
        lambda_dc=cfg.lambda_dc,
        lambda_sgrad=cfg.lambda_sgrad,
        lambda_sdir=cfg.lambda_sdir,
        lambda_ns_l1=cfg.lambda_ns_l1,
        lambda_srf_region=cfg.lambda_srf_region,
        srf_weights=srf_weights,
    )
    shadow_sam = ShadowWeightedSAMLoss(weight_scale=2.0)
    ref_dir = ReferenceDirectionLoss()
    return base, shadow_sam, ref_dir


def train_one_epoch(model, loader, optimizer, base_loss, shadow_sam_loss, ref_dir_loss, cfg, device):
    model.train()
    meter = MetricAverager()
    total_loss = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        optimizer.zero_grad(set_to_none=True)
        pred, aux = model(lr_hsi, hr_msi, return_aux=True)
        loss, loss_dict = base_loss(pred, gt, lr_hsi, hr_msi)

        loss_shadow = shadow_sam_loss(pred, gt, aux.get("shadow_risk", None))
        loss_ref = ref_dir_loss(model)
        loss = loss + cfg.lambda_shadow_sam * loss_shadow + cfg.lambda_ref_dir * loss_ref

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        metrics = {k: float(v.item()) for k, v in loss_dict.items()}
        metrics["shadow_sam"] = float(loss_shadow.detach().item())
        metrics["ref_dir"] = float(loss_ref.detach().item())
        meter.update(metrics)

    avg = meter.average()
    avg["train_loss_total"] = total_loss / max(len(loader), 1)
    return avg


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    meter = MetricAverager()
    for batch in loader:
        batch = move_to_device(batch, device)
        pred = model(batch["lr_hsi"], batch["hr_msi"])
        pred = torch.clamp(pred, 0.0, 1.0)
        metrics = calc_metrics(pred, batch["gt"], cfg.scale_ratio)
        meter.update(metrics)
    return meter.average()


def main():
    cfg = parse_srg_args()
    print_config(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_loader, test_loader, info = build_loaders(cfg)
    model = build_srg_caun(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_select_bands"],
        scale_ratio=cfg.scale_ratio,
        hidden_dim=cfg.hidden_dim,
        num_stages=cfg.num_stages,
        ref_topk=cfg.ref_topk,
        srf_weights=info.get("srf_weights", None),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    base_loss, shadow_sam_loss, ref_dir_loss = build_criterion(cfg, info)
    base_loss = base_loss.to(device)
    shadow_sam_loss = shadow_sam_loss.to(device)
    ref_dir_loss = ref_dir_loss.to(device)

    save_name = cfg.save_name or f"{cfg.dataset}_srg_caun.pth"
    ckpt_dir = os.path.join(cfg.checkpoint_root, "srg_caun")
    ckpt_path = os.path.join(ckpt_dir, save_name)
    best_path = os.path.join(ckpt_dir, save_name.replace(".pth", "_best.pth"))
    log_path = os.path.join(cfg.log_root, f"{cfg.dataset}_srg_caun.log")
    csv_path = os.path.join(cfg.log_root, f"{cfg.dataset}_srg_caun.csv")

    start_epoch = 1
    best_psnr = -1.0
    if cfg.resume:
        start_epoch, best_psnr = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=False,
            map_location=str(device),
            load_optimizer=True,
        )
        start_epoch += 1

    write_log(log_path, f"Model parameters: {count_parameters(model):.3f} M")
    write_log(log_path, f"Dataset info: {info}")

    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "train_loss_total", "loss", "l1", "sam", "dc", "sgrad", "sdir", "ns_l1", "srf_region",
            "shadow_sam", "ref_dir", "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC", "best_psnr",
        ],
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, base_loss, shadow_sam_loss, ref_dir_loss, cfg, device
        )

        eval_metrics = {}
        if epoch % cfg.eval_interval == 0:
            eval_metrics = evaluate(model, test_loader, cfg, device)
            psnr = eval_metrics.get("PSNR", -1.0)
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(model, optimizer, epoch, best_psnr, best_path, extra={"cfg": cfg.__dict__, "info": info})

        if epoch % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, best_psnr, ckpt_path, extra={"cfg": cfg.__dict__, "info": info})

        msg = f"Epoch {epoch:04d}/{cfg.epochs} | train_loss={train_metrics['train_loss_total']:.6f}"
        if eval_metrics:
            msg += f" | PSNR={eval_metrics['PSNR']:.4f} SAM={eval_metrics['SAM']:.4f} best={best_psnr:.4f}"
        write_log(log_path, msg)

        row = {"epoch": epoch, **train_metrics, **eval_metrics, "best_psnr": best_psnr}
        csv_logger.write(row)

    save_checkpoint(model, optimizer, cfg.epochs, best_psnr, ckpt_path, extra={"cfg": cfg.__dict__, "info": info})
    write_log(log_path, f"Training finished. Best PSNR={best_psnr:.4f}. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
