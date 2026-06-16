# train_drt_baseline.py
"""Train / test entry for the DRT baseline under the SRG-CAUN pipeline."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import parse_args, print_config
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from models import build_drt_baseline
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


class DRTPaperLoss(nn.Module):
    """DRT-Net paper loss without the removed contrastive term.

    Paper total loss: Ltotal = Lmse + Lspe + Lc.
    This baseline removes contrastive learning, so Lc is fixed to 0 and the
    optimized reconstruction loss is Lmse + Lspe.
    """

    def __init__(self, lambda_mse: float = 1.0, lambda_spe: float = 1.0):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_spe = lambda_spe

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, model: nn.Module):
        loss_mse = 0.5 * F.mse_loss(pred, gt)
        pred_spe = model.spectral_reconstruction(pred)
        loss_spe = 0.5 * F.mse_loss(pred_spe, pred)
        loss_contrast = pred.new_tensor(0.0)
        loss = self.lambda_mse * loss_mse + self.lambda_spe * loss_spe + loss_contrast
        return loss, {
            "loss": loss.detach(),
            "mse": loss_mse.detach(),
            "spe": loss_spe.detach(),
            "contrast": loss_contrast.detach(),
        }


def parse_drt_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda_spe", type=float, default=1.0)
    parser.add_argument("--score_sam_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args, remaining = parser.parse_known_args()

    cfg = parse_args(remaining)
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def build_criterion(cfg):
    return DRTPaperLoss(
        lambda_mse=float(getattr(cfg, "lambda_mse", 1.0)),
        lambda_spe=float(getattr(cfg, "lambda_spe", 1.0)),
    )


def calc_checkpoint_score(eval_metrics, cfg):
    psnr = float(eval_metrics.get("PSNR", -1.0))
    sam = float(eval_metrics.get("SAM", 1e6))
    return psnr - float(cfg.score_sam_weight) * sam


def train_one_epoch(model, loader, optimizer, criterion, cfg, device, epoch: int):
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
        loss, loss_dict = criterion(pred, gt, model)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        meter.update({k: float(v.item()) for k, v in loss_dict.items()})

    avg = meter.average()
    avg["train_loss_total"] = total_loss / max(len(loader), 1)
    avg["w_mse"] = float(getattr(cfg, "lambda_mse", 1.0))
    avg["w_spe"] = float(getattr(cfg, "lambda_spe", 1.0))
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
    cfg = parse_drt_args()
    print_config(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_loader, test_loader, info = build_loaders(cfg)
    model = build_drt_baseline(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_select_bands"],
        scale_ratio=cfg.scale_ratio,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dropout=cfg.dropout,
        srf_weights=info.get("srf_weights", None),
    ).to(device)

    criterion = build_criterion(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    save_name = cfg.save_name or f"{cfg.dataset}_drt_baseline.pth"
    ckpt_dir = os.path.join(cfg.checkpoint_root, "drt_baseline")
    ckpt_path = os.path.join(ckpt_dir, save_name)
    best_path = os.path.join(ckpt_dir, save_name.replace(".pth", "_best.pth"))
    log_path = os.path.join(cfg.log_root, f"{cfg.dataset}_drt_baseline.log")
    csv_path = os.path.join(cfg.log_root, f"{cfg.dataset}_drt_baseline.csv")

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

    write_log(log_path, f"Model parameters: {count_parameters(model):.3f} M")
    write_log(log_path, f"Dataset info: {info}")
    write_log(log_path, "Model: DRT baseline without rectangular transformer, SAFA multiscale aggregation, or contrastive learning.")
    write_log(log_path, "Loss: DRT paper loss Ltotal = Lmse + Lspe + Lc, with Lc=0 because contrastive learning is removed.")
    write_log(log_path, f"Loss weights: lambda_mse={getattr(cfg, 'lambda_mse', 1.0):.3f}, lambda_spe={cfg.lambda_spe:.3f}")
    write_log(log_path, f"Checkpoint score: PSNR - {cfg.score_sam_weight:.3f} * SAM")

    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "train_loss_total", "loss", "mse", "spe", "contrast", "w_mse", "w_spe",
            "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC",
            "checkpoint_score", "best_score", "best_psnr", "best_sam",
        ],
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, cfg, device, epoch)

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
                        "loss_formula": "Lmse + Lspe + Lc, Lc=0 for no-contrast baseline",
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
                    "loss_formula": "Lmse + Lspe + Lc, Lc=0 for no-contrast baseline",
                },
            )

        msg = f"Epoch {epoch:04d}/{cfg.epochs} | train_loss={train_metrics['train_loss_total']:.6f}"
        msg += (
            f" | mse={train_metrics.get('mse', 0.0):.6f}, "
            f"spe={train_metrics.get('spe', 0.0):.6f}, "
            f"contrast={train_metrics.get('contrast', 0.0):.6f}"
        )
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
            "loss_formula": "Lmse + Lspe + Lc, Lc=0 for no-contrast baseline",
        },
    )
    write_log(
        log_path,
        f"Training finished. Best score={best_score:.4f}, best PSNR={best_psnr:.4f}, "
        f"best SAM={best_sam:.4f}. Best checkpoint: {best_path}",
    )


if __name__ == "__main__":
    main()
