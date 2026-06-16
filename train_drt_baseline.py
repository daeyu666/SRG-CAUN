# train_drt_baseline.py
"""Train / test entry for the DRT baseline under the SRG-CAUN pipeline."""

from __future__ import annotations

import argparse
import os

import torch
from torch.optim import AdamW

from config import parse_args, print_config
from data_loader import build_loaders
from losses import BaseHSRLoss
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


def parse_drt_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--score_sam_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_schedule", type=str, default="off", choices=["off", "warmup"])
    parser.add_argument("--phase1_epochs", type=int, default=50)
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


def get_loss_weights(epoch: int, cfg) -> dict:
    if getattr(cfg, "loss_schedule", "off") == "warmup" and epoch <= cfg.phase1_epochs:
        return {
            "l1": 1.0,
            "sam": 0.05,
            "dc": 0.10,
            "sgrad": 0.0,
            "sdir": 0.0,
            "ns_l1": 0.30,
            "srf_region": 0.0,
        }

    return {
        "l1": cfg.lambda_l1,
        "sam": cfg.lambda_sam,
        "dc": cfg.lambda_dc,
        "sgrad": cfg.lambda_sgrad,
        "sdir": cfg.lambda_sdir,
        "ns_l1": cfg.lambda_ns_l1,
        "srf_region": cfg.lambda_srf_region,
    }


def apply_base_loss_weights(base_loss: BaseHSRLoss, weights: dict):
    base_loss.lambda_l1 = weights["l1"]
    base_loss.lambda_sam = weights["sam"]
    base_loss.lambda_dc = weights["dc"]
    base_loss.lambda_sgrad = weights["sgrad"]
    base_loss.lambda_sdir = weights["sdir"]
    base_loss.lambda_ns_l1 = weights["ns_l1"]
    base_loss.lambda_srf_region = weights["srf_region"]


def calc_checkpoint_score(eval_metrics, cfg):
    psnr = float(eval_metrics.get("PSNR", -1.0))
    sam = float(eval_metrics.get("SAM", 1e6))
    return psnr - float(cfg.score_sam_weight) * sam


def train_one_epoch(model, loader, optimizer, base_loss, cfg, device, epoch: int):
    model.train()
    meter = MetricAverager()
    total_loss = 0.0

    loss_weights = get_loss_weights(epoch, cfg)
    apply_base_loss_weights(base_loss, loss_weights)

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        optimizer.zero_grad(set_to_none=True)
        pred = model(lr_hsi, hr_msi)
        loss, loss_dict = base_loss(pred, gt, lr_hsi, hr_msi)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        meter.update({k: float(v.item()) for k, v in loss_dict.items()})

    avg = meter.average()
    avg["train_loss_total"] = total_loss / max(len(loader), 1)
    for key, value in loss_weights.items():
        avg[f"w_{key}"] = value
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

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    base_loss = build_criterion(cfg, info).to(device)

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
    write_log(log_path, f"Checkpoint score: PSNR - {cfg.score_sam_weight:.3f} * SAM")
    write_log(log_path, f"Loss schedule: {cfg.loss_schedule}, phase1={cfg.phase1_epochs}")

    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "train_loss_total", "loss", "l1", "sam", "dc", "sgrad", "sdir", "ns_l1", "srf_region",
            "w_l1", "w_sam", "w_dc", "w_sgrad", "w_sdir", "w_ns_l1", "w_srf_region",
            "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC",
            "checkpoint_score", "best_score", "best_psnr", "best_sam",
        ],
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, base_loss, cfg, device, epoch)

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
                        "loss_weights": get_loss_weights(epoch, cfg),
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
                    "loss_weights": get_loss_weights(epoch, cfg),
                },
            )

        msg = f"Epoch {epoch:04d}/{cfg.epochs} | train_loss={train_metrics['train_loss_total']:.6f}"
        msg += (
            f" | w: L1={train_metrics['w_l1']:.2f}, SAM={train_metrics['w_sam']:.2f}, "
            f"DC={train_metrics['w_dc']:.2f}, NS={train_metrics['w_ns_l1']:.2f}"
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
            "loss_weights": get_loss_weights(cfg.epochs, cfg),
        },
    )
    write_log(
        log_path,
        f"Training finished. Best score={best_score:.4f}, best PSNR={best_psnr:.4f}, "
        f"best SAM={best_sam:.4f}. Best checkpoint: {best_path}",
    )


if __name__ == "__main__":
    main()
