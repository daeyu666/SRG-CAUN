# train_srg_caun.py
"""Train / test entry for SRG-CAUN."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import parse_args, print_config
from data_loader import build_loaders
from losses import BaseHSRLoss
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
            losses.append(torch.mean(torch.abs(ref) * shadow))

        if not losses:
            return next(model.parameters()).new_tensor(0.0)
        return sum(losses) / len(losses)


def parse_srg_args():
    """解析 SRG-CAUN 自有参数和模板通用参数。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--ref_topk", type=int, default=4)
    parser.add_argument("--lambda_shadow_sam", type=float, default=0.10)
    parser.add_argument("--lambda_ref_dir", type=float, default=0.02)
    parser.add_argument("--lambda_stage_loss", type=float, default=1.0,
                        help="Weight for averaged losses on every stage z_next output.")
    parser.add_argument("--score_sam_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_schedule", type=str, default="warmup", choices=["warmup", "off"])
    parser.add_argument("--phase1_epochs", type=int, default=50)
    parser.add_argument("--phase2_epochs", type=int, default=150)
    args, remaining = parser.parse_known_args()

    cfg = parse_args(remaining)
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


def get_loss_weights(epoch: int, cfg) -> dict:
    """分阶段损失权重。"""
    if getattr(cfg, "loss_schedule", "warmup") == "off":
        return {
            "l1": cfg.lambda_l1,
            "sam": cfg.lambda_sam,
            "dc": cfg.lambda_dc,
            "sgrad": cfg.lambda_sgrad,
            "sdir": cfg.lambda_sdir,
            "ns_l1": cfg.lambda_ns_l1,
            "srf_region": cfg.lambda_srf_region,
            "shadow_sam": cfg.lambda_shadow_sam,
            "ref_dir": cfg.lambda_ref_dir,
        }

    if epoch <= cfg.phase1_epochs:
        return {
            "l1": 1.0,
            "sam": 0.05,
            "dc": 0.10,
            "sgrad": 0.0,
            "sdir": 0.0,
            "ns_l1": 0.30,
            "srf_region": 0.0,
            "shadow_sam": 0.0,
            "ref_dir": 0.0,
        }

    if epoch <= cfg.phase2_epochs:
        return {
            "l1": 1.0,
            "sam": 0.05,
            "dc": 0.10,
            "sgrad": 0.02,
            "sdir": 0.05,
            "ns_l1": 0.30,
            "srf_region": 0.10,
            "shadow_sam": 0.0,
            "ref_dir": 0.0,
        }

    return {
        "l1": 1.0,
        "sam": 0.05,
        "dc": 0.10,
        "sgrad": 0.02,
        "sdir": 0.05,
        "ns_l1": 0.30,
        "srf_region": 0.10,
        "shadow_sam": 0.05,
        "ref_dir": 0.01,
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
    """Best checkpoint 采用 PSNR 与 SAM 的加权分数，不再只看 PSNR。"""
    psnr = float(eval_metrics.get("PSNR", -1.0))
    sam = float(eval_metrics.get("SAM", 1e6))
    return psnr - float(cfg.score_sam_weight) * sam


def average_tensors(values, fallback: torch.Tensor) -> torch.Tensor:
    if not values:
        return fallback
    return sum(values) / len(values)


def train_one_epoch(model, loader, optimizer, base_loss, shadow_sam_loss, ref_dir_loss, cfg, device, epoch: int):
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
        pred, aux = model(lr_hsi, hr_msi, return_aux=True)
        loss_final, loss_dict = base_loss(pred, gt, lr_hsi, hr_msi)

        zero = pred.new_tensor(0.0)
        stage_outputs = aux.get("stage_outputs", [])
        stage_base_losses = []
        stage_shadow_losses = []
        for stage_pred in stage_outputs:
            stage_loss, _ = base_loss(stage_pred, gt, lr_hsi, hr_msi)
            stage_base_losses.append(stage_loss)
            stage_shadow_losses.append(shadow_sam_loss(stage_pred, gt, aux.get("shadow_risk", None)))

        loss_stage = average_tensors(stage_base_losses, zero)
        loss_stage_shadow = average_tensors(stage_shadow_losses, zero)
        loss_shadow = shadow_sam_loss(pred, gt, aux.get("shadow_risk", None))
        loss_ref = ref_dir_loss(model)

        stage_weight = float(getattr(cfg, "lambda_stage_loss", 1.0))
        loss = (
            loss_final
            + stage_weight * loss_stage
            + loss_weights["shadow_sam"] * (loss_shadow + stage_weight * loss_stage_shadow)
            + loss_weights["ref_dir"] * loss_ref
        )

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        metrics = {k: float(v.item()) for k, v in loss_dict.items()}
        metrics["stage_loss"] = float(loss_stage.detach().item())
        metrics["stage_shadow_sam"] = float(loss_stage_shadow.detach().item())
        metrics["shadow_sam"] = float(loss_shadow.detach().item())
        metrics["ref_dir"] = float(loss_ref.detach().item())
        metrics["stage_count"] = float(len(stage_outputs))
        meter.update(metrics)

    avg = meter.average()
    avg["train_loss_total"] = total_loss / max(len(loader), 1)
    for key, value in loss_weights.items():
        avg[f"w_{key}"] = value
    avg["w_stage_loss"] = float(getattr(cfg, "lambda_stage_loss", 1.0))
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
    write_log(log_path, f"Checkpoint score: PSNR - {cfg.score_sam_weight:.3f} * SAM")
    write_log(log_path, f"Loss schedule: {cfg.loss_schedule}, phase1={cfg.phase1_epochs}, phase2={cfg.phase2_epochs}")
    write_log(log_path, f"Stage deep supervision: lambda_stage_loss={cfg.lambda_stage_loss:.3f}")
    write_log(log_path, "Initial reconstruction: LR-HSI bicubic upsample only; HR-MSI is not injected in init.")
    write_log(log_path, "Phase1 loss: L1=1.0, SAM=0.05, DC=0.10, NS_L1=0.30; others=0")
    write_log(log_path, "Phase2 loss: +SGrad=0.02, SDir=0.05, SRFRegion=0.10")
    write_log(log_path, "Phase3 loss: +ShadowSAM=0.05, RefDir=0.01")

    csv_logger = CSVLogger(
        csv_path,
        fieldnames=[
            "epoch", "train_loss_total", "loss", "stage_loss", "l1", "sam", "dc", "sgrad", "sdir", "ns_l1", "srf_region",
            "shadow_sam", "stage_shadow_sam", "ref_dir", "stage_count",
            "w_l1", "w_sam", "w_dc", "w_sgrad", "w_sdir", "w_ns_l1", "w_srf_region",
            "w_shadow_sam", "w_ref_dir", "w_stage_loss",
            "PSNR", "RMSE", "SAM", "ERGAS", "SSIM", "CC",
            "checkpoint_score", "best_score", "best_psnr", "best_sam",
        ],
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, base_loss, shadow_sam_loss, ref_dir_loss, cfg, device, epoch
        )

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
                        "lambda_stage_loss": cfg.lambda_stage_loss,
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
                    "lambda_stage_loss": cfg.lambda_stage_loss,
                },
            )

        msg = f"Epoch {epoch:04d}/{cfg.epochs} | train_loss={train_metrics['train_loss_total']:.6f}"
        msg += (
            f" | w: L1={train_metrics['w_l1']:.2f}, SAM={train_metrics['w_sam']:.2f}, "
            f"DC={train_metrics['w_dc']:.2f}, NS={train_metrics['w_ns_l1']:.2f}, "
            f"Stage={train_metrics['w_stage_loss']:.2f}, ShSAM={train_metrics['w_shadow_sam']:.2f}"
        )
        msg += f" | stage_loss={train_metrics.get('stage_loss', 0.0):.6f}"
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
            "lambda_stage_loss": cfg.lambda_stage_loss,
        },
    )
    write_log(
        log_path,
        f"Training finished. Best score={best_score:.4f}, best PSNR={best_psnr:.4f}, "
        f"best SAM={best_sam:.4f}. Best checkpoint: {best_path}",
    )


if __name__ == "__main__":
    main()
