# train_srg_caun_hier_match.py
"""Train / test entry for SRG-CAUN hierarchical coarse-to-fine matching variant."""

from __future__ import annotations

import argparse
import os

from torch.optim import AdamW

from config import parse_args, print_config
from data_loader import build_loaders
from models import build_srg_caun_hier_match
from train_srg_caun import (
    apply_base_loss_weights,
    average_tensors,
    build_criterion,
    calc_checkpoint_score,
    evaluate,
    get_loss_weights,
    train_one_epoch,
)
from utils import (
    CSVLogger,
    count_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_log,
)


def parse_hier_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--ref_topk", type=int, default=4)
    parser.add_argument("--ref_window", type=int, default=11,
                        help="Local matching window at scale=1 and scale=2.")
    parser.add_argument("--ref_fine_window", type=int, default=5,
                        help="Fine search window after coarse positions are mapped back to the current scale.")
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


def main():
    cfg = parse_hier_args()
    print_config(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    train_loader, test_loader, info = build_loaders(cfg)
    model = build_srg_caun_hier_match(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_select_bands"],
        scale_ratio=cfg.scale_ratio,
        hidden_dim=cfg.hidden_dim,
        num_stages=cfg.num_stages,
        ref_topk=cfg.ref_topk,
        ref_window=cfg.ref_window,
        ref_fine_window=cfg.ref_fine_window,
        srf_weights=info.get("srf_weights", None),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    base_loss, shadow_sam_loss, ref_dir_loss = build_criterion(cfg, info)
    base_loss = base_loss.to(device)
    shadow_sam_loss = shadow_sam_loss.to(device)
    ref_dir_loss = ref_dir_loss.to(device)

    save_name = cfg.save_name or f"{cfg.dataset}_srg_caun_hier_match.pth"
    ckpt_dir = os.path.join(cfg.checkpoint_root, "srg_caun_hier_match")
    ckpt_path = os.path.join(ckpt_dir, save_name)
    best_path = os.path.join(ckpt_dir, save_name.replace(".pth", "_best.pth"))
    log_path = os.path.join(cfg.log_root, f"{cfg.dataset}_srg_caun_hier_match.log")
    csv_path = os.path.join(cfg.log_root, f"{cfg.dataset}_srg_caun_hier_match.csv")

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
    write_log(log_path, f"Hierarchical reference matching: scale1/2 local window={cfg.ref_window}, fine window={cfg.ref_fine_window}, topk={cfg.ref_topk}")
    write_log(log_path, "Scale4 uses global non-shadow attention; scale2 refines scale4 coarse positions; scale1 refines scale2 and scale4 coarse positions.")

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
                        "variant": "srg_caun_hier_match",
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
                    "variant": "srg_caun_hier_match",
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
            "variant": "srg_caun_hier_match",
        },
    )
    write_log(
        log_path,
        f"Training finished. Best score={best_score:.4f}, best PSNR={best_psnr:.4f}, "
        f"best SAM={best_sam:.4f}. Best checkpoint: {best_path}",
    )


if __name__ == "__main__":
    main()
