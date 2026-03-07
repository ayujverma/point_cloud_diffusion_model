"""
Distributed Training Script — Rotation-Equivariant Rectified Flow
===================================================================

A production-grade DDP training loop with:
  • Multi-node, multi-GPU support via ``torch.distributed``
  • Weights & Biases logging (rank-0 only)
  • Gradient clipping, mixed-precision (AMP), learning-rate warmup + cosine decay
  • Periodic checkpointing with automatic resume
  • Validation loss tracking for early stopping / model selection

Usage (single-node, single-GPU — for debugging)::

    python train.py --data_root data/human --epochs 300

Usage (DDP, launched by torchrun or Slurm)::

    torchrun --nproc_per_node=3 --nnodes=8 \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        --node_rank=$NODE_RANK \\
        train.py --data_root data/human --epochs 300

See ``scripts/run_tacc.sh`` for a ready-made Slurm sbatch script.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from models.vn_transformer import FlowTransformer
from core.flow_matcher import RectifiedFlowMatcher
from core.dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Rotation-Equivariant Rectified Flow")

    # --- Data ---
    p.add_argument("--data_root", type=str, default="data/human",
                   help="Path to dataset root containing train/val/test folders.")
    p.add_argument("--n_points", type=int, default=15000,
                   help="Number of points loaded per cloud from disk (full resolution).")
    p.add_argument("--train_n_points", type=int, default=2048,
                   help="FPS subsample size during training. Set to 0 to disable "
                        "subsampling (use all n_points). E.g. 2048 or 4096.")
    p.add_argument("--local_scratch", type=str, default=None,
                   help="Local scratch dir for fast I/O (e.g. /tmp/human).")

    # --- Model ---
    p.add_argument("--channels", type=int, default=128,
                   help="VN channel width.")
    p.add_argument("--n_heads", type=int, default=8,
                   help="Number of attention heads.")
    p.add_argument("--enc_depth", type=int, default=6,
                   help="Encoder transformer depth.")
    p.add_argument("--dec_depth", type=int, default=6,
                   help="Decoder transformer depth.")
    p.add_argument("--latent_dim", type=int, default=256,
                   help="Shape latent dimension.")
    p.add_argument("--time_dim", type=int, default=128,
                   help="Time embedding dimension.")
    p.add_argument("--knn_k", type=int, default=32,
                   help="KNN neighbours for local attention. 0 = global attention.")

    # --- Flow / OT ---
    p.add_argument("--lambda_ot", type=float, default=0.1,
                   help="Sinkhorn divergence loss weight.")
    p.add_argument("--lambda_reg", type=float, default=0.01,
                   help="Template regularisation weight.")
    p.add_argument("--sinkhorn_iters", type=int, default=20,
                   help="Sinkhorn iterations for assignment.")
    p.add_argument("--sinkhorn_reg", type=float, default=0.05,
                   help="Sinkhorn entropic regularisation.")

    # --- Training ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use automatic mixed precision.")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP.")
    p.add_argument("--grad_ckpt", action="store_true", default=False,
                   help="Enable gradient checkpointing to reduce VRAM (~30%% slower, ~4x less memory).")

    # --- Logging / Checkpointing ---
    p.add_argument("--wandb_project", type=str, default="rectified-flow-pc",
                   help="W&B project name.")
    p.add_argument("--wandb_entity", type=str, default="dense-3d-point-correspondences")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging entirely (useful for smoke tests).")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="Checkpoint directory.")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--val_every", type=int, default=5,
                   help="Validate every N epochs.")

    args = p.parse_args()
    if args.no_amp:
        args.amp = False
    return args


# ---------------------------------------------------------------------------
# DDP setup / teardown
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """
    Initialise torch.distributed.  Works with both ``torchrun`` and manual
    environment variables (Slurm).

    Returns (rank, local_rank, world_size).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Single-GPU fallback
        rank, local_rank, world_size = 0, 0, 1

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(epoch: int, warmup: int, total: int, base_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    # Cosine decay after warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    best_val_loss: float,
):
    """Save training state to disk (rank-0 only)."""
    state = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint and return (start_epoch, best_val_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    print(f"  [Checkpoint] Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"] + 1, ckpt.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    flow_matcher: RectifiedFlowMatcher,
    loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, float]:
    """Run one training epoch.  Returns dict of average losses."""
    flow_matcher.train()
    running = {"loss": 0.0, "loss_velocity": 0.0, "loss_ot": 0.0, "loss_reg": 0.0}
    n_batches = 0

    for batch_idx, points in enumerate(loader):
        points = points.to(device, non_blocking=True)            # [B, N, 3]

        optimiser.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=args.amp):
            losses = flow_matcher(points)

        scaler.scale(losses["loss"]).backward()

        # Gradient clipping (unscale first for correct norm)
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), args.grad_clip)

        scaler.step(optimiser)
        scaler.update()

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

        # Log every 50 batches
        if is_main_process(rank) and (batch_idx + 1) % 50 == 0:
            avg_loss = running["loss"] / n_batches
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {avg_loss:.5f}")

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    flow_matcher: RectifiedFlowMatcher,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run validation.  Returns dict of average losses."""
    flow_matcher.eval()
    running = {"loss": 0.0, "loss_velocity": 0.0, "loss_ot": 0.0, "loss_reg": 0.0}
    n_batches = 0

    for points in loader:
        points = points.to(device, non_blocking=True)
        with autocast("cuda", enabled=args.amp):
            losses = flow_matcher(points)
        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print("=" * 70)
        print("  Rotation-Equivariant Rectified Flow — Training")
        print(f"  World size: {world_size}  |  Device: {device}")
        print("=" * 70)

    # ---- W&B init (rank-0 only) ----
    wandb_run = None
    if is_main_process(rank) and not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config=vars(args),
            )
        except ImportError:
            print("  [Warning] wandb not installed — logging disabled.")
        except Exception as e:
            print(f"  [Warning] wandb init failed ({e}) — logging disabled.")

    # ---- Build data loaders ----
    loaders = build_dataloaders(
        data_root=args.data_root,
        n_points=args.n_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        local_scratch=args.local_scratch,
        distributed=(world_size > 1),
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    # ---- Build model ----
    model = FlowTransformer(
        n_points=args.n_points,
        channels=args.channels,
        n_heads=args.n_heads,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        latent_dim=args.latent_dim,
        time_dim=args.time_dim,
        knn_k=args.knn_k,
        use_checkpoint=args.grad_ckpt,
    ).to(device)

    if is_main_process(rank):
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model parameters: {n_params:.2f}M")
        print(f"  Template:  {args.n_points} pts (full)  |  "
              f"Train subsample: {args.train_n_points or 'disabled'}  |  "
              f"KNN-k: {args.knn_k or 'global'}")

    # Wrap in DDP if multi-GPU
    # (Moved to wrap flow_matcher instead, since flow_matcher calls non-forward model methods)

    # ---- Flow Matcher (training wrapper) ----
    flow_matcher = RectifiedFlowMatcher(
        model=model,
        lambda_ot=args.lambda_ot,
        lambda_reg=args.lambda_reg,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_reg=args.sinkhorn_reg,
        train_n_points=args.train_n_points,
    ).to(device)

    if world_size > 1:
        flow_matcher = DDP(flow_matcher, device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=False)

    # ---- Optimiser ----
    optimiser = torch.optim.AdamW(
        flow_matcher.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_val_loss = float("inf")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimiser, scaler, device,
        )

    # ================================================================
    #  Training loop
    # ================================================================
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Set epoch for DistributedSampler reproducibility
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Adjust learning rate
        lr = get_lr(epoch, args.warmup_epochs, args.epochs, args.lr)
        for pg in optimiser.param_groups:
            pg["lr"] = lr

        # ---- Train ----
        train_metrics = train_one_epoch(
            flow_matcher, train_loader, optimiser, scaler, device, epoch, args, rank,
        )

        epoch_time = time.time() - t0

        # ---- Validate ----
        val_metrics = None
        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            val_metrics = validate(flow_matcher, val_loader, device, args)

        # ---- Logging (rank-0) ----
        if is_main_process(rank):
            log = {
                "epoch": epoch,
                "lr": lr,
                "time_s": epoch_time,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }
            if val_metrics:
                log.update({f"val/{k}": v for k, v in val_metrics.items()})

            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {train_metrics['loss']:.5f} | "
                f"Vel: {train_metrics['loss_velocity']:.5f} | "
                f"OT: {train_metrics['loss_ot']:.5f} | "
                f"Reg: {train_metrics['loss_reg']:.5f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            if val_metrics:
                print(
                    f"          Val Loss: {val_metrics['loss']:.5f} | "
                    f"Vel: {val_metrics['loss_velocity']:.5f}"
                )

            if wandb_run:
                wandb_run.log(log, step=epoch)

            # ---- Checkpointing ----
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    str(ckpt_dir / f"epoch_{epoch:04d}.pt"),
                    epoch, model, optimiser, scaler, best_val_loss,
                )

            # Save best model
            if val_metrics and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    str(ckpt_dir / "best.pt"),
                    epoch, model, optimiser, scaler, best_val_loss,
                )
                print(f"  ★ New best val loss: {best_val_loss:.5f}")

    # ---- Final save ----
    if is_main_process(rank):
        save_checkpoint(
            str(ckpt_dir / "final.pt"),
            args.epochs - 1, model, optimiser, scaler, best_val_loss,
        )
        print("\nTraining complete.")
        if wandb_run:
            wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
