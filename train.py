"""
Distributed Training Script — Rotation-Equivariant Rectified Flow
===================================================================

A production-grade DDP training loop with:
  • Multi-node, multi-GPU support via ``torch.distributed``
  • Weights & Biases logging (rank-0 only)
  • Gradient clipping, mixed-precision (AMP), learning-rate warmup + cosine decay
  • Periodic checkpointing with automatic resume
  • Validation loss tracking for early stopping / model selection
  • Training-time visualization every --vis_every epochs

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

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from models.vn_transformer import FlowTransformer
from core.flow_matcher import RectifiedFlowMatcher
from core.dataset import build_dataloaders
from core.point_ops import farthest_point_sample, fps_gather


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Rotation-Equivariant Rectified Flow")

    # --- Data ---
    p.add_argument("--data_root", type=str, default="data/human",
                   help="Path to dataset root containing train/val/test folders.")
    p.add_argument("--n_points", type=int, default=2048,
                   help="Template size and inference resolution (e.g. 2048).")
    p.add_argument("--train_n_points", type=int, default=2048,
                   help="FPS subsample size during training. Must be <= n_points. "
                        "The dataset loads full resolution and FPS happens in flow_matcher.")
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
    p.add_argument("--lambda_reg", type=float, default=0.001,
                   help="Template regularisation weight.")
    p.add_argument("--lambda_reg_decay", type=float, default=0.995,
                   help="Per-epoch decay factor for lambda_reg.")
    p.add_argument("--template_reg_radius", type=float, default=1.5,
                   help="Target norm for template regularisation.")
    p.add_argument("--lambda_chamfer", type=float, default=0.1,
                   help="Chamfer distance loss weight.")
    p.add_argument("--lambda_repulsion", type=float, default=0.01,
                   help="Repulsion loss weight.")
    p.add_argument("--sinkhorn_iters", type=int, default=50,
                   help="Sinkhorn iterations for assignment.")
    p.add_argument("--sinkhorn_reg", type=float, default=0.01,
                   help="Sinkhorn entropic regularisation (lower = sharper).")
    p.add_argument("--use_hard_assignment", action="store_true", default=True,
                   help="Use hard 1-to-1 assignment after Sinkhorn (default: True).")
    p.add_argument("--no_hard_assignment", action="store_true",
                   help="Disable hard assignment (use soft Sinkhorn permutation).")

    # --- Training ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--template_init", type=str, default=None,
                   help="Path to .npy mean shape for template init (overrides Fibonacci sphere).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use automatic mixed precision.")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP.")
    p.add_argument("--grad_ckpt", action="store_true", default=False,
                   help="Enable gradient checkpointing to reduce VRAM (~30%% slower, ~4x less memory).")

    # --- Visualization ---
    p.add_argument("--vis_every", type=int, default=50,
                   help="Visualize training progress every N epochs.")
    p.add_argument("--vis_dir", type=str, default=None,
                   help="Directory to save visualization PNGs (default: ckpt_dir/vis).")

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
    if args.no_hard_assignment:
        args.use_hard_assignment = False
    if args.vis_dir is None:
        args.vis_dir = os.path.join(args.ckpt_dir, "vis")
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
# Training-time visualization
# ---------------------------------------------------------------------------

def visualize_training_progress(
    flow_matcher: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    vis_dir: str,
    n_samples: int = 2,
    n_steps: int = 10,
    wandb_run=None,
):
    """
    Visualize flow progress: Template → Flowed → Target.

    Picks n_samples from the validation set, runs a quick Euler integration,
    and renders a 3-panel matplotlib figure saved as PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    os.makedirs(vis_dir, exist_ok=True)

    # Get the raw model (unwrap DDP)
    raw_fm = flow_matcher.module if hasattr(flow_matcher, "module") else flow_matcher
    raw_model = raw_fm.model
    raw_fm.eval()

    # Grab n_samples from validation set
    vis_targets = []
    for batch in val_loader:
        batch = batch.to(device)
        for i in range(batch.shape[0]):
            vis_targets.append(batch[i:i+1])
            if len(vis_targets) >= n_samples:
                break
        if len(vis_targets) >= n_samples:
            break

    template = raw_model.get_template(1).detach().cpu().squeeze(0).numpy()  # [N, 3]

    for idx, target_batch in enumerate(vis_targets):
        # FPS target to training resolution
        N = target_batch.shape[1]
        n_train = raw_fm.train_n_points
        if n_train > 0 and n_train < N:
            fps_idx = farthest_point_sample(target_batch, n_train)
            target_sub = fps_gather(target_batch, fps_idx)
        else:
            target_sub = target_batch

        with torch.no_grad():
            trajectory = raw_fm.sample(
                target_sub,
                n_steps=n_steps,
                method="euler",
            )
        flowed = trajectory[-1, 0].cpu().numpy()                  # [N, 3]
        target_np = target_sub[0].cpu().numpy()                   # [N, 3]

        # Render 3-panel figure
        fig = plt.figure(figsize=(18, 6))

        for panel_idx, (data, title) in enumerate([
            (template, "Template (Canonical)"),
            (flowed, f"Flowed (Epoch {epoch})"),
            (target_np, "Target"),
        ]):
            ax = fig.add_subplot(1, 3, panel_idx + 1, projection="3d")
            # Color by height (z-coordinate) for visual structure
            colors = data[:, 1]  # use y-coordinate for coloring
            ax.scatter(
                data[:, 0], data[:, 1], data[:, 2],
                c=colors, cmap="viridis", s=1.0, alpha=0.7,
            )
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-1.2, 1.2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        fig.suptitle(f"Training Progress — Epoch {epoch}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        save_path = os.path.join(vis_dir, f"epoch_{epoch:04d}_sample_{idx}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Vis] Saved → {save_path}")

        # Log to W&B
        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log(
                    {f"vis/sample_{idx}": wandb.Image(save_path)},
                    step=epoch,
                )
            except Exception:
                pass

    raw_fm.train()


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
    loss_keys = ["loss", "loss_velocity", "loss_ot", "loss_reg",
                 "loss_chamfer", "loss_repulsion"]
    running = {k: 0.0 for k in loss_keys}
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
    loss_keys = ["loss", "loss_velocity", "loss_ot", "loss_reg",
                 "loss_chamfer", "loss_repulsion"]
    running = {k: 0.0 for k in loss_keys}
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
    # Dataset loads full 15k from disk and FPS-subsamples to n_points (2048)
    # on CPU during __getitem__.  The GPU only ever sees [B, 2048, 3].
    loaders = build_dataloaders(
        data_root=args.data_root,
        n_points=args.n_points,  # FPS to 2048 in dataset
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        local_scratch=args.local_scratch,
        distributed=(world_size > 1),
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    if is_main_process(rank):
        sample = next(iter(train_loader))
        print(f"  Data: FPS in dataset → batch shape = {list(sample.shape)}")

    # ---- Build model ----
    model = FlowTransformer(
        n_points=args.n_points,  # Template size = 2048
        channels=args.channels,
        n_heads=args.n_heads,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        latent_dim=args.latent_dim,
        time_dim=args.time_dim,
        knn_k=args.knn_k,
        use_checkpoint=args.grad_ckpt,
    ).to(device)

    # ---- Load mean shape template (if provided) ----
    if args.template_init:
        mean_shape = np.load(args.template_init).astype(np.float32)
        assert mean_shape.shape == (args.n_points, 3), (
            f"template_init shape {mean_shape.shape} != expected ({args.n_points}, 3)"
        )
        with torch.no_grad():
            model.template.copy_(torch.from_numpy(mean_shape).unsqueeze(0))
        if is_main_process(rank):
            print(f"  Template initialised from {args.template_init}")

    if is_main_process(rank):
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Model parameters: {n_params:.2f}M")
        print(f"  Template:  {args.n_points} pts  |  "
              f"Train subsample: {args.train_n_points}  |  "
              f"KNN-k: {args.knn_k or 'global'}")
        print(f"  Sinkhorn: reg={args.sinkhorn_reg}, iters={args.sinkhorn_iters}, "
              f"hard={args.use_hard_assignment}")
        print(f"  Losses: λ_ot={args.lambda_ot}, λ_reg={args.lambda_reg}, "
              f"λ_chamfer={args.lambda_chamfer}, λ_repulsion={args.lambda_repulsion}")

    # ---- Flow Matcher (training wrapper) ----
    flow_matcher = RectifiedFlowMatcher(
        model=model,
        lambda_ot=args.lambda_ot,
        lambda_reg=args.lambda_reg,
        lambda_chamfer=args.lambda_chamfer,
        lambda_repulsion=args.lambda_repulsion,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_reg=args.sinkhorn_reg,
        train_n_points=args.train_n_points,
        use_hard_assignment=args.use_hard_assignment,
        template_reg_radius=args.template_reg_radius,
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

    # Current lambda_reg (will decay over training)
    current_lambda_reg = args.lambda_reg

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

        # Decay lambda_reg
        if epoch > 0:
            current_lambda_reg *= args.lambda_reg_decay
            raw_fm = flow_matcher.module if hasattr(flow_matcher, "module") else flow_matcher
            raw_fm.lambda_reg = current_lambda_reg

        # ---- Train ----
        train_metrics = train_one_epoch(
            flow_matcher, train_loader, optimiser, scaler, device, epoch, args, rank,
        )

        epoch_time = time.time() - t0

        # ---- Validate ----
        val_metrics = None
        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            val_metrics = validate(flow_matcher, val_loader, device, args)

        # ---- Visualization (rank-0) ----
        if is_main_process(rank) and val_loader is not None and (epoch + 1) % args.vis_every == 0:
            visualize_training_progress(
                flow_matcher, val_loader, device, epoch,
                vis_dir=args.vis_dir,
                wandb_run=wandb_run,
            )

        # ---- Logging (rank-0) ----
        if is_main_process(rank):
            log = {
                "epoch": epoch,
                "lr": lr,
                "lambda_reg": current_lambda_reg,
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
                f"CD: {train_metrics['loss_chamfer']:.5f} | "
                f"Rep: {train_metrics['loss_repulsion']:.5f} | "
                f"Reg: {train_metrics['loss_reg']:.5f} | "
                f"LR: {lr:.2e} | "
                f"λ_reg: {current_lambda_reg:.2e} | "
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
