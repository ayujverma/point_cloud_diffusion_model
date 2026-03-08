"""
Correspondence Visualisation — Template Flow Alignment
=======================================================

Renders animated GIFs showing the canonical template "flowing" into
different target shapes while maintaining **consistent point colours**.

The same colour assigned to template point i appears on the corresponding
region of every target shape.  If point #417 is coloured red on the template,
it should land on the "left elbow" of every human shape, regardless of pose.

Outputs for each of 10 samples
-------------------------------
1. **Per-target GIF**: animated flow from template → target, points keep
   their colour throughout, showing the deformation in real time.
2. **Before/After comparison image**: grey unaligned target next to the
   colour-coded flowed result.
3. **Summary grid**: all 10 flowed shapes side by side with the *same*
   colour map — makes correspondence consistency visually obvious.

Usage
-----
::

    python visualize.py \\
        --checkpoint checkpoints/best.pt \\
        --data_root data/human \\
        --n_targets 10 \\
        --n_steps 30 \\
        --knn_k 32 \\
        --output_dir visualisations
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.vn_transformer import FlowTransformer
from core.flow_matcher import RectifiedFlowMatcher
from core.dataset import PointCloudDataset


# ---------------------------------------------------------------------------
# Colour assignment: consistent per-template-point
# ---------------------------------------------------------------------------

def assign_template_colours(template: np.ndarray) -> np.ndarray:
    """
    Assign a unique colour to each template point based on its spatial
    position (using the signed distance along a canonical axis + height).

    Parameters
    ----------
    template : [N, 3]

    Returns
    -------
    colours : [N, 3]  RGB in [0, 1]
    """
    score = template[:, 1] * 0.6 + template[:, 0] * 0.3 + template[:, 2] * 0.1
    norm = Normalize(vmin=score.min(), vmax=score.max())
    cmap = cm.get_cmap("rainbow")
    colours = cmap(norm(score))[:, :3]  # drop alpha
    return colours


# ---------------------------------------------------------------------------
# Rendering utilities
# ---------------------------------------------------------------------------

def render_pointcloud(
    ax: plt.Axes,
    points: np.ndarray,
    colours: np.ndarray,
    title: str = "",
    elev: float = 20,
    azim: float = 45,
    point_size: float = 1.5,
):
    """Plot a 3D point cloud on a matplotlib Axes3D."""
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colours, s=point_size, alpha=0.8,
    )
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_axis_off()


def save_flow_gif(
    trajectory: np.ndarray,
    colours: np.ndarray,
    target: np.ndarray,
    out_path: str,
    fps: int = 10,
):
    """
    Save the flow trajectory as an animated GIF.
    Each frame shows the flowing points (coloured) alongside the grey target.

    trajectory : [n_steps, N, 3]
    colours    : [N, 3]
    target     : [M, 3]  — original unaligned target
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [Warning] Pillow not installed — skipping GIF generation.")
        return

    frames = []
    n_steps = trajectory.shape[0]
    grey = np.full((target.shape[0], 3), 0.6)

    for i in range(n_steps):
        fig = plt.figure(figsize=(10, 5))
        t_frac = i / max(n_steps - 1, 1)

        # Left: grey target (static)
        ax1 = fig.add_subplot(121, projection="3d")
        render_pointcloud(ax1, target, grey, title="Target (unaligned)")

        # Right: flowing template (coloured)
        ax2 = fig.add_subplot(122, projection="3d")
        render_pointcloud(ax2, trajectory[i], colours,
                          title=f"Template → Target  (t={t_frac:.2f})")

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(Image.fromarray(img))
        plt.close(fig)

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=1000 // fps, loop=0,
    )
    print(f"  Saved GIF → {out_path}")


def save_before_after(
    target: np.ndarray,
    flowed: np.ndarray,
    colours: np.ndarray,
    out_path: str,
    sample_idx: int,
):
    """
    Side-by-side: grey unaligned target vs colour-coded flowed result.
    """
    fig = plt.figure(figsize=(10, 5))
    grey = np.full((target.shape[0], 3), 0.6)

    ax1 = fig.add_subplot(121, projection="3d")
    render_pointcloud(ax1, target, grey,
                      title=f"Sample {sample_idx}: BEFORE (unaligned)")

    ax2 = fig.add_subplot(122, projection="3d")
    render_pointcloud(ax2, flowed, colours,
                      title=f"Sample {sample_idx}: AFTER (aligned)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved before/after → {out_path}")


def save_correspondence_grid(
    template: np.ndarray,
    flowed: list[np.ndarray],
    colours: np.ndarray,
    out_path: str,
):
    """
    Summary grid: template + all flowed shapes with the SAME colour map.
    All shapes use identical point-index colouring, so consistent regions
    prove the correspondence is working.

    template : [N, 3]
    flowed   : list of [N, 3]
    colours  : [N, 3]
    """
    n = len(flowed)
    cols = min(n + 1, 6)
    rows = (n + 1 + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    # First panel: template
    ax = fig.add_subplot(rows, cols, 1, projection="3d")
    render_pointcloud(ax, template, colours, title="Template", point_size=1.0)

    # Remaining panels: each flowed shape
    for i, fl in enumerate(flowed):
        ax = fig.add_subplot(rows, cols, i + 2, projection="3d")
        render_pointcloud(ax, fl, colours, title=f"Shape {i+1}", point_size=1.0)

    fig.suptitle(
        "Correspondence Consistency — Same colour = same body part across all shapes",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved correspondence grid → {out_path}")


def save_before_after_grid(
    targets: list[np.ndarray],
    flowed: list[np.ndarray],
    colours: np.ndarray,
    out_path: str,
):
    """
    Two-row grid: top row = grey unaligned targets, bottom row = coloured
    flowed results.  Makes the alignment improvement visually obvious.
    """
    n = len(targets)
    fig = plt.figure(figsize=(4 * n, 8))
    grey_base = np.array([0.6, 0.6, 0.6])

    for i in range(n):
        # Top row: unaligned targets (grey)
        ax = fig.add_subplot(2, n, i + 1, projection="3d")
        grey = np.broadcast_to(grey_base, (targets[i].shape[0], 3)).copy()
        render_pointcloud(ax, targets[i], grey,
                          title=f"Target {i+1} (unaligned)", point_size=0.8)

        # Bottom row: flowed (coloured)
        ax = fig.add_subplot(2, n, n + i + 1, projection="3d")
        render_pointcloud(ax, flowed[i], colours,
                          title=f"Shape {i+1} (aligned)", point_size=0.8)

    fig.suptitle("BEFORE (grey, unaligned) vs AFTER (coloured, aligned)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved before/after grid → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise flow correspondence")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, default="data/human",
                        help="Dataset root with test/ folder")
    parser.add_argument("--n_targets", type=int, default=10,
                        help="Number of target shapes to visualise")
    parser.add_argument("--n_steps", type=int, default=30,
                        help="ODE integration steps")
    parser.add_argument("--n_points", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="visualisations")
    parser.add_argument("--device", type=str, default="cuda")

    # Model architecture (must match checkpoint)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--enc_depth", type=int, default=6)
    parser.add_argument("--dec_depth", type=int, default=6)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--knn_k", type=int, default=32,
                        help="KNN neighbours for local attention (must match training)")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    print("Loading model...")
    model = FlowTransformer(
        n_points=args.n_points,
        channels=args.channels,
        n_heads=args.n_heads,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        latent_dim=args.latent_dim,
        time_dim=args.time_dim,
        knn_k=args.knn_k,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    flow_matcher = RectifiedFlowMatcher(model=model).to(device)

    # ---- Load test data ----
    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        for fallback in ["val", "train"]:
            test_dir = os.path.join(args.data_root, fallback)
            if os.path.isdir(test_dir):
                break

    dataset = PointCloudDataset(test_dir, n_points=args.n_points, normalise=True)
    n_targets = min(args.n_targets, len(dataset))

    # Select evenly-spaced samples
    indices = np.linspace(0, len(dataset) - 1, n_targets, dtype=int)
    targets = torch.stack([dataset[i] for i in indices]).to(device)  # [K, N, 3]

    # ---- Get template + colours ----
    template = model.get_template(1).squeeze(0).detach().cpu().numpy()  # [N, 3]
    colours = assign_template_colours(template)

    # ---- Integrate flow for each target ----
    print(f"Integrating flow for {n_targets} targets ({args.n_steps} steps)...")
    all_flowed = []
    all_trajectories = []

    for i in range(n_targets):
        target_i = targets[i : i + 1]                            # [1, N, 3]
        traj = flow_matcher.sample(target_i, n_steps=args.n_steps, method="midpoint")
        traj_np = traj[:, 0].cpu().numpy()                       # [S+1, N, 3]
        all_trajectories.append(traj_np)
        all_flowed.append(traj_np[-1])                           # final frame

    raw_targets = [targets[i].cpu().numpy() for i in range(n_targets)]

    # ---- Save per-sample GIFs (flow animation with grey target alongside) ----
    print("Rendering GIFs...")
    for i, traj_np in enumerate(all_trajectories):
        gif_path = os.path.join(args.output_dir, f"flow_target_{i:02d}.gif")
        save_flow_gif(traj_np, colours, raw_targets[i], gif_path)

    # ---- Save per-sample before/after images ----
    print("Rendering before/after comparisons...")
    for i in range(n_targets):
        ba_path = os.path.join(args.output_dir, f"before_after_{i:02d}.png")
        save_before_after(raw_targets[i], all_flowed[i], colours, ba_path, i)

    # ---- Save correspondence grid (all shapes with same colours) ----
    print("Rendering correspondence grid...")
    grid_path = os.path.join(args.output_dir, "correspondence_grid.png")
    save_correspondence_grid(template, all_flowed, colours, grid_path)

    # ---- Save before/after grid ----
    print("Rendering before/after grid...")
    ba_grid_path = os.path.join(args.output_dir, "before_after_grid.png")
    save_before_after_grid(raw_targets, all_flowed, colours, ba_grid_path)

    print(f"\nDone!  All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
