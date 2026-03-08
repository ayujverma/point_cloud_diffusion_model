"""
Latent Tour Visualisation — Template Flow Correspondence
=========================================================

Renders the canonical template "flowing" into different target shapes
while maintaining consistent point colours to **prove** dense correspondence.

The same colour assigned to template point i appears on the corresponding
region of every target shape.  If point #417 is coloured red on the template,
it should land on the "left elbow" of every human shape, regardless of pose.

Usage
-----
::

    python scripts/visualize_latent_tour.py \\
        --checkpoint checkpoints/best.pt \\
        --data_root data/human \\
        --n_targets 6 \\
        --n_steps 30 \\
        --output_dir visualisations

Outputs
-------
- Per-target GIF showing the flow from template → target.
- A grid image showing template + all final correspondences side by side.
- Optionally, an interactive Open3D window.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    # Colour by a combination of height (y) and lateral position (x)
    # to get a visually distinctive per-point colour.
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
    out_path: str,
    fps: int = 10,
):
    """
    Save the flow trajectory as an animated GIF.

    trajectory : [n_steps, N, 3]
    colours    : [N, 3]
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [Warning] Pillow not installed — skipping GIF generation.")
        return

    frames = []
    n_steps = trajectory.shape[0]

    for i in range(n_steps):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        t_frac = i / max(n_steps - 1, 1)
        render_pointcloud(ax, trajectory[i], colours, title=f"t = {t_frac:.2f}")
        fig.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(Image.fromarray(img))
        plt.close(fig)

    # Save GIF
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"  Saved GIF → {out_path}")


def save_correspondence_grid(
    template: np.ndarray,
    targets: list[np.ndarray],
    flowed: list[np.ndarray],
    colours: np.ndarray,
    out_path: str,
):
    """
    Save a grid image: top row = template + targets (raw),
    bottom row = template + flowed results with correspondence colours.

    template : [N, 3]
    targets  : list of [N, 3]  (raw target shapes)
    flowed   : list of [N, 3]  (template flowed to each target)
    colours  : [N, 3]
    """
    n = len(targets)
    fig = plt.figure(figsize=(4 * (n + 1), 8))

    # Row 1: template + raw targets (neutral grey)
    ax = fig.add_subplot(2, n + 1, 1, projection="3d")
    render_pointcloud(ax, template, colours, title="Template")

    for i, tgt in enumerate(targets):
        ax = fig.add_subplot(2, n + 1, i + 2, projection="3d")
        grey = np.full_like(colours, 0.6)
        render_pointcloud(ax, tgt, grey, title=f"Target {i+1}")

    # Row 2: template + flowed shapes with correspondence colours
    ax = fig.add_subplot(2, n + 1, n + 2, projection="3d")
    render_pointcloud(ax, template, colours, title="Template (ref)")

    for i, fl in enumerate(flowed):
        ax = fig.add_subplot(2, n + 1, n + 3 + i, projection="3d")
        render_pointcloud(ax, fl, colours, title=f"Flowed → {i+1}")

    fig.suptitle("Dense Correspondence via Rectified Flow", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise flow correspondence")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, default="data/human",
                        help="Dataset root with test/ folder")
    parser.add_argument("--n_targets", type=int, default=6,
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
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    flow_matcher = RectifiedFlowMatcher(model=model).to(device)

    # ---- Load test data ----
    test_dir = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_dir):
        # Fall back to val or train
        for fallback in ["val", "train"]:
            test_dir = os.path.join(args.data_root, fallback)
            if os.path.isdir(test_dir):
                break

    dataset = PointCloudDataset(test_dir, n_points=args.n_points, normalise=True)
    n_targets = min(args.n_targets, len(dataset))

    # Select evenly-spaced samples from the dataset
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
        # traj: [S+1, 1, N, 3]
        traj_np = traj[:, 0].cpu().numpy()                       # [S+1, N, 3]
        all_trajectories.append(traj_np)
        all_flowed.append(traj_np[-1])                           # final frame

    # ---- Save GIFs ----
    print("Rendering GIFs...")
    for i, traj_np in enumerate(all_trajectories):
        gif_path = os.path.join(args.output_dir, f"flow_target_{i:02d}.gif")
        save_flow_gif(traj_np, colours, gif_path)

    # ---- Save correspondence grid ----
    print("Rendering correspondence grid...")
    raw_targets = [targets[i].cpu().numpy() for i in range(n_targets)]
    grid_path = os.path.join(args.output_dir, "correspondence_grid.png")
    save_correspondence_grid(template, raw_targets, all_flowed, colours, grid_path)

    print(f"\nDone!  All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
