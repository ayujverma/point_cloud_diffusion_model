"""
Full-Resolution Inference Script — Flow All 15k Points
=======================================================

This script loads a trained checkpoint and flows the **full-resolution**
canonical template (e.g. 15 000 points) through the learned velocity field
to reach each target shape.

Why this works without retraining
---------------------------------
The velocity network ``v_θ(x_t, t; z)`` is a *point-wise* function:
it takes each point's position x_t and produces a velocity vector.  The
only inter-point interaction is through KNN-local attention, which is
O(N·K) in memory.  Since the model was trained on 2 048-point subsamples
of the same shapes, the velocity field has learned the correct *spatial*
mapping.  At test time we simply evaluate it on a denser grid (all 15k
template points), and each point follows the learned flow to land on the
target surface.

Usage
-----
::

    python test.py \\
        --checkpoint checkpoints/best.pt \\
        --data_root data/human \\
        --n_steps 50 \\
        --output_dir results/full_res

Outputs
-------
- ``{output_dir}/flowed_{idx:04d}.npy``  — flowed point cloud  [N_template, 3]
- ``{output_dir}/target_{idx:04d}.npy``  — original target      [M, 3]
- ``{output_dir}/template.npy``          — canonical template    [N_template, 3]
- ``{output_dir}/summary.txt``           — per-shape Chamfer distance
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.vn_transformer import FlowTransformer
from core.flow_matcher import RectifiedFlowMatcher
from core.dataset import PointCloudDataset


# ---------------------------------------------------------------------------
# Chamfer distance (for evaluation)
# ---------------------------------------------------------------------------

def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Chamfer distance between two point clouds.

    a : [N, 3]
    b : [M, 3]
    returns : scalar
    """
    # a→b: for each point in a, distance to closest point in b
    diff_ab = a.unsqueeze(1) - b.unsqueeze(0)                   # [N, M, 3]
    dist_ab = (diff_ab ** 2).sum(-1)                             # [N, M]
    min_ab = dist_ab.min(dim=1).values.mean()                    # scalar

    # b→a
    min_ba = dist_ab.min(dim=0).values.mean()

    return min_ab + min_ba


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full-resolution inference: flow 15k template → target"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, default="data/human",
                        help="Dataset root with test/ folder")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to evaluate on (test, val, train)")
    parser.add_argument("--n_steps", type=int, default=50,
                        help="ODE integration steps (more = more accurate)")
    parser.add_argument("--method", type=str, default="midpoint",
                        choices=["euler", "midpoint"],
                        help="Integration method")
    parser.add_argument("--output_dir", type=str, default="results/full_res")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max number of test shapes to process (-1 = all)")

    # Model architecture (must match checkpoint)
    parser.add_argument("--n_points", type=int, default=15000,
                        help="Full template resolution (must match training)")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--enc_depth", type=int, default=6)
    parser.add_argument("--dec_depth", type=int, default=6)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--knn_k", type=int, default=32,
                        help="KNN neighbours for local attention")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Full-Resolution Inference — Rectified Flow")
    print(f"  Template points: {args.n_points}")
    print(f"  KNN-k: {args.knn_k}  |  Steps: {args.n_steps}  |  Method: {args.method}")
    print(f"  Device: {device}")
    print("=" * 70)

    # ---- Load model ----
    print("\nLoading model...")
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
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    flow_matcher = RectifiedFlowMatcher(model=model).to(device)

    # ---- Load test data (full resolution) ----
    split_dir = os.path.join(args.data_root, args.split)
    if not os.path.isdir(split_dir):
        print(f"Error: {split_dir} does not exist.")
        sys.exit(1)

    # Load at full resolution (n_points=-1 returns all raw points)
    dataset = PointCloudDataset(split_dir, n_points=-1, normalise=True)
    n_samples = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))

    # ---- Save the canonical template ----
    template_np = model.get_template(1).squeeze(0).detach().cpu().numpy()  # [N_tmpl, 3]
    np.save(os.path.join(args.output_dir, "template.npy"), template_np)
    print(f"  Template saved: {template_np.shape}")

    # ---- Process each test shape ----
    chamfer_dists = []
    total_time = 0.0

    for idx in range(n_samples):
        target_full = dataset[idx].to(device)                    # [M, 3]
        M = target_full.shape[0]

        print(f"\n  [{idx+1}/{n_samples}] Target: {M} points")

        # Encode needs [B, M, 3]
        target_batch = target_full.unsqueeze(0)                  # [1, M, 3]

        t0 = time.time()

        with torch.no_grad():
            # Flow the FULL template (all 15k points) using the learned field
            # The encoder processes the target to get z, then the decoder
            # velocity field is evaluated on each template point.
            trajectory = flow_matcher.sample(
                target_batch,
                n_steps=args.n_steps,
                method=args.method,
                use_full_template=True,
            )
            # trajectory: [n_steps+1, 1, N_tmpl, 3]
            flowed = trajectory[-1, 0]                           # [N_tmpl, 3]

        elapsed = time.time() - t0
        total_time += elapsed

        # Chamfer distance between flowed template and target
        cd = chamfer_distance(flowed, target_full).item()
        chamfer_dists.append(cd)

        print(f"    Chamfer: {cd:.6f}  |  Time: {elapsed:.2f}s")

        # Save results
        flowed_np = flowed.cpu().numpy()
        target_np = target_full.cpu().numpy()
        np.save(os.path.join(args.output_dir, f"flowed_{idx:04d}.npy"), flowed_np)
        np.save(os.path.join(args.output_dir, f"target_{idx:04d}.npy"), target_np)

    # ---- Summary ----
    mean_cd = np.mean(chamfer_dists) if chamfer_dists else 0.0
    std_cd = np.std(chamfer_dists) if chamfer_dists else 0.0

    summary = (
        f"Full-Resolution Inference Summary\n"
        f"{'=' * 40}\n"
        f"Checkpoint:    {args.checkpoint}\n"
        f"Split:         {args.split}\n"
        f"Num shapes:    {n_samples}\n"
        f"Template pts:  {args.n_points}\n"
        f"KNN-k:         {args.knn_k}\n"
        f"Steps:         {args.n_steps}\n"
        f"Method:        {args.method}\n"
        f"{'=' * 40}\n"
        f"Mean Chamfer:  {mean_cd:.6f} ± {std_cd:.6f}\n"
        f"Total time:    {total_time:.1f}s\n"
        f"Avg time/shape:{total_time / max(n_samples, 1):.2f}s\n"
    )
    print(f"\n{summary}")

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)
        f.write("\nPer-shape Chamfer distances:\n")
        for i, cd in enumerate(chamfer_dists):
            f.write(f"  {i:4d}: {cd:.6f}\n")

    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
