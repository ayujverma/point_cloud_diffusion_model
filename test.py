"""
Full-Resolution Inference Script — Flow All 15k Points
=======================================================

This script loads a trained checkpoint and flows the **full-resolution**
canonical template (e.g. 15 000 points) through the learned velocity field
to reach each target shape.  The resulting flowed point clouds form the
**aligned test set**: point index k in every output refers to the same
semantic body part, because all outputs originate from the same template.

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
- ``{output_dir}/flowed_{idx:04d}.npy``  — aligned point cloud   [N_template, 3]
- ``{output_dir}/target_{idx:04d}.npy``  — original target        [M, 3]
- ``{output_dir}/template.npy``          — canonical template      [N_template, 3]
- ``{output_dir}/summary.txt``           — per-shape Chamfer distances (before & after)
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
    template = model.get_template(1).squeeze(0)                  # [N_tmpl, 3]
    template_np = template.detach().cpu().numpy()
    np.save(os.path.join(args.output_dir, "template.npy"), template_np)
    print(f"  Template saved: {template_np.shape}")

    # ---- Process each test shape ----
    chamfer_before = []   # template (no flow) vs target
    chamfer_after = []    # flowed template vs target
    total_time = 0.0

    for idx in range(n_samples):
        target_full = dataset[idx].to(device)                    # [M, 3]
        M = target_full.shape[0]

        print(f"\n  [{idx+1}/{n_samples}] Target: {M} points")

        # "Before" metric: Chamfer distance from raw template to target
        cd_before = chamfer_distance(template, target_full).item()
        chamfer_before.append(cd_before)

        # Encode needs [B, M, 3]
        target_batch = target_full.unsqueeze(0)                  # [1, M, 3]

        t0 = time.time()

        with torch.no_grad():
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

        # "After" metric: Chamfer distance from flowed template to target
        cd_after = chamfer_distance(flowed, target_full).item()
        chamfer_after.append(cd_after)

        improvement = (1 - cd_after / max(cd_before, 1e-8)) * 100
        print(f"    CD before: {cd_before:.6f}  |  CD after: {cd_after:.6f}  "
              f"|  Improvement: {improvement:.1f}%  |  Time: {elapsed:.2f}s")

        # Save results
        flowed_np = flowed.cpu().numpy()
        target_np = target_full.cpu().numpy()
        np.save(os.path.join(args.output_dir, f"flowed_{idx:04d}.npy"), flowed_np)
        np.save(os.path.join(args.output_dir, f"target_{idx:04d}.npy"), target_np)

    # ---- Summary ----
    mean_before = np.mean(chamfer_before) if chamfer_before else 0.0
    std_before = np.std(chamfer_before) if chamfer_before else 0.0
    mean_after = np.mean(chamfer_after) if chamfer_after else 0.0
    std_after = np.std(chamfer_after) if chamfer_after else 0.0
    mean_improvement = (1 - mean_after / max(mean_before, 1e-8)) * 100

    summary = (
        f"Full-Resolution Inference Summary\n"
        f"{'=' * 50}\n"
        f"Checkpoint:       {args.checkpoint}\n"
        f"Split:            {args.split}\n"
        f"Num shapes:       {n_samples}\n"
        f"Template pts:     {args.n_points}\n"
        f"KNN-k:            {args.knn_k}\n"
        f"Steps:            {args.n_steps}\n"
        f"Method:           {args.method}\n"
        f"{'=' * 50}\n"
        f"Chamfer BEFORE (template vs target):\n"
        f"  Mean:  {mean_before:.6f} +/- {std_before:.6f}\n"
        f"Chamfer AFTER  (flowed vs target):\n"
        f"  Mean:  {mean_after:.6f} +/- {std_after:.6f}\n"
        f"Improvement:      {mean_improvement:.1f}%\n"
        f"{'=' * 50}\n"
        f"Total time:       {total_time:.1f}s\n"
        f"Avg time/shape:   {total_time / max(n_samples, 1):.2f}s\n"
    )
    print(f"\n{summary}")

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)
        f.write("\nPer-shape Chamfer distances (before → after):\n")
        for i in range(len(chamfer_after)):
            imp = (1 - chamfer_after[i] / max(chamfer_before[i], 1e-8)) * 100
            f.write(f"  {i:4d}: {chamfer_before[i]:.6f} → {chamfer_after[i]:.6f}  "
                    f"({imp:+.1f}%)\n")

    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
