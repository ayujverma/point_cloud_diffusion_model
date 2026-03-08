"""
Evaluating Full-Resolution Point Correspondences
=================================================

Analyses the output of ``test.py`` with three metric families:

1. **Before / After Chamfer Distance** — How much closer does the flowed
   template get to each target compared to the raw (un-flowed) template?
2. **Correspondence Consistency** — For each template point index k
   across all aligned shapes, how tightly clustered is that point?
   Low per-point positional variance = consistent correspondence.
3. **Per-body-part breakdown** — Partition the template into rough
   spatial regions (head, torso, arms, legs) and report per-region
   consistency (useful for spotting failure modes).

Usage
-----
::

    python evaluate.py \\
        --results_dir results/full_res \\
        --output_dir results/full_res/eval

"""

import argparse
import os
import glob
import numpy as np
import torch
import json


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Bidirectional Chamfer distance between two point clouds.

    a : [N, 3]
    b : [M, 3]
    returns : scalar
    """
    diff_ab = a.unsqueeze(1) - b.unsqueeze(0)  # [N, M, 3]
    dist_ab = (diff_ab ** 2).sum(-1)           # [N, M]
    min_ab = dist_ab.min(dim=1).values.mean()
    min_ba = dist_ab.min(dim=0).values.mean()
    return (min_ab + min_ba).item()


def compute_correspondence_consistency(flowed_clouds: np.ndarray) -> dict:
    """
    Measure how consistently each point index maps to the same location
    across all aligned shapes.

    Parameters
    ----------
    flowed_clouds : [S, N, 3]  — S aligned shapes, each with N points

    Returns
    -------
    dict with:
        per_point_std     : [N]    — std of position per point index
        mean_std          : float  — average across all points
        median_std        : float
        worst_points      : top-20 highest-variance point indices
        best_points       : top-20 lowest-variance point indices
    """
    # Per-point mean across samples: [N, 3]
    mean_pos = flowed_clouds.mean(axis=0)
    # Per-point deviation from mean: [S, N, 3]
    deviations = flowed_clouds - mean_pos[None, :, :]
    # Per-point positional std (L2 of deviation): [S, N] -> mean over S -> [N]
    per_sample_dist = np.linalg.norm(deviations, axis=-1)   # [S, N]
    per_point_std = per_sample_dist.mean(axis=0)             # [N]

    sorted_idx = np.argsort(per_point_std)

    return {
        "per_point_std": per_point_std,
        "mean_std": float(np.mean(per_point_std)),
        "median_std": float(np.median(per_point_std)),
        "max_std": float(np.max(per_point_std)),
        "min_std": float(np.min(per_point_std)),
        "worst_20_indices": sorted_idx[-20:].tolist(),
        "best_20_indices": sorted_idx[:20].tolist(),
    }


def assign_body_regions(template: np.ndarray) -> dict:
    """
    Partition template points into rough spatial regions based on position.
    Assumes normalised coordinates (centred, unit-sphere scaled).
    Returns dict mapping region_name → array of point indices.
    """
    y = template[:, 1]  # height axis
    x = template[:, 0]  # lateral axis

    y_sorted = np.sort(y)
    n = len(y)

    # Rough body segmentation by height percentiles
    regions = {}
    regions["head"] = np.where(y > np.percentile(y, 85))[0]
    regions["torso"] = np.where((y > np.percentile(y, 40)) & (y <= np.percentile(y, 85)))[0]

    lower = y <= np.percentile(y, 40)
    mid_band = (y > np.percentile(y, 40)) & (y <= np.percentile(y, 85))

    # Arms: lateral points in the torso band
    regions["left_arm"] = np.where(mid_band & (x < np.percentile(x[mid_band], 15)))[0]
    regions["right_arm"] = np.where(mid_band & (x > np.percentile(x[mid_band], 85)))[0]

    # Legs: lower body
    regions["left_leg"] = np.where(lower & (x < np.median(x[lower])))[0]
    regions["right_leg"] = np.where(lower & (x >= np.median(x[lower])))[0]

    return regions


def evaluate_results(results_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load template ----
    template_path = os.path.join(results_dir, "template.npy")
    if not os.path.exists(template_path):
        print(f"Error: {template_path} does not exist. Did test.py save it?")
        return
    template = np.load(template_path).astype(np.float32)        # [N, 3]
    template_t = torch.from_numpy(template).float().cuda()

    # ---- Find flowed / target pairs ----
    flowed_files = sorted(glob.glob(os.path.join(results_dir, "flowed_*.npy")))
    target_files = sorted(glob.glob(os.path.join(results_dir, "target_*.npy")))

    if len(flowed_files) == 0 or len(target_files) == 0:
        print(f"Error: No flowed or target .npy files found in {results_dir}")
        return
    if len(flowed_files) != len(target_files):
        print(f"Warning: {len(flowed_files)} flowed files but {len(target_files)} target files.")

    num_samples = min(len(flowed_files), len(target_files))

    print(f"Evaluating {num_samples} samples...")
    print("=" * 60)

    # ---- Compute per-sample metrics ----
    cd_before_list = []
    cd_after_list = []
    all_flowed = []

    for i in range(num_samples):
        flowed_np = np.load(flowed_files[i]).astype(np.float32)
        target_np = np.load(target_files[i]).astype(np.float32)

        flowed_t = torch.from_numpy(flowed_np).float().cuda()
        target_t = torch.from_numpy(target_np).float().cuda()

        cd_before = chamfer_distance(template_t, target_t)
        cd_after = chamfer_distance(flowed_t, target_t)

        cd_before_list.append(cd_before)
        cd_after_list.append(cd_after)
        all_flowed.append(flowed_np)

        if (i + 1) % 10 == 0 or i == num_samples - 1:
            print(f"  Processed {i+1}/{num_samples} shapes")

    # ---- Aggregate Chamfer metrics ----
    mean_before = np.mean(cd_before_list)
    std_before = np.std(cd_before_list)
    mean_after = np.mean(cd_after_list)
    std_after = np.std(cd_after_list)
    improvement = (1 - mean_after / max(mean_before, 1e-8)) * 100

    # ---- Correspondence consistency ----
    print("\nComputing correspondence consistency...")
    flowed_stack = np.stack(all_flowed, axis=0)                  # [S, N, 3]
    consistency = compute_correspondence_consistency(flowed_stack)

    # ---- Per-region consistency ----
    regions = assign_body_regions(template)
    region_stats = {}
    for name, indices in regions.items():
        if len(indices) > 0:
            region_std = consistency["per_point_std"][indices]
            region_stats[name] = {
                "n_points": int(len(indices)),
                "mean_std": float(np.mean(region_std)),
                "median_std": float(np.median(region_std)),
                "max_std": float(np.max(region_std)),
            }

    # ---- Build summary ----
    summary = {
        "num_evaluated": num_samples,
        "chamfer_before": {
            "mean": float(mean_before),
            "std": float(std_before),
        },
        "chamfer_after": {
            "mean": float(mean_after),
            "std": float(std_after),
        },
        "improvement_pct": float(improvement),
        "correspondence_consistency": {
            "mean_positional_std": consistency["mean_std"],
            "median_positional_std": consistency["median_std"],
            "max_positional_std": consistency["max_std"],
            "min_positional_std": consistency["min_std"],
        },
        "per_region_consistency": region_stats,
    }

    # ---- Save outputs ----
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    np.save(os.path.join(output_dir, "per_point_std.npy"), consistency["per_point_std"])

    with open(os.path.join(output_dir, "per_sample_chamfer.txt"), "w") as f:
        f.write("idx    CD_before    CD_after    improvement_%\n")
        f.write("-" * 50 + "\n")
        for i in range(num_samples):
            imp = (1 - cd_after_list[i] / max(cd_before_list[i], 1e-8)) * 100
            f.write(f"{i:04d}   {cd_before_list[i]:.6f}    {cd_after_list[i]:.6f}    {imp:+.1f}%\n")

    # ---- Print report ----
    print("\n" + "=" * 60)
    print("  ALIGNMENT EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Samples evaluated: {num_samples}")
    print(f"\n  Chamfer Distance (template → target — lower is better):")
    print(f"    BEFORE (raw template):   {mean_before:.6f} +/- {std_before:.6f}")
    print(f"    AFTER  (flowed):         {mean_after:.6f} +/- {std_after:.6f}")
    print(f"    Improvement:             {improvement:.1f}%")
    print(f"\n  Correspondence Consistency (lower std = more consistent):")
    print(f"    Mean per-point std:      {consistency['mean_std']:.6f}")
    print(f"    Median per-point std:    {consistency['median_std']:.6f}")
    print(f"    Range: [{consistency['min_std']:.6f}, {consistency['max_std']:.6f}]")
    print(f"\n  Per-Region Consistency:")
    for name, stats in sorted(region_stats.items()):
        print(f"    {name:12s}: mean_std={stats['mean_std']:.6f}  "
              f"({stats['n_points']} points)")
    print("=" * 60)
    print(f"  Saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing .npy outputs from test.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    args = parser.parse_args()

    evaluate_results(args.results_dir, args.output_dir)