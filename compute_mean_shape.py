"""
Compute the mean training shape for template initialization.

This script loads all training point clouds, normalizes each to the unit
sphere, **aligns them via optimal transport to a reference shape**, and
averages them to produce a mean shape for template initialization.

Why alignment matters
---------------------
Point clouds are unordered sets — there's no natural pairing between point i
in shape A and point j in shape B.  Simply sorting by (x,y,z) coordinates
(lexicographic) blends unrelated body parts when averaging.  Instead, we:

1. Pick a reference shape (the first training sample).
2. For each other shape, compute the optimal 1-to-1 assignment to the
   reference via the Hungarian algorithm (exact) or greedy nearest-neighbor
   (approximate, for large N).
3. Reorder each shape's points to match the reference ordering.
4. Average all aligned shapes → proper mean human body.

Usage
-----
    python compute_mean_shape.py \\
        --data_root data/human \\
        --n_points 2048 \\
        --output data/mean_shape_2048.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def normalize_to_unit_sphere(pts: np.ndarray) -> np.ndarray:
    """Centre at origin and scale so max norm = 1."""
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-8:
        pts = pts / max_dist
    return pts


def fps_numpy(pts: np.ndarray, n: int) -> np.ndarray:
    """Farthest point sampling (greedy) on CPU."""
    N = pts.shape[0]
    if N <= n:
        return pts[:n]
    selected = [np.random.randint(N)]
    dists = np.full(N, np.inf)
    for _ in range(n - 1):
        last = pts[selected[-1]]
        d = np.sum((pts - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected.append(np.argmax(dists))
    return pts[np.array(selected)]


def align_hungarian(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align `source` to `reference` via the Hungarian algorithm (exact OT).

    Both inputs must be [N, 3] with the same N.
    Returns reordered source where source_aligned[i] ↔ reference[i].

    Complexity: O(N^3).  Use for N ≤ ~4096.
    """
    # Pairwise squared distances: [N, N]
    a_sq = np.sum(source ** 2, axis=1, keepdims=True)       # [N, 1]
    b_sq = np.sum(reference ** 2, axis=1, keepdims=True).T  # [1, N]
    ab = source @ reference.T                                # [N, N]
    cost = np.maximum(a_sq + b_sq - 2 * ab, 0.0)

    # Hungarian: find optimal 1-to-1 assignment minimising total cost
    row_ind, col_ind = linear_sum_assignment(cost)

    # Reorder: source_aligned[ref_idx] = source[src_idx]
    source_aligned = np.empty_like(source)
    source_aligned[row_ind] = source[col_ind]
    return source_aligned


def align_nearest_neighbor(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Fast approximate alignment using greedy nearest-neighbor with deconfliction.

    For each reference point (processed closest-first), assign the nearest
    unused source point.  O(N·K) with KD-tree queries where K=10.

    Use for N > ~4096 where Hungarian is too slow.
    """
    N = source.shape[0]
    K = min(20, N)

    # Build KD-tree on source points
    src_tree = cKDTree(source)

    # For each reference point, query K nearest source neighbours
    dists, indices = src_tree.query(reference, k=K)
    if K == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    # Greedy assignment: process reference points by closest distance first
    best_dists = dists[:, 0]
    order = np.argsort(best_dists)

    used = np.zeros(N, dtype=bool)
    assignment = np.full(N, -1, dtype=int)

    for ref_idx in order:
        assigned = False
        for k in range(K):
            src_idx = indices[ref_idx, k]
            if not used[src_idx]:
                assignment[ref_idx] = src_idx
                used[src_idx] = True
                assigned = True
                break
        if not assigned:
            # All K neighbours used — find any nearest unused source point
            remaining = np.where(~used)[0]
            if len(remaining) > 0:
                remaining_dists = np.sum(
                    (source[remaining] - reference[ref_idx]) ** 2, axis=1
                )
                best = remaining[np.argmin(remaining_dists)]
                assignment[ref_idx] = best
                used[best] = True

    return source[assignment]


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean training shape for template initialization."
    )
    parser.add_argument("--data_root", type=str, default="data/human",
                        help="Path to data directory containing train/ folder.")
    parser.add_argument("--n_points", type=int, default=2048,
                        help="Number of points per shape (FPS subsample).")
    parser.add_argument("--output", type=str, default="data/mean_shape.npy",
                        help="Output .npy file path.")
    parser.add_argument("--max_shapes", type=int, default=-1,
                        help="Max shapes to use (-1 = all). Fewer = faster.")
    parser.add_argument("--hungarian_limit", type=int, default=4096,
                        help="Use Hungarian for N <= this, nearest-neighbor above.")
    args = parser.parse_args()

    train_dir = Path(args.data_root) / "train"
    files = sorted(list(train_dir.glob("*.npy")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files in {train_dir}")

    if 0 < args.max_shapes < len(files):
        files = files[:args.max_shapes]

    use_hungarian = args.n_points <= args.hungarian_limit
    method_name = "Hungarian (exact)" if use_hungarian else "Nearest-neighbor (greedy)"

    print(f"Found {len(files)} training shapes in {train_dir}")
    print(f"Subsampling each to {args.n_points} points via FPS")
    print(f"Alignment method: {method_name}")

    # --- Load and normalize all shapes ---
    np.random.seed(42)  # reproducible FPS
    all_shapes = []
    for i, f in enumerate(files):
        pts = np.load(f).astype(np.float32)
        pts = normalize_to_unit_sphere(pts)
        pts = fps_numpy(pts, args.n_points)
        all_shapes.append(pts)
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(files)}")

    print(f"Loaded {len(all_shapes)} shapes, each ({args.n_points}, 3)")

    # --- Pick reference shape (first one) ---
    reference = all_shapes[0].copy()
    print(f"Reference shape: {files[0].name}")

    # --- Align all shapes to reference ---
    align_fn = align_hungarian if use_hungarian else align_nearest_neighbor
    aligned_shapes = [reference]  # reference is already aligned to itself

    for i in range(1, len(all_shapes)):
        aligned = align_fn(all_shapes[i], reference)
        aligned_shapes.append(aligned)
        if (i + 1) % 50 == 0 or i == len(all_shapes) - 1:
            print(f"  Aligned {i + 1}/{len(all_shapes)}")

    # --- Average ---
    aligned_stack = np.stack(aligned_shapes, axis=0)  # [K, N, 3]
    mean_shape = aligned_stack.mean(axis=0)            # [N, 3]

    # Re-normalize to unit sphere
    mean_shape = normalize_to_unit_sphere(mean_shape)

    np.save(args.output, mean_shape)
    print(f"\nMean shape saved to {args.output}  (shape: {mean_shape.shape})")

    # --- Quick quality check ---
    bbox_min = mean_shape.min(axis=0)
    bbox_max = mean_shape.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"Bounding box: min={np.array2string(bbox_min, precision=3)}, "
          f"max={np.array2string(bbox_max, precision=3)}")
    print(f"Bounding box size: {np.array2string(bbox_size, precision=3)}")


if __name__ == "__main__":
    main()
