"""
Select the most central training shape for template initialization.

Instead of trying to align and average unordered point clouds (which produces
blobs when poses vary), we pick the single training shape that has the lowest
average Chamfer distance to a random subset of other shapes.  This gives us
a clean, human-shaped template that is geometrically "central" in the dataset.

The template is a learnable parameter — it just needs a good starting point.

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


def chamfer_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Bidirectional Chamfer distance (numpy, CPU).
    a, b: [N, 3]
    """
    # a→b: for each point in a, squared distance to closest in b
    # Use chunked computation to avoid huge [N, M] matrices
    N = a.shape[0]
    M = b.shape[0]
    chunk = 512

    # a→b
    min_ab = np.zeros(N)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        diff = a[i:end, None, :] - b[None, :, :]  # [chunk, M, 3]
        dists = np.sum(diff ** 2, axis=-1)          # [chunk, M]
        min_ab[i:end] = dists.min(axis=1)

    # b→a
    min_ba = np.zeros(M)
    for i in range(0, M, chunk):
        end = min(i + chunk, M)
        diff = b[i:end, None, :] - a[None, :, :]
        dists = np.sum(diff ** 2, axis=-1)
        min_ba[i:end] = dists.min(axis=1)

    return min_ab.mean() + min_ba.mean()


def main():
    parser = argparse.ArgumentParser(
        description="Select most central training shape for template initialization."
    )
    parser.add_argument("--data_root", type=str, default="data/human",
                        help="Path to data directory containing train/ folder.")
    parser.add_argument("--n_points", type=int, default=2048,
                        help="Number of points per shape (FPS subsample).")
    parser.add_argument("--output", type=str, default="data/mean_shape.npy",
                        help="Output .npy file path.")
    parser.add_argument("--max_shapes", type=int, default=-1,
                        help="Max shapes to load (-1 = all).")
    parser.add_argument("--n_probe", type=int, default=100,
                        help="Number of random shapes to compare against when "
                             "finding the most central shape. Higher = more "
                             "accurate but slower.")
    args = parser.parse_args()

    train_dir = Path(args.data_root) / "train"
    files = sorted(list(train_dir.glob("*.npy")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files in {train_dir}")

    if 0 < args.max_shapes < len(files):
        files = files[:args.max_shapes]

    print(f"Found {len(files)} training shapes in {train_dir}")
    print(f"Subsampling each to {args.n_points} points via FPS")

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

    # --- Find most central shape ---
    # Compare each shape against a random probe set to find the one
    # with lowest average Chamfer distance (= most "central")
    n_total = len(all_shapes)
    n_probe = min(args.n_probe, n_total)

    probe_indices = np.random.choice(n_total, n_probe, replace=False)
    probe_shapes = [all_shapes[i] for i in probe_indices]

    print(f"\nComputing centrality scores against {n_probe} probe shapes...")
    best_idx = 0
    best_score = float("inf")

    for i in range(n_total):
        total_cd = 0.0
        for probe in probe_shapes:
            total_cd += chamfer_distance_np(all_shapes[i], probe)
        avg_cd = total_cd / n_probe

        if avg_cd < best_score:
            best_score = avg_cd
            best_idx = i

        if (i + 1) % 100 == 0 or i == n_total - 1:
            print(f"  Evaluated {i + 1}/{n_total} | "
                  f"Current best: shape {best_idx} (avg CD = {best_score:.6f})")

    # --- Save the most central shape as template ---
    template = all_shapes[best_idx]

    # Re-normalize just in case
    template = normalize_to_unit_sphere(template)

    np.save(args.output, template)
    print(f"\nTemplate saved to {args.output}  (shape: {template.shape})")
    print(f"Selected shape index: {best_idx}  ({files[best_idx].name})")
    print(f"Average Chamfer distance to probe set: {best_score:.6f}")

    # --- Quick quality check ---
    bbox_min = template.min(axis=0)
    bbox_max = template.max(axis=0)
    bbox_size = bbox_max - bbox_min
    print(f"Bounding box: min={np.array2string(bbox_min, precision=3)}, "
          f"max={np.array2string(bbox_max, precision=3)}")
    print(f"Bounding box size: {np.array2string(bbox_size, precision=3)}")


if __name__ == "__main__":
    main()
