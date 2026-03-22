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

import argparse
from pathlib import Path
import time

import numpy as np
import torch


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
    probe_shapes = np.stack([all_shapes[i] for i in probe_indices])  # [n_probe, N, 3]
    all_shapes_tensor = np.stack(all_shapes)  # [n_total, N, 3]

    print(f"\nComputing centrality scores against {n_probe} probe shapes...")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for Chamfer Distance calculations")

    # Move data to PyTorch
    shapes_pt = torch.from_numpy(all_shapes_tensor).to(device)  # [5000, 2048, 3]
    probes_pt = torch.from_numpy(probe_shapes).to(device)       # [100, 2048, 3]
    
    # We want average CD over probes for each shape.
    # We can batch over the probes to avoid MemoryError if running all 5000 x 100 at once.
    total_cd = torch.zeros(n_total, device=device)
    
    # Batch size for evaluating probes against all shapes
    # 5000 shapes * 1 probe = 5000 pairs of (2048x3). Fits easily in GPU.
    batch_size = 128
    
    for p_idx in range(n_probe):
        probe = probes_pt[p_idx:p_idx+1]  # [1, 2048, 3]
        
        for i in range(0, n_total, batch_size):
            batch = shapes_pt[i:i+batch_size]  # [B, 2048, 3]
            
            # cdist: [B, N, M] where N=2048, M=2048
            dists = torch.cdist(batch, probe)  # [B, 2048, 2048]
            
            # min distances
            min_a_to_b, _ = dists.min(dim=2)  # [B, 2048]
            min_b_to_a, _ = dists.min(dim=1)  # [B, 2048]
            
            # chamfer: squared distance usually, but since the dists is L2 norm, 
            # we need squared distance to match original CD definition
            # Actually, the original CD used `diff ** 2`. `torch.cdist` returns sqrt(sum(diff**2)).
            # So we square it:
            sq_dists = dists ** 2
            
            min_a_to_b_sq, _ = sq_dists.min(dim=2)
            min_b_to_a_sq, _ = sq_dists.min(dim=1)
            
            cd = min_a_to_b_sq.mean(dim=1) + min_b_to_a_sq.mean(dim=1)  # [B]
            total_cd[i:i+batch_size] += cd

    avg_cd = total_cd / n_probe  # [5000]
    
    best_idx = torch.argmin(avg_cd).item()
    best_score = avg_cd[best_idx].item()
    
    end_time = time.time()
    print(f"Alignment completed in {end_time - start_time:.2f} seconds.")

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
