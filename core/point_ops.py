"""
Point Cloud Geometric Operations
==================================

GPU-accelerated utilities for point cloud processing:

- **Farthest Point Sampling (FPS)**: Greedily select a maximally-spread subset
  of N points.  Used during training to subsample both the template and target
  from 15k → 2048 points while preserving surface coverage.

- **K-Nearest Neighbours (KNN)**: For each query point, find its K closest
  neighbours in a reference set.  Used to build the local attention graph in
  VNKNNAttention, reducing memory from O(N²) to O(N·K).

Both routines are pure PyTorch (no C++/CUDA extensions) so they work on any
GPU without a custom build step.  For extreme point counts (>50k), consider
swapping in `torch_cluster.fps` / `torch_cluster.knn` for 2-3× speed-ups.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def farthest_point_sample(
    points: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS).

    Greedy algorithm: start from a random seed point, then iteratively pick
    the point that is farthest from the already-selected set.  This yields
    a subset with excellent surface coverage.

    Parameters
    ----------
    points : [B, N, 3]
        Input point clouds.
    n_samples : int
        Number of points to select (must be ≤ N).

    Returns
    -------
    indices : [B, n_samples]  (LongTensor)
        Indices into the N dimension of `points`.
    """
    B, N, _ = points.shape
    device = points.device

    assert n_samples <= N, f"Cannot sample {n_samples} from {N} points"

    # Output index buffer
    indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)

    # Distance from each point to the nearest selected point (initialised to ∞)
    dists = torch.full((B, N), float("inf"), device=device)

    # Random seed point per batch element
    seed = torch.randint(0, N, (B,), device=device)
    indices[:, 0] = seed

    # Gather seed coordinates: [B, 3]
    batch_idx = torch.arange(B, device=device)
    current = points[batch_idx, seed]                            # [B, 3]

    for i in range(1, n_samples):
        # Squared distance from `current` to every point: [B, N]
        diff = points - current.unsqueeze(1)                     # [B, N, 3]
        sq_dist = (diff * diff).sum(dim=-1)                      # [B, N]

        # Update running min-distance to selected set
        dists = torch.min(dists, sq_dist)

        # Pick the point with the largest min-distance (farthest)
        farthest = dists.argmax(dim=-1)                          # [B]
        indices[:, i] = farthest

        # Move current pointer
        current = points[batch_idx, farthest]                    # [B, 3]

    return indices


def fps_gather(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather points by FPS indices.

    Parameters
    ----------
    points  : [B, N, D]   (D can be 3 for xyz, or 3×C for VN features)
    indices : [B, S]       (from farthest_point_sample)

    Returns
    -------
    sampled : [B, S, D]
    """
    B, S = indices.shape
    # Expand indices to match last dim
    idx = indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])  # [B, S, D]
    return torch.gather(points, 1, idx)


# ---------------------------------------------------------------------------
# K-Nearest Neighbours
# ---------------------------------------------------------------------------

def knn(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Find K nearest neighbours in `reference` for each point in `query`.

    Uses brute-force pairwise distances (efficient up to ~15k points on A100).

    Parameters
    ----------
    query     : [B, N, 3]
        Points to find neighbours for.
    reference : [B, M, 3]
        Points to search in (often same as query for self-attention).
    k : int
        Number of neighbours.

    Returns
    -------
    knn_idx : [B, N, K]  (LongTensor)
        Indices into the M dimension of `reference`.
    """
    # Pairwise squared distances: [B, N, M]
    # ||q_i - r_j||² = ||q_i||² + ||r_j||² - 2 q_i · r_j
    q_sq = (query * query).sum(dim=-1, keepdim=True)             # [B, N, 1]
    r_sq = (reference * reference).sum(dim=-1, keepdim=True)     # [B, M, 1]
    qr = torch.bmm(query, reference.transpose(1, 2))            # [B, N, M]
    dist_sq = q_sq + r_sq.transpose(1, 2) - 2 * qr              # [B, N, M]

    # Top-K smallest distances
    _, knn_idx = dist_sq.topk(k, dim=-1, largest=False)          # [B, N, K]
    return knn_idx


def knn_gather(
    features: torch.Tensor,
    knn_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Gather features for KNN indices.

    Parameters
    ----------
    features : [B, M, D]
        Feature tensor to gather from (D can be any trailing dims).
    knn_idx  : [B, N, K]
        Neighbour indices.

    Returns
    -------
    gathered : [B, N, K, D]
    """
    B, M, D = features.shape
    _, N, K = knn_idx.shape
    # Expand knn_idx to [B, N, K, D] for gather along dim=1
    idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)   # [B, N, K, D]
    # Expand features to [B, N_repeat, M, D] is wasteful — use gather instead
    # We need to reshape features so gather works on dim=1
    features_expanded = features.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, D]
    return torch.gather(features_expanded, 2, idx_expanded)      # [B, N, K, D]


def knn_gather_vn(
    features: torch.Tensor,
    knn_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Gather VN features ([B, M, 3, C]) for KNN indices.

    Parameters
    ----------
    features : [B, M, 3, C]
    knn_idx  : [B, N, K]

    Returns
    -------
    gathered : [B, N, K, 3, C]
    """
    B, M, three, C = features.shape
    _, N, K = knn_idx.shape
    # Flatten spatial + channel: [B, M, 3*C]
    flat = features.reshape(B, M, three * C)
    gathered_flat = knn_gather(flat, knn_idx)                    # [B, N, K, 3*C]
    return gathered_flat.reshape(B, N, K, three, C)
