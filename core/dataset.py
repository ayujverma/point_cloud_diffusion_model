"""
Point Cloud Dataset for Human.PC15k
=====================================

Loads .npy point cloud files from the Human.PC15k directory structure::

    data/human/
    ├── train/   ← .npy files  (each [N, 3])
    ├── val/
    └── test/

Each .npy file contains a single point cloud of shape (N, 3).

Subsampling strategy
--------------------
When ``n_points > 0`` and the raw cloud has more points, we use Farthest
Point Sampling (FPS) to select a well-distributed subset.  FPS is performed
on CPU (numpy) at data-load time so the full-resolution cloud never reaches
the GPU.  Each epoch uses a different random seed point for FPS, providing
mild data augmentation.

DDP-aware features
------------------
- Optional fast I/O path: copies data to a local scratch directory (e.g.
  /tmp on TACC) before training starts, so each node reads from local SSD
  instead of the shared filesystem.
- Compatible with DistributedSampler.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# CPU-based FPS for use in dataset __getitem__
# ---------------------------------------------------------------------------

def _fps_numpy(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Farthest Point Sampling on CPU (numpy).

    Selects `n` points from `pts` [M, 3] with maximal spatial coverage.
    Uses a random seed point each call for data augmentation.

    Returns
    -------
    sampled : [n, 3]
    """
    M = pts.shape[0]
    if M <= n:
        # Not enough points — resample with replacement to pad
        extra_idx = np.random.choice(M, n - M, replace=True)
        return np.concatenate([pts, pts[extra_idx]], axis=0)

    # Random seed point
    selected = [np.random.randint(M)]
    dists = np.full(M, np.inf, dtype=np.float32)

    for _ in range(n - 1):
        last = pts[selected[-1]]
        d = np.sum((pts - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))

    return pts[np.array(selected)]


class PointCloudDataset(Dataset):
    """
    Dataset that loads .npy point cloud files.

    Parameters
    ----------
    root : str or Path
        Path to the split folder (e.g. ``data/human/train``).
    n_points : int
        Number of points to subsample from each cloud via FPS.
        Set to ``-1`` to return **all** points (for full-resolution use);
        in that case the DataLoader must use ``batch_size=1``.
    normalise : bool
        If True, centre each cloud at origin and scale to unit sphere.
    local_scratch : str or None
        If provided, copy data here for fast local I/O (useful on HPC).
        Example: ``/tmp/human_train``.
    """

    def __init__(
        self,
        root: str | Path,
        n_points: int = 2048,
        normalise: bool = True,
        local_scratch: Optional[str] = None,
    ):
        self.n_points = n_points
        self.normalise = normalise
        self.root = Path(root)

        # Optionally stage data to local scratch ---
        if local_scratch is not None:
            scratch = Path(local_scratch)
            if not scratch.exists():
                print(f"[Dataset] Copying data to local scratch: {scratch}")
                shutil.copytree(self.root, scratch)
            self.root = scratch

        # Discover all .npy files ---
        self.files = sorted(list(self.root.glob("*.npy")))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.root}. "
                "Check that your data directory is correct."
            )
        print(f"[Dataset] Found {len(self.files)} point clouds in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns
        -------
        points : torch.Tensor  [n_points, 3]  (or [M, 3] if n_points == -1)
            A single point cloud, normalised and FPS-subsampled.
        """
        pts = np.load(self.files[idx]).astype(np.float32)        # [M, 3]

        # Normalise FIRST (before FPS) so FPS operates on the normalised cloud
        if self.normalise:
            centroid = pts.mean(axis=0)
            pts = pts - centroid
            max_dist = np.max(np.linalg.norm(pts, axis=1))
            if max_dist > 1e-8:
                pts = pts / max_dist

        # FPS subsample to n_points (unless -1 = use all points)
        if self.n_points > 0 and pts.shape[0] > self.n_points:
            pts = _fps_numpy(pts, self.n_points)
        elif self.n_points > 0 and pts.shape[0] < self.n_points:
            # Pad with resampled points if too few
            extra_idx = np.random.choice(pts.shape[0], self.n_points - pts.shape[0], replace=True)
            pts = np.concatenate([pts, pts[extra_idx]], axis=0)

        return torch.from_numpy(pts)                             # [N, 3]


def build_dataloaders(
    data_root: str,
    n_points: int = 2048,
    batch_size: int = 32,
    num_workers: int = 4,
    local_scratch: Optional[str] = None,
    distributed: bool = False,
):
    """
    Convenience function to build train / val / test DataLoaders.

    Parameters
    ----------
    data_root : str
        Path to the parent directory containing train/, val/, test/ folders.
    distributed : bool
        If True, wrap with DistributedSampler for DDP.

    Returns
    -------
    dict  with keys 'train', 'val', 'test' mapping to DataLoaders.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"[Dataset] Warning: {split_dir} not found, skipping.")
            continue

        scratch = None
        if local_scratch is not None:
            scratch = os.path.join(local_scratch, split)

        ds = PointCloudDataset(
            root=split_dir,
            n_points=n_points,
            normalise=True,
            local_scratch=scratch,
        )

        sampler = DistributedSampler(ds, shuffle=(split == "train")) if distributed else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders
