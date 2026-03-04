"""
Point Cloud Dataset for Human.PC15k
=====================================

Loads .npy point cloud files from the Human.PC15k directory structure::

    data/human/
    ├── train/   ← .npy files  (each [N, 3])
    ├── val/
    └── test/

Each .npy file contains a single point cloud of shape (N, 3).

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


class PointCloudDataset(Dataset):
    """
    Dataset that loads .npy point cloud files.

    Parameters
    ----------
    root : str or Path
        Path to the split folder (e.g. ``data/human/train``).
    n_points : int
        Number of points to (sub)sample from each cloud.  If the raw cloud
        has more points, we randomly subsample; if fewer, we resample with
        replacement.  Set to ``-1`` to return **all** points (for full-
        resolution inference); in this case the DataLoader must use
        ``batch_size=1`` or a custom collate function.
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
            A single point cloud, optionally normalised.
        """
        pts = np.load(self.files[idx]).astype(np.float32)        # [M, 3]

        # (Sub)sample to exactly n_points, unless -1 (full resolution)
        if self.n_points > 0:
            M = pts.shape[0]
            if M >= self.n_points:
                choice = np.random.choice(M, self.n_points, replace=False)
            else:
                choice = np.random.choice(M, self.n_points, replace=True)
            pts = pts[choice]

        # Normalise: centre + scale to unit sphere
        if self.normalise:
            centroid = pts.mean(axis=0)
            pts = pts - centroid
            max_dist = np.max(np.linalg.norm(pts, axis=1))
            if max_dist > 1e-8:
                pts = pts / max_dist

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
