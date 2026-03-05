"""
Test farthest_point_sample on human_000000.npy, outputting
FPS-downsampled clouds of 1024 and 2048 points.
"""

import sys
import pathlib
import numpy as np
import torch

# Allow imports from the project root
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from core.point_ops import farthest_point_sample, fps_gather

INPUT  = pathlib.Path("data/human/train/human_000000.npy")
OUT_1024 = pathlib.Path("human_000000_fps1024.npy")
OUT_2048 = pathlib.Path("human_000000_fps2048.npy")


def main() -> None:
    raw = np.load(INPUT)          # expected shape: [N, 3]
    print(f"Loaded {INPUT}  shape={raw.shape}")

    # farthest_point_sample expects [B, N, 3]
    pts = torch.from_numpy(raw).float().unsqueeze(0)   # [1, N, 3]

    for n_samples, out_path in [(1024, OUT_1024), (2048, OUT_2048)]:
        idx      = farthest_point_sample(pts, n_samples)          # [1, n_samples]
        sampled  = fps_gather(pts, idx)                           # [1, n_samples, 3]
        result   = sampled.squeeze(0).numpy()                     # [n_samples, 3]
        np.save(out_path, result)
        print(f"Saved {n_samples}-point cloud → {out_path}  shape={result.shape}")


if __name__ == "__main__":
    main()
