"""
Split a dataset .npy file into train/val/test folders.

Takes a single .npy file of shape (M, P, 3) — M point clouds, each with
P points and 3 coordinates — and saves each cloud as an individual .npy
file under data/human/{train,val,test}/.

No resampling is done here.  FPS downsampling (15k → 2048) happens
during training inside the model.

Usage::

    python convert_to_npy.py -i dataset.npy
    python convert_to_npy.py -i dataset.npy -o data/human --train-frac 0.8
"""

import os
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Split .npy dataset into train/val/test folders")
    p.add_argument("--input", "-i", required=True,
                   help="Input .npy file of shape (M, P, 3)")
    p.add_argument("--output", "-o", default="data/human",
                   help="Output root folder (default: data/human)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    args = p.parse_args()

    # --- Validate fractions ---
    fracs = [args.train_frac, args.val_frac, args.test_frac]
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {sum(fracs):.4f}")

    # --- Load ---
    arr = np.load(args.input)
    # Handle (M, 3, P) → (M, P, 3)
    if arr.ndim == 3 and arr.shape[1] == 3 and arr.shape[2] != 3:
        arr = arr.transpose(0, 2, 1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected shape (M, P, 3), got {arr.shape}")

    M = arr.shape[0]
    print(f"Loaded {M} point clouds, each {arr.shape[1]} points")

    # --- Shuffle and split ---
    rng = np.random.RandomState(args.seed)
    idxs = np.arange(M)
    rng.shuffle(idxs)

    n_train = int(M * args.train_frac)
    n_val = int(M * args.val_frac)

    splits = {
        "train": idxs[:n_train],
        "val":   idxs[n_train:n_train + n_val],
        "test":  idxs[n_train + n_val:],
    }

    # --- Save ---
    for split_name, split_idxs in splits.items():
        out_dir = os.path.join(args.output, split_name)
        os.makedirs(out_dir, exist_ok=True)
        for i in split_idxs:
            np.save(os.path.join(out_dir, f"human_{i:06d}.npy"), arr[i])
        print(f"  {split_name}: {len(split_idxs)} files → {out_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()