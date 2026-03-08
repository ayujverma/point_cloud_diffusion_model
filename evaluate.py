"""
Evaluating Full-Resolution Point Correspondences
=================================================

This script analyzes the output of `test.py` to evaluate how well the
canonical template aligns with the target shapes. We measure metrics 
such as Chamfer Distance to evaluate the overall shape fidelity and alignment.

Usage
-----
::

    python scripts/evaluate.py \
        --results_dir results/full_res \
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
    # a->b
    diff_ab = a.unsqueeze(1) - b.unsqueeze(0)  # [N, M, 3]
    dist_ab = (diff_ab ** 2).sum(-1)           # [N, M]
    min_ab = dist_ab.min(dim=1).values.mean()  # scalar
    
    # b->a
    min_ba = dist_ab.min(dim=0).values.mean()
    
    return (min_ab + min_ba).item()

def evaluate_results(results_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if a template is saved
    template_path = os.path.join(results_dir, "template.npy")
    if not os.path.exists(template_path):
        print(f"Error: {template_path} does not exist. Did test.py save it?")
        return
        
    template = torch.from_numpy(np.load(template_path)).float().cuda()
    
    # Find all flowed and target pairs
    flowed_files = sorted(glob.glob(os.path.join(results_dir, "flowed_*.npy")))
    target_files = sorted(glob.glob(os.path.join(results_dir, "target_*.npy")))
    
    if len(flowed_files) == 0 or len(target_files) == 0:
        print(f"Error: No flowed or target `.npy` files found in {results_dir}")
        return
        
    if len(flowed_files) != len(target_files):
        print(f"Warning: Found {len(flowed_files)} flowed files but {len(target_files)} target files.")
    
    num_samples = min(len(flowed_files), len(target_files))
    
    print(f"Evaluating {num_samples} samples...")
    print("=" * 50)
    
    metrics = {
        "chamfer_distances": [],
        "best_idx": -1,
        "best_cd": float('inf'),
        "worst_idx": -1,
        "worst_cd": -1.0,
    }
    
    for i in range(num_samples):
        flow_path = flowed_files[i]
        tgt_path = target_files[i]
        
        flowed_np = np.load(flow_path)
        target_np = np.load(tgt_path)
        
        flowed = torch.from_numpy(flowed_np).float().cuda()
        target = torch.from_numpy(target_np).float().cuda()
        
        cd = chamfer_distance(flowed, target)
        metrics["chamfer_distances"].append(cd)
        
        if cd < metrics["best_cd"]:
            metrics["best_cd"] = cd
            metrics["best_idx"] = i
            
        if cd > metrics["worst_cd"]:
            metrics["worst_cd"] = cd
            metrics["worst_idx"] = i
            
        if (i+1) % 10 == 0 or i == num_samples - 1:
            print(f"  Processed {i+1}/{num_samples} shapes")
            
    # Compile Summary
    mean_cd = np.mean(metrics["chamfer_distances"])
    std_cd = np.std(metrics["chamfer_distances"])
    
    summary = {
        "num_evaluated": num_samples,
        "chamfer_mean": float(mean_cd),
        "chamfer_std": float(std_cd),
        "best_sample": {
            "index": metrics["best_idx"],
            "chamfer": float(metrics["best_cd"])
        },
        "worst_sample": {
            "index": metrics["worst_idx"],
            "chamfer": float(metrics["worst_cd"])
        }
    }
    
    # Save the full results and summary
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    with open(os.path.join(output_dir, "all_chamfer_dists.txt"), "w") as f:
        for i, cd in enumerate(metrics["chamfer_distances"]):
            f.write(f"{i:04d}: {cd:.6f}\n")
            
    print("=" * 50)
    print(f"Mean Chamfer Distance: {mean_cd:.6f} ± {std_cd:.6f}")
    print(f"Best shape CD: {metrics['best_cd']:.6f} (Idx: {metrics['best_idx']})")
    print(f"Worst shape CD: {metrics['worst_cd']:.6f} (Idx: {metrics['worst_idx']})")
    print("=" * 50)
    print(f"Evaluation metrics saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing .npy outputs from test.py")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    args = parser.parse_args()
    
    evaluate_results(args.results_dir, args.output_dir)