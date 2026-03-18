#!/bin/bash
#===========================================================================
# Local Test Script — Mean Shape Computation
#===========================================================================
#
# Runs compute_mean_shape.py locally on a small subset of training data
# to verify the script works end-to-end.
#
# Usage:
#   bash scripts/test_mean_shape.sh
#
# Uses 5 real training point clouds from data/human/train, subsampled
# to 2048 points via FPS, aligned with Hungarian OT, and averaged.
#===========================================================================

# Change to repo directory
REPO_DIR="/work/10692/ayuj/point_cloud_diffusion_model"
cd "${REPO_DIR}"

DATA_ROOT="$REPO_DIR/data/human"
OUTPUT_DIR="$REPO_DIR/results/mean_shape"
OUTPUT_FILE="${OUTPUT_DIR}/mean_shape_test.npy"
N_POINTS=2048
MAX_SHAPES=4000

echo "=========================================="
echo " TESTING — local Mean Shape Computation  "
echo "=========================================="
echo " Data root:  $DATA_ROOT"
echo " N points:   $N_POINTS"
echo " Max shapes: $MAX_SHAPES"
echo " Output:     $OUTPUT_FILE"
echo "=========================================="

# Check if training data exists
if [ ! -d "${DATA_ROOT}/train" ]; then
    echo "ERROR: Training data not found at ${DATA_ROOT}/train"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="."
python compute_mean_shape.py \
    --data_root "$DATA_ROOT" \
    --n_points $N_POINTS \
    --output "$OUTPUT_FILE" \
    --max_shapes $MAX_SHAPES \
    --hungarian_limit 4096

# Quick sanity check on the output
if [ -f "$OUTPUT_FILE" ]; then
    python -c "
import numpy as np
arr = np.load('${OUTPUT_FILE}')
norms = np.linalg.norm(arr, axis=1)
print(f'  Shape:    {arr.shape}')
print(f'  Max norm: {norms.max():.4f}')
assert arr.shape == (${N_POINTS}, 3), f'Expected (${N_POINTS}, 3), got {arr.shape}'
assert norms.max() <= 1.01, f'Max norm {norms.max():.4f} > 1.01'
print('  Checks passed.')
"
else
    echo "ERROR: Output file not created"
    exit 1
fi

echo "=========================================="
echo " Test completed."
echo "=========================================="
