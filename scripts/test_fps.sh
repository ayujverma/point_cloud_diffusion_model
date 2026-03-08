#!/bin/bash
#===========================================================================
# Local Test Script — Farthest Point Sampling (FPS)
#===========================================================================
#
# Runs the FPS downsampling locally on a sample file.
# Since it does not require GPUs or TACC job dispatchers, this script
# runs as a direct Python execution.
#
# Usage:
#   bash scripts/test_fps.sh [path_to_npy_file] [n_samples...]
#
# If no file is provided, it defaults to: data/human/test/human_000004.npy
# By default, it will sample to 1024 and 2048 points if no n_samples are provided.
#===========================================================================

# Use the provided file or default to the test dataset sample
INPUT_FILE="${1:-data/human/test/human_000004.npy}"
shift || true
N_SAMPLES="${@:-1024 2048}"
OUTPUT_DIR="results/fps"

echo "=========================================="
echo " TESTING — local Farthest Point Sampling "
echo "=========================================="
echo " Input file:   $INPUT_FILE"
echo " Points:       $N_SAMPLES"
echo " Output dir:   $OUTPUT_DIR"
echo "=========================================="

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found at $INPUT_FILE"
    exit 1
fi

export PYTHONPATH="."
python core/point_ops.py "$INPUT_FILE" --n_samples $N_SAMPLES --output_dir "$OUTPUT_DIR"

echo "=========================================="
echo " Test completed."
echo "=========================================="
